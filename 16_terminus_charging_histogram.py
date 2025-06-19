#! /usr/bin/env python3


import logging
import os
import warnings
from typing import Dict, Tuple

import pandas as pd
import seaborn as sns
import sqlalchemy.orm.session
from eflips.eval.output.prepare import power_and_occupancy
from eflips.model import *
from eflips.model import ConsistencyWarning
from matplotlib import pyplot as plt
from sqlalchemy import create_engine, or_
from sqlalchemy.orm import Session
from tqdm import tqdm

import plotutils
from util import (
    update_database_to_most_recent_schema,
)
import numpy as np

USE_SIMBA_CONSUMPTION = True
warnings.simplefilter("ignore", ConsistencyWarning)

from util import START_OF_SIMULATION, END_OF_SIMULATION

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # Set log lvel to debug
    logging.basicConfig(level=logging.INFO)

    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    # Update the database schema
    update_database_to_most_recent_schema()

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # These are the scenarios where we actually need station electrification
    SCENARIO_NAMES = ["OU", "TERM"]

    dfs: Dict[str, pd.DataFrame] = {}

    peak_powers_by_scenario: Dict[str, Dict[str, float]] = {}

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).one()
        )

        # Now, find all the places where charging did happen
        charging_stations = (
            session.query(Station)
            .filter(
                Station.scenario_id == scenario.id,
                Station.charge_type == ChargeType.OPPORTUNITY,
            )
            .all()
        )
        vehicle_kms_by_station: Dict[str, Tuple[float, bool]] = {}
        peak_powers_by_station: Dict[str, float] = {}
        for station in tqdm(charging_stations):
            rotations_part_of_rots_visiting_station = (
                session.query(Rotation)
                .join(Trip)
                .join(Route)
                .filter(Rotation.allow_opportunity_charging == True)
                .filter(
                    or_(
                        Route.departure_station == station,
                        Route.arrival_station == station,
                    )
                )
                .distinct()
                .options(
                    sqlalchemy.orm.joinedload(Rotation.trips).joinedload(Trip.route)
                )
                .all()
            )
            vehicle_kms_by_station[station.name_short] = (
                (
                    sum(
                        [
                            sum([trip.route.distance for trip in rotation.trips])
                            for rotation in rotations_part_of_rots_visiting_station
                        ]
                    )
                    / 1000
                ),
                False,
            )

            # peak power
            occupancy_df = power_and_occupancy(
                area_id=None,
                session=session,
                station_id=station.id,
                sim_start_time=START_OF_SIMULATION,
                sim_end_time=END_OF_SIMULATION,
            )
            peak_power = occupancy_df["power"].max()
            peak_powers_by_station[station.name_short] = peak_power

        peak_powers_by_scenario[scenario_name] = peak_powers_by_station

        # Add the "no charging" kms to the dictionary
        # first, find all rotations with no opportunity charging
        no_charging_rotations = (
            session.query(Rotation)
            .filter(Rotation.scenario_id == scenario.id)
            .filter(Rotation.allow_opportunity_charging == False)
            .all()
        )
        # sum up the vehicle kms for these rotations
        total_no_charge_kms = (
            sum(
                [
                    sum([trip.route.distance for trip in rotation.trips])
                    for rotation in no_charging_rotations
                ]
            )
            / 1000
        )  # convert to km

        vehicle_kms_by_station["No Charging"] = (total_no_charge_kms, True)

        # Sort the dictionary by vehicle kilometers
        vehicle_kms_by_station = dict(
            sorted(
                vehicle_kms_by_station.items(),
                key=lambda item: item[1][0],
                reverse=True,
            )
        )

        # Create a histogram of the vehicle kilometers by station
        station_names = [station for station in vehicle_kms_by_station.keys()]
        vehicle_kms = [
            vehicle_kms[0] for vehicle_kms in vehicle_kms_by_station.values()
        ]
        special = [vehicle_kms[1] for vehicle_kms in vehicle_kms_by_station.values()]
        peak_powers_by_scenario[scenario_name] = peak_powers_by_station
        df = pd.DataFrame(
            {
                "Station": station_names,
                "Vehicle Kilometers": vehicle_kms,
                "Special": special,
            }
        )

        name_for_dict = (
            "Existing Blocks" if scenario_name == "OU" else "Small Batteries"
        )

        dfs[name_for_dict] = df

    # in each df, rename the "Vehicle Kilometers" column to the scenario name
    for scenario_name, df in dfs.items():
        df.rename(columns={"Vehicle Kilometers": scenario_name}, inplace=True)

    # Merge the dataframes on the "Station" column
    # Allowing empty cells for stations that are not in both

    dfs = list(dfs.values())
    combined_df = pd.merge(dfs[0], dfs[1], on="Station", how="outer")
    # Create a new "Special" column that is True if both of the original "Special" columns are True
    combined_df["Special"] = combined_df["Special_x"] & combined_df["Special_y"]

    fig, ax = plt.subplots(
        1, 1, figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.FULLSIZE_PLOT_HEIGHT)
    )

    # Save the df to an xlsx file
    combined_df.to_excel("16_terminus_charging_histogram.xlsx", index=False)

    first_color = "#1f77b4"
    second_color = "#ff7f0e"
    colors = [first_color, second_color]
    colnames = ["Existing Blocks", "Small Batteries"]

    fig, ax = plt.subplots(
        2,
        1,
        figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT),
        sharey=True,
    )

    for i, df in enumerate(dfs):
        df_to_plot = dfs[i]
        color = colors[i]
        colname = colnames[i]

        # order the dataframe by vehicle kilometers
        df_to_plot = df_to_plot.sort_values(by=colname, ascending=False)

        # Fibd the percentage between the sum of everything but "No Charging" and the sum of everything
        total_charging = df_to_plot[df_to_plot["Station"] != "No Charging"][
            colname
        ].sum()
        total_overall = df_to_plot[colname].sum()
        percentage = total_charging / total_overall * 100
        print(
            f"Percentage of {colname} vehicle kilometers that are charged: {percentage:.2f}%"
        )

        # Remove "No Charging" from the dataframe
        df_to_plot = df_to_plot[df_to_plot["Station"] != "No Charging"]

        # print the length of the dataframe#
        print(f"Length of {colname} dataframe: {len(df_to_plot)}")

        df_to_plot.reset_index(drop=True, inplace=True)

        ax[i].bar(
            df_to_plot.index,
            df_to_plot[colname],
            color=color,
            label=colname,
        )

        # Configure axes and labels
        ax[i].set_xticks(range(len(df_to_plot)))
        ax[i].set_xticklabels(df_to_plot["Station"], rotation=60)
        ax[i].set_ylabel("Vehicle Kilometers")
        ax[i].set_xlabel("Station")

        # Remove all x tick labels
        ax[i].set_xticklabels([])

        # Add legend
        ax[i].legend()

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.0)

    plt.savefig("16_terminus_charging_histogram.pdf")
    plt.close()

    # Also create a seaborn histogram of the peak powers
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT),
    )

    # Turn the peak powers into a dict of lists
    peak_powers_dict = {}
    for scenario_name, peak_powers in peak_powers_by_scenario.items():
        peak_powers_dict[scenario_name] = list(peak_powers.values())

    # Rename the keys
    # "OU" -> "Existing Blocks"
    # "TERM" -> "Small Batteries"
    peak_powers_dict["Existing Blocks Unchanged"] = peak_powers_dict.pop("OU")
    peak_powers_dict["Small Batteries \& Termini"] = peak_powers_dict.pop("TERM")

    max_power = max([max(p) for p in peak_powers_dict.values()])

    # Define bins as 0-500, 500-1000, etc.
    bins = np.arange(0, max_power + 500, 500)
    categories = [f"{int(bins[i])}-{int(bins[i + 1])}" for i in range(len(bins) - 1)]

    # Calculate counts per scenario and bin
    data = []
    for scenario, powers in peak_powers_dict.items():
        counts, _ = np.histogram(powers, bins=bins)
        for i, count in enumerate(counts):
            data.append({"Scenario": scenario, "Bin": categories[i], "Count": count})

    # Create DataFrame and set categorical order for bins and scenarios
    df = pd.DataFrame(data)
    df["Bin"] = pd.Categorical(df["Bin"], categories=categories, ordered=True)
    df["Scenario"] = pd.Categorical(
        df["Scenario"],
        categories=[
            "Existing Blocks Unchanged",
            "Small Batteries \& Termini",
        ],
        ordered=True,
    )

    # Plot configuration
    fig, ax = plt.subplots(
        1, 1, figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT)
    )
    palette = sns.color_palette()[1:]  # Using the same color palette as before

    sns.barplot(data=df, x="Bin", y="Count", hue="Scenario", palette=palette, ax=ax)

    ax.set_xlabel("Peak Power [kW]")
    ax.set_ylabel("Count")

    # The x ticks are too close together, so we need to rotate them
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Save and show
    plt.savefig("16b_terminus_peak_power_histogram.pdf")
    # plt.show()
