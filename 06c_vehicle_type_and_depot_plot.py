#! /usr/bin/env python3

import os
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
from eflips.model import *
from eflips.model import TripType
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import plotutils

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]
    SCENARIO_ID = 1

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    names_of_depots = [
        "Betriebshof Britz",
        "Betriebshof Lichtenberg",
        "Betriebshof Lichtenberg",
        "Betriebshof Britz",
        "Betriebshof Spandau",
        "Betriebshof Cicerostr.",
        "Betriebshof Müllerstr.",
        "Betriebshof Indira-Gandhi-Str.",
        "Betriebshof Köpenicker Landstraße",
        "Betriebshof Rummelsburger Landstraße",
        "Betriebshof Säntisstraße",
    ]
    names_of_depots = set(names_of_depots)

    # Provide some information about the remaining data. Create a dataframe showing the number of rotations, trips and
    # total kilometers for each vehicle type and originating depot
    vehicle_type_data: List[Dict[str, int | float]] = []
    for vehicle_type in (
        session.query(VehicleType).filter(VehicleType.scenario_id == SCENARIO_ID).all()
    ):
        for depot_name in names_of_depots:
            depot_station = (
                session.query(Station)
                .filter(Station.name == depot_name)
                .filter(Station.scenario_id == SCENARIO_ID)
                .first()
            )
            depot_station_name = depot_station.name
            depot_station_name = depot_station_name.removeprefix("Betriebshof ")
            depot_station_name = depot_station_name.removeprefix("Abstellfläche ")
            rotations = (
                session.query(Rotation)
                .join(Trip)
                .join(Route)
                .join(Station, Route.departure_station_id == Station.id)
                .join(VehicleType)
                .filter(
                    Rotation.scenario_id == SCENARIO_ID,
                    VehicleType.id == vehicle_type.id,
                    Station.name == depot_station.name,
                )
                .all()
            )
            trips = sum([len(rotation.trips) for rotation in rotations]) - 2
            total_distance = (
                sum(
                    [
                        sum(
                            [
                                (
                                    trip.route.distance
                                    if trip.trip_type == TripType.PASSENGER
                                    else 0
                                )
                                for trip in rotation.trips
                            ]
                        )
                        for rotation in rotations
                    ]
                )
                / 1000
            )
            vehicle_type_data.append(
                {
                    "Fahrzeugtyp": vehicle_type.name_short,
                    "depot": depot_station_name,
                    "Umläufe": len(rotations),
                    "trips": trips,
                    "Fahrzeugkilometer": total_distance,
                }
            )
    vehicle_type_df = pd.DataFrame(vehicle_type_data)

    # Sum up all the "Fahrzeugkilometer"
    print(f"Total Fahrzeugkilometer: {vehicle_type_df['Fahrzeugkilometer'].sum()}")

    vehicle_name_translation = {
        "EN": "Single Decker",
        "GN": "Articulated Bus",
        "DD": "Double Decker",
    }

    # Now, do two stacked bar plots, one for kilometers and one for trips
    # Stack each vehicle type on top of each other
    fig, ax = plt.subplots(
        1, 1, figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT)
    )

    # Replace the vehicle type names with the translated names
    vehicle_type_df["Fahrzeugtyp"] = vehicle_type_df["Fahrzeugtyp"].apply(
        lambda x: vehicle_name_translation[x]
    )
    vehicle_type_df["Fahrzeugkilometer"] *= 52 / 1000000

    # Change the color palette to seaborns default color palette, but start at the 3rd color
    palette = sns.color_palette("Set2")

    df2 = vehicle_type_df.pivot(
        index="depot", columns="Fahrzeugtyp", values="Fahrzeugkilometer"
    )

    # Order:Single Decker, Double Decker, Articulated Bus
    df2 = df2[["Single Decker", "Articulated Bus", "Double Decker"]]
    # Plot the data

    # df2 contains the data after depot assignment.
    OLD_FILE = "03a_vehicle_type_and_depot_plot.pkl"
    old_df = pd.read_pickle(OLD_FILE)
    # Old df contains the data before depot assignment. Its rows are depots, its columns are vehicle types.
    # Same structure as df2, but the values are the sum of all vehicle types for each depot.

    # We want to plot stacked bars, with the old and new side by side for each depot.

    # Ensure depots are the index in both DataFrames
    depots = set(df2.index.tolist() + old_df.index.tolist())
    # Order the depots in the same order
    depots = list(sorted(depots))

    # Reindex both DataFrames to ensure they have the same order (and fill missing values with 0)
    df2 = df2.reindex(depots, fill_value=0)
    old_df = old_df.reindex(depots, fill_value=0)

    N = len(depots)
    positions = np.arange(N)
    width = 0.33  # Width of each bar

    fig, ax = plt.subplots()

    # Define vehicle types and colors
    vehicle_types = df2.columns  # Order: Single Decker, Articulated Bus, Double Decker
    colors = palette[: len(vehicle_types)]

    # Plot old data (stacked)
    old_bottom = np.zeros(N)
    for i, vtype in enumerate(vehicle_types):
        current_values = old_df[vtype].values
        ax.bar(
            positions - width / 2,
            current_values,
            width,
            bottom=old_bottom,
            color=colors[i],
            label="_nolegend_",
            alpha=0.33,
            hatch="/",
        )
        old_bottom += current_values

    # Plot new data (stacked)
    new_bottom = np.zeros(N)
    for i, vtype in enumerate(vehicle_types):
        current_values = df2[vtype].values
        ax.bar(
            positions + width / 2,
            current_values,
            width,
            bottom=new_bottom,
            color=colors[i],
            label="_nolegend_",
        )
        new_bottom += current_values

    # Configure x-axis and labels
    ax.set_xticks(positions)
    # Replace a space in the depot name with a line break
    ax.set_xticklabels(
        [depot.replace(" ", "\n").replace("straße", "str.") for depot in depots],
        rotation=60,
    )  # Rotate depot labels for better readability

    ax.set_xlabel("")
    ax.set_ylabel(r"Revenue Mileage $\left[ \frac{km \times 10^6}{a} \right]$")
    ax.set_ylim(0, 21)  # Set y limit to 10 km

    # Create handles for vehicle types (as before)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(vehicle_types))
    ]

    # Create handles for Before/After (hatch and no hatch)
    before_patch = plt.Rectangle(
        (0, 0), 1, 1, facecolor="lightgrey", hatch="/", alpha=0.33
    )
    after_patch = plt.Rectangle((0, 0), 1, 1, color="lightgrey")

    # Combine all handles and labels
    all_handles = handles + [before_patch, after_patch]
    all_labels = list(vehicle_types) + [
        "Before",
        "After",
    ]

    # Update the legend to include both vehicle types and Before/After
    ax.legend(
        all_handles,
        all_labels,
        title="",
        # bbox_to_anchor=(0, 1.02, 1, 0.2),
        ncols=2,  # Adjust as needed for spacing
    )

    # Final adjustments and save
    ax.set_title("")
    plt.tight_layout()
    plt.savefig("06c_vehicle_type_and_depot_plot.pdf")
