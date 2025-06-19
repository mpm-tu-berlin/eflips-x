#! /usr/bin/env python3
import os
import warnings
from datetime import timedelta, datetime
from typing import List

import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from eflips.eval.output.prepare import power_and_occupancy
from eflips.model import *
from eflips.model import ConsistencyWarning
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import plotutils

warnings.simplefilter("ignore", ConsistencyWarning)
USE_SIMBA_CONSUMPTION = True
PARALLELISM = True

TEMPORAL_RESOLUTION = 30

tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)


def resample_occ_for_areas(areas: List[Area], times, session):
    area_ids = [area.id for area in areas]
    occupancy = power_and_occupancy(
        area_id=area_ids,
        session=session,
        temporal_resolution=TEMPORAL_RESOLUTION,
    )
    occupancies = occupancy["occupancy_total"].values
    occupancy_times = occupancy["time"].values
    occupancy = np.interp(
        [t.timestamp() for t in times],
        [pd.Timestamp(t).timestamp() for t in occupancy_times],
        occupancies,
    )
    return pd.Series(occupancy)


if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    SCENARIO_NAME = "OU"

    scenario = (
        session.query(Scenario).filter(Scenario.name_short == SCENARIO_NAME).one()
    )
    depot = (
        session.query(Depot)
        .filter(Depot.name_short == "BF I")
        .filter(Depot.scenario_id == scenario.id)
        .one()
    )

    # Results_df will store the occupancy for each area grou
    results_df = pd.DataFrame()

    # The time of our results_df will be the range between the start and end of the simulation, with the temporal resolution
    results_df["time"] = pd.date_range(
        start=START_OF_SIMULATION, end=END_OF_SIMULATION, freq=f"{TEMPORAL_RESOLUTION}S"
    )

    # Load the total occupancy for this depot
    all_areas = session.query(Area).filter(Area.depot_id == depot.id).all()
    results_df["Total"] = resample_occ_for_areas(all_areas, results_df["time"], session)

    # Identify the waiting area
    waiting_area = (
        session.query(Area)
        .filter(Area.name == "Waiting Area for every type of vehicle")
        .filter(Area.depot_id == depot.id)
        .one()
    )
    try:
        results_df["Waiting"] = resample_occ_for_areas(
            [waiting_area], results_df["time"], session
        )
    except ValueError:  # Sometimes there are no vehicles in the waiting area
        results_df["Waiting"] = 0

    # Identify all cleaning areas
    cleaning_areas = (
        session.query(Area)
        .filter(Area.name.like("Cleaning Area%"))
        .filter(Area.depot_id == depot.id)
        .all()
    )
    results_df["Cleaning"] = resample_occ_for_areas(
        cleaning_areas, results_df["time"], session
    )

    # Identify all charging areas
    charging_areas = (
        session.query(Area)
        .filter(Area.name.like("Direct Area%"))
        .filter(Area.depot_id == depot.id)
        .all()
    )
    results_df["Charging"] = resample_occ_for_areas(
        charging_areas, results_df["time"], session
    )

    # Identify the shunting areas
    shunting_areas = (
        session.query(Area)
        .filter(Area.name.like("Shunting Area%"))
        .filter(Area.depot_id == depot.id)
        .all()
    )
    occupancy = power_and_occupancy(
        area_id=[area.id for area in shunting_areas],
        session=session,
        temporal_resolution=TEMPORAL_RESOLUTION,
    )
    results_df["Shunting"] = resample_occ_for_areas(
        shunting_areas, results_df["time"], session
    )

    # create a stackplot
    fig, ax = plt.subplots(
        figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT)
    )

    # Change the color palette to colorbrewer Set2
    palette = sns.color_palette("Spectral", n_colors=3)

    ax.stackplot(
        results_df["time"],
        results_df["Cleaning"],
        results_df["Shunting"],
        results_df["Waiting"],
        results_df["Charging"],
        labels=["Cleaning", "Shunting", "Waiting", "Charging"],
        colors=palette,
    )
    # ax.plot(results_df["time"], results_df["Total"], label="Total", color="black")

    # Configure x-axis to show weekdays and midnight
    # Set major ticks at midnight
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    # Format the ticks to show weekday name
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))

    # Optional: Add minor ticks for hours or other intervals if needed
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    ax.set_xlabel("Day of week")
    ax.set_ylabel("Number of vehicles")

    ax.set_xlim(START_OF_SIMULATION, END_OF_SIMULATION)

    ax.legend()

    plt.tight_layout()
    plt.savefig("21_ig_occupancy.pdf")
