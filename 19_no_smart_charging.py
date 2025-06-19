#! /usr/bin/env python3
import os
import warnings
from datetime import timedelta, datetime
from typing import Dict

import pandas as pd
import pytz
from eflips.eval.output.prepare import power_and_occupancy
from eflips.model import *
from eflips.model import ConsistencyWarning
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

warnings.simplefilter("ignore", ConsistencyWarning)
USE_SIMBA_CONSUMPTION = True
PARALLELISM = True

tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)


def peak_power_per_depot(
    scenario: Scenario, session: sqlalchemy.orm.session.Session
) -> Dict[Depot, float]:
    """
    Find the peak power for each depot in the scenario
    :param scenario: The scenario to analyze
    :param session: An SQLAlchemy session
    :return: A dictionary mapping depots to their peak power
    """
    depot_peak_powers: Dict[str, float] = {}
    for depot in session.query(Depot).filter(Depot.scenario_id == scenario.id).all():
        areas = session.query(Area).filter(Area.depot_id == depot.id).all()
        powers_df = power_and_occupancy(
            area_id=[area.id for area in areas],
            session=session,
            temporal_resolution=1,
            sim_start_time=START_OF_SIMULATION,
            sim_end_time=END_OF_SIMULATION,
        )
        depot_peak_powers[depot.name] = powers_df["power"].max()
        depot_peak_powers["scenario"] = scenario.name_short
    return depot_peak_powers


if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # Create a dataframe of the rotation distance and vehicle type short name for each rotaiton
    SCENARIO_NAMES = ["OU", "DEP", "TERM"]

    peak_powers = []

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).one()
        )

        depot_peak_powers = peak_power_per_depot(scenario, session)
        peak_powers.append(depot_peak_powers)

        # Also, for the "OU" scenario, "BF I" depot, save the power and occupancy data
        if scenario_name == "OU":
            depot = (
                session.query(Depot)
                .filter(Depot.scenario == scenario)
                .filter(Depot.name_short == "BF I")
            ).one()
            areas = session.query(Area).filter(Area.depot_id == depot.id).all()
            powers_df = power_and_occupancy(
                area_id=[area.id for area in areas],
                session=session,
                temporal_resolution=1,
                sim_start_time=START_OF_SIMULATION,
                sim_end_time=END_OF_SIMULATION,
            )
            powers_df.to_pickle("19_power_and_occupanc_no_smart.pkl")

    # Combine the results, making the "power" column from the "smart" the "With smart charging" column
    # and the "power" column from the "no smart" the "No smart charging" column
    df = pd.DataFrame(peak_powers)
    df.to_excel("19_no_smart_charging_results.xlsx", index=False)
