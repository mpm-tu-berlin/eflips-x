#! /usr/bin/env python3


import os

from eflips.model import *
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from util_depot_assignment import optimize_scenario

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # Create a dataframe of the rotation distance and vehicle type short name for each rotaiton
    SCENARIO_NAMES = ["OU", "DEP", "TERM"]

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).first()
        )

        optimize_scenario(scenario, session)
        session.commit()
