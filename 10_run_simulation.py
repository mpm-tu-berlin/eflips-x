#! /usr/bin/env python3


import multiprocessing
import os
import warnings
from datetime import timedelta, datetime

import pytz
from eflips.model import *
from eflips.model import ConsistencyWarning
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from util import (
    _inner_run_simulation,
)

warnings.simplefilter("ignore", ConsistencyWarning)
USE_SIMBA_CONSUMPTION = True
PARALLELISM = True


tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # Create a dataframe of the rotation distance and vehicle type short name for each rotaiton
    SCENARIO_NAMES = ["OU", "DEP", "TERM"]
    scenario_ids = []

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).first()
        )
        scenario_ids.append(scenario.id)

    if PARALLELISM:
        pool_args = [(scenario_id, DATABASE_URL) for scenario_id in scenario_ids]
        with multiprocessing.Pool() as pool:
            pool.starmap(_inner_run_simulation, pool_args)
    else:
        for scenario_id in scenario_ids:
            _inner_run_simulation(scenario_id, DATABASE_URL)
