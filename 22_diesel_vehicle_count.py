#! /usr/bin/env python3
import logging
import os
import warnings

from eflips.model import *
from eflips.model import ConsistencyWarning
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import util
from util import clear_previous_simulation_results

warnings.simplefilter("ignore", ConsistencyWarning)
USE_SIMBA_CONSUMPTION = False
PARALLELISM = True


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # Create a dataframe of the rotation distance and vehicle type short name for each rotaiton
    SCENARIO_NAME = "OU"
    SCENARIO_NAME_DIESEL = "OU_DIESEL"

    if (
        session.query(Scenario)
        .filter(Scenario.name_short == SCENARIO_NAME_DIESEL)
        .count()
        == 0
    ):
        logger.warning(f"Creating scenario {SCENARIO_NAME_DIESEL}")
        old_scenario = (
            session.query(Scenario).filter(Scenario.name_short == SCENARIO_NAME).one()
        )

        # Duplicate the scenario
        old_scenario: Scenario

        logger.info("Starting to duplicate scenario")
        scenario = old_scenario.clone(session)
        logger.info("Finished duplicating scenario")

        scenario.name_short = "OU_DIESEL"
        scenario.name_long = "OU (Diesel)"

        logger.info("Starting to remove results")
        clear_previous_simulation_results(scenario, session)
        logger.info("Finished removing results")

        logger.info("Comitting changes")
        session.commit()
        logger.info("Finished comitting changes")

        print("Scenario has been duplicated. Please run this script again.")
        # session.close()
        # exit() TODO: Re-enable this
    else:
        scenario = (
            session.query(Scenario)
            .filter(Scenario.name_short == SCENARIO_NAME_DIESEL)
            .one()
        )

    # We want to remove the vehicle classes and comsumptions
    for vehicle_type in session.query(VehicleType).filter(
        VehicleType.scenario_id == scenario.id
    ):
        vehicle_type: VehicleType

        for vehicle_class in vehicle_type.vehicle_classes:
            vehicle_class: VehicleClass
            if vehicle_class.consumption_lut is not None:
                clut = vehicle_class.consumption_lut
                vehicle_class.consumption_lut = None
                session.delete(clut)
            vehicle_type.vehicle_classes = []
            session.delete(vehicle_class)
        session.flush()

        vehicle_type.consumption = 0.001  # Very low consumption
        session.flush()

    session.commit()

    # Now, simulate the consumption
    util._inner_run_simulation(scenario.id, DATABASE_URL, USE_SIMBA_CONSUMPTION=False)

    session.commit()
    session.close()
