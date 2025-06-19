#! /usr/bin/env python3


import logging
import os
import warnings
from collections import Counter
from typing import List, Dict

import pandas as pd
import sqlalchemy.orm.session
from eflips.depot.api import simple_consumption_simulation
from eflips.model import *
from eflips.model import ConsistencyWarning
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from util import (
    make_depot_stations_electrified,
    create_consumption_results,
    update_database_to_most_recent_schema,
)
from util_station_electrification import remove_terminus_charging_from_okay_rotations

USE_SIMBA_CONSUMPTION = True
warnings.simplefilter("ignore", ConsistencyWarning)


def number_of_rotations_below_zero(scenario: Scenario, session: Session) -> int:
    """
    Counts the number of rotations in a scenario where the SOC at the depot is below 0%.
    :param scenario: The scenario to check.
    :param session: The database session.
    :return: The number of rotations with SOC below 0%.
    """
    rotations_q = (
        session.query(Rotation)
        .join(Trip)
        .join(Event)
        .filter(Rotation.scenario_id == scenario.id)
        .filter(Event.event_type == EventType.DRIVING)
        .filter(Event.soc_end < 0)
    )
    return rotations_q.count()


def clear_previous_simulation_results(scenario: Scenario, session: Session) -> None:
    session.query(Rotation).filter(Rotation.scenario_id == scenario.id).update(
        {"vehicle_id": None}
    )
    session.query(Event).filter(Event.scenario_id == scenario.id).delete()
    session.query(Vehicle).filter(Vehicle.scenario_id == scenario.id).delete()


def add_charging_station(scenario, session, power: float = 450) -> int | None:
    """
    Adds a charging station to the scenario. The heuristic for selecting the charging station is to add is to select
    the one where the rotations with negative SoC spend the most time. If no such charging station can be found, None
    is returned.

    If the station selected is already electrified, the next best station is selected.

    :param scenario: THE scenario to add the charging station to.
    :param session: An open database session.
    :param power: The power of the charging station to be added. Default is 450 kW.
    :return: Either the id of the charging station that was added, or None if no charging station could be added.
    """
    # First, we identify all the rotations containing a SoC < 0 event
    logger = logging.getLogger(__name__)

    rotations_with_low_soc = (
        session.query(Rotation)
        .join(Trip)
        .join(Event)
        .filter(Event.soc_end < 0)
        .filter(Event.event_type == EventType.DRIVING)
        .filter(Event.scenario == scenario)
        .options(sqlalchemy.orm.joinedload(Rotation.trips).joinedload(Trip.route))
        .distinct()
        .all()
    )

    # For these rotations, we find all the arrival statiosn but the last one. The last one is the depot.
    # We sum up the time spent at a break at each of these stations.
    total_break_time_by_station = Counter()
    for rotation in rotations_with_low_soc:
        for i in range(len(rotation.trips) - 1):
            trip = rotation.trips[i]
            total_break_time_by_station[trip.route.arrival_station_id] += int(
                (
                    rotation.trips[i + 1].departure_time - trip.arrival_time
                ).total_seconds()
            )

    # If all stations have a score iof 0, we terminate the optimization
    if all(v == 0 for v in total_break_time_by_station.values()):
        return None

    for most_popular_station_id, _ in total_break_time_by_station.most_common():
        station: Station = (
            session.query(Station).filter(Station.id == most_popular_station_id).one()
        )
        if station.is_electrified:
            logger.warning(
                f"Station {station.name} is already electrified. Choosing the next best station."
            )
            continue

        if logger.isEnabledFor(logging.DEBUG):
            station_name = (
                session.query(Station)
                .filter(Station.id == most_popular_station_id)
                .one()
                .name
            )
            logger.debug(
                f"Station {most_popular_station_id} ({station_name}) was selected as the station where the most time is spent."
            )

        # Actually add the charging station in the database.

        station.is_electrified = True
        station.amount_charging_places = 100
        station.power_per_charger = power
        station.power_total = station.amount_charging_places * station.power_per_charger
        station.charge_type = ChargeType.OPPORTUNITY
        station.voltage_level = VoltageLevel.MV

        return most_popular_station_id


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

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).one()
        )

        # For the "OU" scenario, we use a charging power of 450 kW
        if scenario_name == "OU":
            CHARGING_POWER = 450
            session.query(VehicleType).filter(
                VehicleType.scenario_id == scenario.id
            ).update({"charging_curve": [[0.0, 450.0], [1.0, 450.0]]})
        else:
            # For term, we only use 300 kW, that should be enough given the longer breaks
            CHARGING_POWER = 300
            session.query(VehicleType).filter(
                VehicleType.scenario_id == scenario.id
            ).update({"charging_curve": [[0.0, 450.0], [1.0, 450.0]]})

        # We need to set the loaded mass for each trip
        # Set all trip's loaded mass to 17.6 passengers, the average for the Germany
        PASSENGER_MASS = 68  # kg
        PASSENGER_COUNT = 17.6  # German-wide average
        payload = PASSENGER_COUNT * PASSENGER_MASS
        session.query(Trip).filter(Trip.scenario == scenario).update(
            {"loaded_mass": payload}
        )

        # Also, for django-simba to run, the depot stations should be electrified
        make_depot_stations_electrified(scenario, session)

        remove_terminus_charging_from_okay_rotations(
            scenario, session, database_url=DATABASE_URL
        )

        # If using simba consumption, we will run it once to determine the delta SoC for each trip
        # Then store these in a dictionary and use simple consumption simulation in "Predefined delta SoC" mode
        if USE_SIMBA_CONSUMPTION:
            consumption_results = create_consumption_results(
                scenario, session, DATABASE_URL
            )

        log: List[Dict[str, int | float]] = []
        while number_of_rotations_below_zero(scenario, session) > 0:
            # Log the current state and write it to an excel table for monitoring
            number_of_eletrified_termini = (
                session.query(Station)
                .filter(Station.scenario_id == scenario.id)
                .filter(Station.is_electrified == True)
                .count()
            )
            log.append(
                {
                    "scenario": scenario_name,
                    "electrified_termini": number_of_eletrified_termini,
                    "rotations_below_zero": number_of_rotations_below_zero(
                        scenario, session
                    ),
                }
            )
            df = pd.DataFrame(log)
            df.to_excel("08b_station_electrification_log.xlsx")

            electrified_station_id = add_charging_station(
                scenario, session, power=CHARGING_POWER
            )
            logger.info(
                f"Added charging station {session.query(Station).filter(Station.id == electrified_station_id).one().name}"
                f" ({electrified_station_id}) to scenario {scenario_name}"
            )

            # Run the consumption simulation again
            clear_previous_simulation_results(scenario, session)
            if USE_SIMBA_CONSUMPTION:
                simple_consumption_simulation(
                    initialize_vehicles=True,
                    scenario=scenario,
                    consumption_result=consumption_results,
                )
            else:
                simple_consumption_simulation(scenario, initialize_vehicles=True)

            min_soc_end = (
                session.query(func.min(Event.soc_end))
                .filter(Event.scenario_id == scenario.id)
                .first()[0]
            )
            count_of_electrified_termini = (
                session.query(Station)
                .filter(
                    Station.scenario_id == scenario.id, Station.is_electrified == True
                )
                .count()
            )
            logger.info(
                f"Minimum SOC at end of day: {min_soc_end}, Electrified termini: {count_of_electrified_termini}"
            )

    session.commit()
