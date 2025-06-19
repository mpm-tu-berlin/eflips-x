#! /usr/bin/env python3


import json
import logging
import os
import warnings
from collections import defaultdict, OrderedDict
from datetime import timedelta
from typing import Dict

from eflips.depot.api import simple_consumption_simulation  # <-- Remove if unused
from eflips.model import *
from eflips.model import ConsistencyWarning
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from util import (
    clear_previous_simulation_results,
    create_consumption_results,
    make_depot_stations_electrified,
    update_trip_loaded_masses,
)

USE_SIMBA_CONSUMPTION = True
warnings.simplefilter("ignore", ConsistencyWarning)


def write_longer_charging_breaks(scenario: Scenario, session: Session) -> None:
    logger = logging.getLogger(__name__)
    # Query for rotations that have at least one driving event with SOC < 0
    negative_soc_rotations_q = (
        session.query(Rotation)
        .join(Trip)
        .join(Event)
        .filter(Rotation.scenario == scenario)
        .filter(Event.event_type == EventType.DRIVING)
        .filter(Event.soc_end < 0)
        .distinct()
    )

    # Check if any such rotations exist
    num_negative_soc_rotations = negative_soc_rotations_q.count()
    if num_negative_soc_rotations == 0:
        logger.info(
            f"No rotations with negative SOC events in scenario '{scenario.name}'"
        )
        return

    logger.info(
        f"Found {num_negative_soc_rotations} rotations with negative SOC events in scenario '{scenario.name}'"
    )

    # Fetch all rotations that need analysis
    negative_soc_rotations = negative_soc_rotations_q.all()

    # Dictionary mapping Station -> dict containing total break duration and set of rotation IDs
    # Example structure: {
    #     Station(...): {
    #         "duration": timedelta(...),
    #         "rotation_ids": {1, 2, 5}
    #     },
    #     ...
    # }
    station_break_info: Dict[Station, Dict[str, object]] = defaultdict(
        lambda: {"duration": timedelta(seconds=0), "rotation_ids": set()}
    )

    # Compute break time for each station where the rotation stops
    for rotation in negative_soc_rotations:
        for i, trip in enumerate(rotation.trips):
            # Skip the last trip since it doesn't have a subsequent departure
            if i == len(rotation.trips) - 1:
                continue

            # Determine station at the end of this trip and station at the start of the next trip
            last_station = trip.route.arrival_station
            next_station = rotation.trips[i + 1].route.departure_station

            # If arrival and departure stations differ, warn
            if last_station != next_station:
                logger.warning(
                    f"Trip {trip.id} ends at station {last_station.id} "
                    f"but next trip starts at station {next_station.id}. "
                    f"Rotation ID {rotation.id}"
                )

            # Calculate break duration between arrival of this trip and departure of next trip
            break_duration = rotation.trips[i + 1].departure_time - trip.arrival_time

            # Accumulate break duration for this station
            station_break_info[last_station]["duration"] += break_duration

            # Add this rotation ID to the station's set
            station_break_info[last_station]["rotation_ids"].add(rotation.id)

    # Sort stations by total break duration (descending order)
    stations_sorted_by_break = OrderedDict(
        sorted(
            station_break_info.items(),
            key=lambda item: item[1]["duration"],
            reverse=True,
        )
    )

    # List of trip ids to give longer break times at arrival stations
    trips_to_give_longer_breaks = []

    # Pop stations from the front of the sorted list and assign them to rotations
    while stations_sorted_by_break:
        # Take the station with the largest total break time
        station, info = stations_sorted_by_break.popitem(last=False)
        station_id = station.id

        # Assign this station ID to each rotation in its set
        for rot_id in info["rotation_ids"]:
            rotation = session.query(Rotation).get(rot_id)
            for i, trip in enumerate(rotation.trips):
                if i == len(rotation.trips) - 1:
                    # Skip the last trip since it doesn't have a subsequent departure
                    continue
                if trip.route.arrival_station.id == station_id:
                    trips_to_give_longer_breaks.append(trip.id)

        # Remove those rotations from consideration in the remaining stations
        for other_station_info in stations_sorted_by_break.values():
            other_station_info["rotation_ids"] = {
                r_id
                for r_id in other_station_info["rotation_ids"]
                if r_id not in info["rotation_ids"]
            }

    # Write the result to a JSON file
    output_filename = "trips_to_give_longer_breaks.json"
    with open(output_filename, "w") as fp:
        json.dump(trips_to_give_longer_breaks, fp, indent=4)

    logger.info(
        f"Extended break assignments written to {output_filename} for scenario '{scenario.name}'."
    )


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


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # Set log lvel to debug
    logging.basicConfig(level=logging.INFO)

    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # These are the scenarios where we actually need station electrification
    SCENARIO_NAMES = ["OU", "TERM"]

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).one()
        )
        update_trip_loaded_masses(scenario, session)
        make_depot_stations_electrified(scenario, session)

        # We will need ro generate the SimBA consumption results for this scenario *before* electrifying the stations
        if USE_SIMBA_CONSUMPTION:
            consumption_results = create_consumption_results(
                scenario, session, DATABASE_URL
            )

        # Update the simba scenario options
        # Set logging level to max
        import logging

        logging.basicConfig()
        logging.getLogger("sqlalchemy.engine").setLevel(logging.DEBUG)
        simba_options = scenario.simba_options
        simba_options["cs_power_deps_oppb"] = 300.0
        simba_options["cs_power_deps_depb"] = 300.0

        scenario.simba_options = None  # This is needed to trigger the update
        session.flush()
        scenario.simba_options = simba_options
        session.flush()

        # Electrify ALL stations
        station_q = session.query(Station).filter(Station.scenario_id == scenario.id)
        station_q.update(
            {
                "is_electrified": True,
                "amount_charging_places": 100,
                "power_per_charger": 300,
                "power_total": 100 * 300,
                "charge_type": ChargeType.OPPORTUNITY,
                "voltage_level": VoltageLevel.MV,
            }
        )

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
            .filter(Station.scenario_id == scenario.id, Station.is_electrified == True)
            .count()
        )
        logger.info(
            f"Scenario {scenario_name}:\n"
            f"Minimum SOC at end of day: {min_soc_end}, Electrified termini: {count_of_electrified_termini}"
        )
        if min_soc_end < 0:
            write_longer_charging_breaks(scenario, session)
            logger.error(
                f"Scenario {scenario_name} has rotations with SOC below 0% even with all stations electrified."
            )
            raise ValueError(
                f"Scenario {scenario_name} has rotations with SOC below 0% even with all stations electrified."
            )

    session.rollback()
    session.close()
