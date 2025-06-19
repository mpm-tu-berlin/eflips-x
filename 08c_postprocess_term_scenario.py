#! /usr/bin/env python3


import logging
import os
import warnings
from datetime import timedelta
from tempfile import gettempdir
from typing import List, Dict, Any

import networkx as nx
import sqlalchemy.orm.session
import sqlalchemy.orm.session
from eflips.model import ConsistencyWarning, ChargeType
from eflips.model import Scenario, Trip, Route, Station
from eflips.opt.scheduling import (
    passenger_trips_by_vehicle_type,
    create_graph,
    solve,
    efficiency_info,
    write_back_rotation_plan,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from params import MAXIMUM_SCHEDULE_DURATION
from util import clear_sim_results
from util import (
    update_database_to_most_recent_schema,
)
from util_depot_assignment import optimize_scenario
from util_station_electrification import remove_terminus_charging_from_okay_rotations

USE_SIMBA_CONSUMPTION = True
warnings.simplefilter("ignore", ConsistencyWarning)


def find_trips_to_give_longer_breaks_to(
    scenario: Scenario, session: sqlalchemy.orm.session.Session
) -> List[int]:
    """
    Ideintify the IDs of the trips which end at a station with is_electrified=True
    :param scenario:
    :param session:
    :return: A list of trip IDs
    """
    all_trips = (
        session.query(Trip)
        .filter(Trip.scenario == scenario)
        .join(Route)
        .join(Station, Route.arrival_station_id == Station.id)
        .filter(Station.is_electrified == True)
        .all()
    )
    return [trip.id for trip in all_trips]


def do_scheduling_for_scenario(
    scenario: Scenario,
    session: sqlalchemy.orm.session.Session,
    longer_break_time_trip_ids=[],
):
    clear_sim_results(scenario, session)

    if scenario.name_short != "TERM":
        raise ValueError("This function is only for the TERM scenario so far.")

    trips_to_give_longer_breaks_to = find_trips_to_give_longer_breaks_to(
        scenario, session
    )
    minimum_break_time = timedelta(
        minutes=0
    )  # For those trips that are not in the list

    for vehicle_type, trips in passenger_trips_by_vehicle_type(
        scenario, session
    ).items():

        rotation_plan_file_name = f"{vehicle_type.name_short}-{scenario.id}.pkl"

        graph = create_graph(
            trips,
            delta_socs=None,
            maximum_schedule_duration=MAXIMUM_SCHEDULE_DURATION,
            minimum_break_time=minimum_break_time,
            longer_break_time_trips=trips_to_give_longer_breaks_to,
        )
        rotation_plan = solve(graph, write_to_file=True)

        trip_lists = []
        for set_of_nodes in nx.connected_components(rotation_plan.to_undirected()):
            topoogical_order = list(
                nx.topological_sort(rotation_plan.subgraph(set_of_nodes))
            )
            trip_lists.append(topoogical_order)
        efficiency_info(trip_lists, session)

        print("Starting writeback…")
        write_back_rotation_plan(rotation_plan, session)
        print("Writeback ended!")


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
    SCENARIO_NAMES = ["TERM"]

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).one()
        )

        # Run the scheduling
        do_scheduling_for_scenario(scenario, session)

        # Run the depot assignment
        optimize_scenario(scenario, session)

        # Now, we need to remove the consumption cache file, if one exists
        consumption_cache_file_path = os.path.join(
            gettempdir(), f"consumption_cache_{scenario.id}_{scenario.name_short}.pkl"
        )
        if os.path.exists(consumption_cache_file_path):
            os.remove(consumption_cache_file_path)

        # We need to remove all electric charging stations from the database
        # SO we can look at "which rotations will be okay without terminus charging"?
        # As well, django-simba messes up otherwise.
        # We will need to remember them, including their electrification parameters. Which are:
        # is_electrified
        # amount_charging_places
        # power_per_charger
        # power_total
        # charge_type
        already_elecrified_stations: List[Dict[str, Any]] = []
        for station in (
            session.query(Station)
            .filter(Station.scenario_id == scenario.id)
            .filter(Station.is_electrified)
            .filter(Station.charge_type != ChargeType.DEPOT)
        ):
            already_elecrified_stations.append(
                {
                    "id": station.id,
                    "is_electrified": station.is_electrified,
                    "amount_charging_places": station.amount_charging_places,
                    "power_per_charger": station.power_per_charger,
                    "power_total": station.power_total,
                    "charge_type": station.charge_type,
                    "voltage_level": station.voltage_level,
                }
            )
            station.is_electrified = False
            station.amount_charging_places = None
            station.power_per_charger = None
            station.power_total = None
            station.charge_type = None
            station.voltage_level = None

        # Remove terminus charging from the rotations wich do not need it
        remove_terminus_charging_from_okay_rotations(
            scenario=scenario, session=session, database_url=DATABASE_URL
        )

        # Now, we need to re-electrify the stations
        for station_data in already_elecrified_stations:
            station = (
                session.query(Station).filter(Station.id == station_data["id"]).one()
            )
            station.is_electrified = station_data["is_electrified"]
            station.amount_charging_places = station_data["amount_charging_places"]
            station.power_per_charger = station_data["power_per_charger"]
            station.power_total = station_data["power_total"]
            station.charge_type = station_data["charge_type"]
            station.voltage_level = station_data["voltage_level"]

    session.commit()
