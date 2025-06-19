#! /usr/bin/env python3

"""
Using system calls, drop the existing data (after asking for use confirmation) and import the data from
`00_bvg_schedule_input.sql`
"""
import logging
import os
from datetime import timedelta
from typing import Dict

import networkx as nx
from ds_wrapper import DjangoSimbaWrapper
from eflips.model import *
from eflips.opt.scheduling import (
    passenger_trips_by_vehicle_type,
    create_graph,
    solve,
    write_back_rotation_plan,
    efficiency_info,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload

from params import MAXIMUM_SCHEDULE_DURATION, LONGER_BREAK_DURATION
from util import make_depot_stations_electrified, clear_sim_results


def delta_soc_lut(
    scenario: Scenario, session: sqlalchemy.orm.session.Session
) -> Dict[int, float]:
    """
    Genarates a (trip_id, energy_consumption) dictionary for a given scenario and session.
    The energy consuption is in kWh and calculated from the trip distance and VehicleType.consumption.

    The additinonal_marging is used to ensure no vehicle finishes its last trip with less than 10% of the battery.
    It is added using the formula energy_consumption_with_margin = energy_consumption * (1 + additional_marging)

    :param scenario: The scenario to use
    :param session: An EBO Database session
    :return: A dictionary with trip_id as key and energy_consumption as value
    """

    # Check that each vehicle in the scenario has a consumption LUT
    for vehicle_type in session.query(VehicleType).filter(
        VehicleType.scenario_id == scenario.id
    ):
        if len(vehicle_type.vehicle_classes) != 1:
            raise ValueError(
                f"Vehicle {vehicle_type.name_short} has {len(vehicle_type.vehicle_classes)} vehicle classes"
            )
        if not vehicle_type.vehicle_classes[0].consumption_lut:
            raise ValueError(
                f"Vehicle {vehicle_type.name_short} has no consumption LUT"
            )
    # Clear all existing driving events, vehicles and rotation-vehicle assignments
    clear_sim_results(scenario, session)

    # Set all trip's loaded mass to 17.6 passengers, the average for the Germany
    PASSENGER_MASS = 68  # kg
    PASSENGER_COUNT = 17.6  # German-wide average
    payload = PASSENGER_COUNT * PASSENGER_MASS
    session.query(Trip).filter(Trip.scenario == scenario).update(
        {"loaded_mass": payload}
    )

    # Simulate the scenario using django-simba
    make_depot_stations_electrified(scenario, session)
    session.commit()
    ds_wrapper = DjangoSimbaWrapper(os.environ["DATABASE_URL"])
    ds_wrapper.run_simba_scenario(scenario.id, assign_vehicles=True)
    del ds_wrapper
    # A second commit is needed to get the results of the consumption simulation.
    # Since it used a separate session, we need to commit our own session again
    # to make the results visible.
    session.commit()
    session.expire_all()  # Just for safety, to make sure that we don't use stale data.
    # Create a dictionary with the energy consumption for each trip
    delta_soc = {}

    # Now, our scenario should contain driving events with a soc_start and soc_end for each trip
    for trip in (
        session.query(Trip)
        .filter(Trip.scenario == scenario)
        .options(joinedload(Trip.events))
    ):
        if len(trip.events) != 1:
            raise ValueError(f"Trip {trip.id} has {len(trip.events)} events")
        event = trip.events[0]
        delta_soc[trip.id] = event.soc_start - event.soc_end

    # Clear all existing driving events, vehicles and rotation-vehicle assignments
    clear_sim_results(scenario, session)

    return delta_soc


if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]
    logger = logging.getLogger(__name__)

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    scenarios = session.query(Scenario).filter(Scenario.name_short.in_(["DEP", "TERM"]))

    for scenario in scenarios:
        clear_sim_results(scenario, session)
        # For the "DEP" scenario, we need to calculate the delta soc for each trip
        if scenario.name_short == "DEP":
            # Cache the delta soc lut
            delta_socs = delta_soc_lut(scenario, session)

            # To get a safety factor, increase all delta socs by 10%
            for trip_id, delta_soc in delta_socs.items():
                delta_socs[trip_id] = 1.1 * delta_soc
        else:
            delta_socs = None
        maximum_schedule_duration = MAXIMUM_SCHEDULE_DURATION

        # For the "TERM" scenario, we use a higher minimum break time in the first run
        # Then, after placing the chargers, we run the scheduling and depot-vehicle-assignment again
        if scenario.name_short == "TERM":
            trips_to_give_longer_breaks = []
            minimum_break_time = LONGER_BREAK_DURATION
        else:
            trips_to_give_longer_breaks = []
            minimum_break_time = timedelta(minutes=0)

        for vehicle_type, trips in passenger_trips_by_vehicle_type(
            scenario, session
        ).items():

            rotation_plan_file_name = f"{vehicle_type.name_short}-{scenario.id}.pkl"

            graph = create_graph(
                trips,
                delta_socs=delta_socs,
                maximum_schedule_duration=maximum_schedule_duration,
                minimum_break_time=minimum_break_time,
                longer_break_time_trips=trips_to_give_longer_breaks,
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

    session.commit()
    session.close()
