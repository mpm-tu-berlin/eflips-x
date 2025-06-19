#! /usr/bin/env python3

"""
Using system calls, drop the existing data (after asking for use confirmation) and import the data from
`00_bvg_schedule_input.sql`
"""

import os

from eflips.model import *
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]
    SCENARIO_ID = 1

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # Clean up the data
    # Remove all routes that have no trips
    all_routes = session.query(Route).all()
    for route in all_routes:
        if len(route.trips) == 0:
            print(f"Removing route {route.name}")
            for assoc_route_station in route.assoc_route_stations:
                session.delete(assoc_route_station)
            session.delete(route)

    # Remove all lines that have no routes
    all_lines = session.query(Line).filter(Line.scenario_id == SCENARIO_ID).all()
    for line in all_lines:
        if len(line.routes) == 0:
            print(f"Removing line {line.name}")
            session.delete(line)

    # Remove all stations that are not part of a route
    all_stations = session.query(Station).all()
    for station in all_stations:
        if (
            len(station.assoc_route_stations) == 0
            and len(station.routes_departing) == 0
            and len(station.routes_arriving) == 0
        ):
            print(f"Removing station {station.name}")
            session.delete(station)

    # Print the number of remaining routes, lines, and stations
    print(
        f"Number of remaining routes: {session.query(Route).filter(Route.scenario_id == SCENARIO_ID).count()}"
    )
    print(
        f"Number of remaining lines: {session.query(Line).filter(Line.scenario_id == SCENARIO_ID).count()}"
    )
    print(
        f"Number of remaining stations: {session.query(Station).filter(Station.scenario_id == SCENARIO_ID).count()}"
    )
    session.flush()
    session.commit()

    # Copy the scenario two more times
    scenario: Scenario = (
        session.query(Scenario).filter(Scenario.id == SCENARIO_ID).one()
    )
    scenario.name = "Originalumläufe"
    scenario.name_short = "OU"
    cloned_scenario = scenario.clone(session)
    cloned_scenario.name = "Depotlader"
    cloned_scenario.name_short = "DEP"
    session.commit()
    session.close()

    session = Session(engine)
    scenario: Scenario = (
        session.query(Scenario).filter(Scenario.id == SCENARIO_ID).one()
    )
    cloned_scenario = scenario.clone(session)
    cloned_scenario.name = "Fokus Endhaltestellen"
    cloned_scenario.name_short = "TERM"

    # For the "TERM" scenario, reduce the battery capacity and mass of the vehicles
    battery_apacities = {
        "DD": 320,
        "EN": 250,
        "GN": 320,
    }

    empty_masses = {
        "DD": 18000,
        "EN": 9950,
        "GN": 17000,
    }

    allowed_masses = {
        "DD": 18000 + (112 * 68),
        "EN": 9950 + (70 * 68),
        "GN": 17000 + (100 * 68),
    }

    for short_name in battery_apacities.keys():
        vehicle_type = (
            session.query(VehicleType)
            .filter(
                VehicleType.scenario_id == cloned_scenario.id,
                VehicleType.name_short == short_name,
            )
            .one()
        )
        vehicle_type.battery_capacity = battery_apacities[short_name]
        vehicle_type.empty_mass = empty_masses[short_name]
        vehicle_type.allowed_mass = allowed_masses[short_name]
        vehicle_type.name = vehicle_type.name.replace("large battery", "small battery")

    session.commit()
    session.close()
