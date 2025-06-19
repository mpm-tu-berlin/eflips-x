#! /usr/bin/env python3

"""
Using system calls, drop the existing data (after asking for use confirmation) and import the data from
`00_bvg_schedule_input.sql`
"""

import os

from eflips.model import *
from sqlalchemy import create_engine, not_
from sqlalchemy.orm import Session

from util import database_url_components

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]
    SCENARIO_ID = 1

    # Ask for user confirmation
    print(
        "This script will drop all existing data and import the data from 00_bvg_schedule_input.sql"
    )
    print("Do you want to continue? (yes/no)")
    user_input = input()
    if user_input != "yes":
        print("Aborting...")
        exit(0)

    _, database_user, database_password, database_host, database_port, database_name = (
        database_url_components(DATABASE_URL)
    )

    os.system(
        f"psql -h {database_host} -U {database_user} -p {database_port} {database_name} -f clear_database.sql"
    )

    os.system("rm -f 00_bvg_schedule_input.sql")
    os.system(f"unzstd -T0 --long=31 00_bvg_schedule_input.sql.zst")

    os.system(
        f"psql -h {database_host} -U {database_user} -p {database_port} {database_name} -f 00_bvg_schedule_input.sql"
    )

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # Reduce the data by removing all rotations that are not of the vehicle types we want to keep
    vehicle_types_we_want_to_keep = ["GN", "EN", "EED-40", "EED", "DL", "GEG", "EE"]
    rotations_to_remove = (
        session.query(Rotation)
        .filter(Rotation.scenario_id == SCENARIO_ID)
        .join(VehicleType)
        .filter(not_(VehicleType.name_short.in_(vehicle_types_we_want_to_keep)))
        .all()
    )
    for rotation in rotations_to_remove:
        print(
            f"Removing rotation {rotation.name}, vehicle type {rotation.vehicle_type.name_short}, start {rotation.trips[0].route.departure_station.name}, end {rotation.trips[-1].route.arrival_station.name}"
        )
        for trip in rotation.trips:
            for stop_time in trip.stop_times:
                session.delete(stop_time)
            session.delete(trip)
        session.delete(rotation)
    session.flush()

    # Create the new vehicle types
    # Create three new vehicle types. One single, one double, and a long bus.
    single_decker = VehicleType(
        name="Ebusco 3.0 12 large battery",
        scenario_id=SCENARIO_ID,
        name_short="EN",
        battery_capacity=500.0,
        battery_capacity_reserve=0.0,
        charging_curve=[[0, 300], [1, 300]],
        opportunity_charging_capable=True,
        minimum_charging_power=10,
        length=12.0,
        width=2.55,
        height=3.19,
        empty_mass=12000,
        allowed_mass=12000 + (70 * 68),  # 70 passengers, 68 kg each
        consumption=None,
    )
    session.add(single_decker)

    bendy_bus = VehicleType(
        name="Solaris Urbino 18 large battery",
        scenario_id=SCENARIO_ID,
        name_short="GN",
        battery_capacity=640.0,
        battery_capacity_reserve=0.0,
        charging_curve=[[0, 300], [1, 300]],
        opportunity_charging_capable=True,
        minimum_charging_power=10,
        length=18.0,
        width=2.55,
        height=3.19,
        empty_mass=19000,
        allowed_mass=19000 + (100 * 68),  # 100 passengers, 68 kg each
        consumption=None,
    )
    session.add(bendy_bus)

    double_decker = VehicleType(
        name="Alexander Dennis Enviro500EV large battery",
        scenario_id=SCENARIO_ID,
        name_short="DD",
        battery_capacity=472,
        battery_capacity_reserve=0,
        charging_curve=[[0, 300], [1, 300]],
        opportunity_charging_capable=True,
        minimum_charging_power=10,
        length=12.0,
        width=2.55,
        height=4.3,
        empty_mass=19000,
        allowed_mass=19000 + (112 * 68),  # 112 passengers, 68 kg each
        consumption=None,
    )
    session.add(double_decker)
    session.flush()

    vehicle_type_mapping = {
        bendy_bus: ["GN", "GEG"],
        single_decker: ["EN", "EED-40", "EED", "EE"],
        double_decker: ["DL"],
    }
    for new_vehicle_type, old_vehicle_types in vehicle_type_mapping.items():
        old_rotations = (
            session.query(Rotation)
            .filter(Rotation.scenario_id == SCENARIO_ID)
            .join(VehicleType)
            .filter(VehicleType.name_short.in_(old_vehicle_types))
            .all()
        )
        for rotation in old_rotations:
            rotation.vehicle_type = new_vehicle_type

    # Delete the old vehicle types
    old_vehicle_types = (
        session.query(VehicleType)
        .filter(VehicleType.scenario_id == SCENARIO_ID)
        .filter(VehicleType.id != single_decker.id)
        .filter(VehicleType.id != bendy_bus.id)
        .filter(VehicleType.id != double_decker.id)
        .all()
    )
    for vehicle_type in old_vehicle_types:
        session.delete(vehicle_type)

    # Reduce by remobing rotations from places we don't want to keep
    short_names_of_stations_to_keep = [
        "BF I",
        "BFI",
        "BF M",
        "BF B",
        "BF S",
        "BF L",
        "BF C",
        "BF MDA",
    ]
    station_ids_to_keep = (
        session.query(Station.id)
        .filter(
            Station.scenario_id == SCENARIO_ID,
            Station.name_short.in_(short_names_of_stations_to_keep),
        )
        .all()
    )
    station_ids_to_keep = [station_id for station_id, in station_ids_to_keep]
    all_rotations = (
        session.query(Rotation).filter(Rotation.scenario_id == SCENARIO_ID).all()
    )
    for rotation in all_rotations:
        first_station_id = rotation.trips[0].route.departure_station_id
        last_station_id = rotation.trips[-1].route.arrival_station_id
        if (
            first_station_id != last_station_id
            or first_station_id not in station_ids_to_keep
        ):
            print(
                f"Removing rotation {rotation.name}, vehicle type {rotation.vehicle_type.name_short}, start {rotation.trips[0].route.departure_station.name}, end {rotation.trips[-1].route.arrival_station.name}"
            )
            for trip in rotation.trips:
                for stop_time in trip.stop_times:
                    session.delete(stop_time)
                session.delete(trip)
            session.delete(rotation)
    session.commit()
    session.close()
