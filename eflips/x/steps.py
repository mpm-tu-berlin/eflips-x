"""

This file contains all sorts of workflow steps used in the eflips pipelines.

"""

import glob
from pathlib import Path
from typing import List, Dict
from fuzzywuzzy import fuzz
import eflips.model
from eflips.model import Rotation, VehicleType, Station, Route, AssocRouteStation, StopTime, Line
from geoalchemy2.shape import to_shape
from sqlalchemy import not_, func
from sqlalchemy.orm import Session

from eflips.x.util import pipeline_step
from eflips.ingest.legacy import bvgxml


@pipeline_step(
    step_name="bvgxml-ingest-2025-06",
    code_version="0.1.0",
)
def bvgxml_ingest_2025_06(db_path: Path, input_files: List[Path]):
    """
    Create a database and put the June 2025 BVGXML data into it.
    :param db_path: The path to the database to create and populate.
    :return: nothing
    """
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    eflips.model.Base.metadata.create_all(engine)

    bvgxml.ingest_bvgxml(
        paths=input_files, database_url=url, clear_database=False, multithreading=True
    )


@pipeline_step(
    step_name="remove-unused-vehicle-types",
    code_version="1.0.0",
)
def remove_unused_vehicle_types(db_path: Path):
    """
    A just-imported BVGXML dataset contains some dummy data that does not seem to refer to actual operations.

    This function removes all vehicle types and rotations that are not actually used in the dataset. It also creates
    generic vehicle types for all vehicles we want to keep.

    :param db_path: THe path to the database to modify.
    :return: Nothing
    """
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")
        SCENARIO_ID = scenarios[0].id

        # Reduce the data by removing all rotations that are not of the vehicle types we want to keep
        vehicle_types_we_want_to_keep = [
            "GN",
            "EN",
            "EED-40",
            "EED-120",
            "EED-320",
            "EED-160",
            "EED",
            "DL",
            "D",
            "GEG",
            "EE",
        ]
        vehicle_types_we_want_to_keep = [
            "D",
            "DL",
            "EED-120",
            "EED-160",
            "EED-320",
            "EN",
            "GEG",
            "GEG-200",
            "GN",
        ]
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
            bendy_bus: ["GN", "GEG", "GEG-200"],
            single_decker: ["EED-120", "EED-160", "EED-320", "EN", "MN"],
            double_decker: ["DL", "D"],
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
        session.commit()


@pipeline_step(
    step_name="remove-unused-rotations",
    code_version="1.0.0",
)
def remove_unused_rotations(db_path: Path):
    """
    A just-imported BVGXML dataset contains some dummy data that does not seem to refer to actual operations.

    This function removes all vehicle types and rotations that are not actually used in the dataset. It also creates
    generic vehicle types for all vehicles we want to keep.

    :param db_path: THe path to the database to modify.
    :return: Nothing
    """
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")
        SCENARIO_ID = scenarios[0].id

        # Reduce by remobing rotations from places we don't want to keep
        short_names_of_stations_to_keep = [
            "BTRB",
            "BF B",
            "BF C",
            "BFI",
            "BF I",
            "BHKO",
            "BHLI",
            "BF L",
            "BHMA",
            "BF M",
            "BF S",
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
        all_rotations = session.query(Rotation).filter(Rotation.scenario_id == SCENARIO_ID).all()
        for rotation in all_rotations:
            first_station_id = rotation.trips[0].route.departure_station_id
            last_station_id = rotation.trips[-1].route.arrival_station_id
            if first_station_id != last_station_id or first_station_id not in station_ids_to_keep:
                print(
                    f"Removing rotation {rotation.name}, vehicle type {rotation.vehicle_type.name_short}, start {rotation.trips[0].route.departure_station.name}, end {rotation.trips[-1].route.arrival_station.name}"
                )
                for trip in rotation.trips:
                    for stop_time in trip.stop_times:
                        session.delete(stop_time)
                    session.delete(trip)
                session.delete(rotation)

        session.commit()


@pipeline_step(
    step_name="merge-stations",
    code_version="1.0.0",
)
def merge_stations(db_path: Path, max_distance_meters: float, match_percentage: float):
    """
    A BVGXML dataset contains quite some stations that are different `Station` objects in the database, but from an
    electrification perspective refer to the same thing. Especially when doing vehicle scheduling and not allowing
    "deadheading" between nearby stations, this can lead to problems. This function merges all these stations.


    :param db_path: The path to the database to modify.
    :return: nothing
    """
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        # Identify all the station IDs where routes start or end.
        start_station_ids = session.query(Route.departure_station_id).distinct().all()
        end_station_ids = session.query(Route.arrival_station_id).distinct().all()
        station_ids_in_use = set(
            [station_id for station_id, in start_station_ids]
            + [station_id for station_id, in end_station_ids]
        )

        to_merge: List[List[Station]] = []
        for station in session.query(Station).filter(Station.id.in_(station_ids_in_use)).all():
            geom_wkb = to_shape(station.geom).wkb

            # Do a fancy geospatial query to find all stations within the given distance
            nearby_stations = (
                session.query(Station)
                .filter(Station.id != station.id)
                .filter(Station.id.in_(station_ids_in_use))
                .filter(
                    func.ST_Distance(Station.geom, func.ST_GeomFromWKB(geom_wkb), 1)
                    <= max_distance_meters
                )
                .all()
            )

            if len(nearby_stations) > 0:
                # ALso check if they're named similarly using fuzzy matching
                for nearby_station in nearby_stations:
                    orig_station_name = station.name
                    nearby_station_name = nearby_station.name
                    percentage = fuzz.ratio(orig_station_name, nearby_station_name)
                    if percentage >= match_percentage:
                        # See if one of the stations is already in the to_merge list
                        found = False
                        for merge_group in to_merge:
                            if station in merge_group or nearby_station in merge_group:
                                if station not in merge_group:
                                    merge_group.append(station)
                                if nearby_station not in merge_group:
                                    merge_group.append(nearby_station)
                                found = True
                                break
                        if not found:
                            to_merge.append([station, nearby_station])
                        print(
                            f"Found nearby stations to merge: {station.name} and {nearby_station.name} ({percentage}%)"
                        )

        for merge_group in to_merge:
            # Print the stations to merge
            print(f"Merging stations: {[station.name for station in merge_group]}")
            # Pick the station with the shortest name as the one to keep
            station_to_keep = min(merge_group, key=lambda s: len(s.name))
            print(f"Keeping station: {station_to_keep.name}")
            stations_to_remove = [s for s in merge_group if s != station_to_keep]
            for other_station in stations_to_remove:
                other_station_geom = other_station.geom
                with session.no_autoflush:
                    with session.no_autoflush:
                        # Update all routes, trips, and stoptimes containing the next station to point to the first station instead
                        session.query(Route).filter(
                            Route.departure_station_id == other_station.id
                        ).update({"departure_station_id": station_to_keep.id})
                        session.query(Route).filter(
                            Route.arrival_station_id == other_station.id
                        ).update({"arrival_station_id": station_to_keep.id})

                        session.query(AssocRouteStation).filter(
                            AssocRouteStation.station_id == other_station.id
                        ).update(
                            {"station_id": station_to_keep.id, "location": other_station_geom}
                        )

                        session.query(StopTime).filter(
                            StopTime.station_id == other_station.id
                        ).update({"station_id": station_to_keep.id})
                    session.flush()
                    session.delete(other_station)
        session.commit()


@pipeline_step(
    step_name="merge-stations",
    code_version="1.0.0",
)
def remove_unused_data(db_path: Path):
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

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
        all_lines = session.query(Line).all()
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
        print(f"Number of remaining routes: {session.query(Route).count()}")
        print(f"Number of remaining lines: {session.query(Line).count()}")
        print(f"Number of remaining stations: {session.query(Station).count()}")
        session.flush()
        session.commit()


@pipeline_step(
    step_name="set-battery-capacity-and-mass",
    code_version="0.1.0",
)
def set_battery_capacity_and_mass(
    db_path: Path,
    battery_capacities: Dict[str, float],
    empty_masses: Dict[str, float],
    full_masses: Dict[str, float],
):
    """
    Set the battery capacity and empty mass for vehicle types based on their short names.
    :param db_path: The path to the database to modify.
    :param battery_capacities: A dictionary mapping vehicle type short names to battery capacities in kWh.
    :param empty_masses: A dictionary mapping vehicle type short names to empty masses in kg.
    :param full_masses: A dictionary mapping vehicle type short names to full masses in kg.
    :return: Nothing
    """
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        # We want all vehicle types to exist in the keys of each of the dictionaries
        vehicle_types = session.query(VehicleType).all()
        short_names_in_db = set([vt.name_short for vt in vehicle_types])
        short_names_in_battery = set(battery_capacities.keys())
        short_names_in_empty_mass = set(empty_masses.keys())
        short_names_in_full_mass = set(full_masses.keys())
        if (
            not short_names_in_db
            == short_names_in_battery
            == short_names_in_empty_mass
            == short_names_in_full_mass
        ):
            raise ValueError(
                f"Vehicle type short names in database {short_names_in_db} do not match those in battery capacities {short_names_in_battery}, empty masses {short_names_in_empty_mass}, and full masses {short_names_in_full_mass}"
            )

        for vehicle_type in vehicle_types:
            vehicle_type.battery_capacity = battery_capacities[vehicle_type.name_short]
            vehicle_type.empty_mass = empty_masses[vehicle_type.name_short]
            vehicle_type.allowed_mass = full_masses[vehicle_type.name_short]
            print(
                f"Set vehicle type {vehicle_type.name_short} battery capacity to {vehicle_type.battery_capacity} kWh, empty mass to {vehicle_type.empty_mass} kg, and allowed mass to {vehicle_type.allowed_mass} kg"
            )
        session.flush()
        session.commit()
