"""

This file contains all sorts of workflow steps used in the eflips pipelines.

"""

import logging
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import List, Dict
from zoneinfo import ZoneInfo

import eflips.model
import numpy as np
import pandas as pd
import seaborn as sns
from eflips.depot.api import (
    simple_consumption_simulation,
    generate_consumption_result,
    generate_depot_optimal_size,
    init_simulation,
    add_evaluation_to_database,
    apply_even_smart_charging,
    ConsumptionResult,
)
from eflips.ingest.legacy import bvgxml
from eflips.model import Base
from eflips.model import (
    Rotation,
    VehicleType,
    Station,
    Route,
    AssocRouteStation,
    StopTime,
    Line,
    Trip,
    TripType,
    VehicleClass,
    ConsumptionLut,
    Scenario,
    Temperatures,
    ChargeType,
    VoltageLevel,
    Event,
)
from eflips.model import create_engine
from eflips.opt.scheduling import create_graph, solve, write_back_rotation_plan
from fuzzywuzzy import fuzz
from geoalchemy2.shape import to_shape
from matplotlib import pyplot as plt
from prefect import task
from prefect.artifacts import create_progress_artifact, update_progress_artifact
from sqlalchemy import not_, func
from sqlalchemy.orm import Session, joinedload
from tqdm import tqdm

from eflips.x.flows.util_station_electrification import (
    remove_terminus_charging_from_okay_rotations,
    number_of_rotations_below_zero,
    add_charging_station,
)
from eflips.x.util import pipeline_step
from eflips.x.util_depot_assignment import optimize_scenario
from eflips.x.util_legacy import (
    update_trip_loaded_masses,
    make_depot_stations_electrified,
    clear_previous_simulation_results,
)

from eflips.x.util_tco_calculation import tco_parameters
from eflips.tco import TCOCalculator, init_tco_parameters


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
    # Remove the database if it already exists
    if db_path.exists():
        db_path.unlink()
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    eflips.model.Base.metadata.create_all(engine)

    bvgxml.ingest_bvgxml(
        paths=input_files, database_url=url, clear_database=False, multithreading=True
    )


@pipeline_step(
    step_name="remove-unused-vehicle-types",
    code_version="1.0.1",
)
def remove_unused_vehicle_types(db_path: Path):
    """
    A just-imported BVGXML dataset contains some dummy data that does not seem to refer to actual operations.

    This function removes all vehicle types and rotations that are not actually used in the dataset. It also creates
    generic vehicle types for all vehicles we want to keep.

    :param db_path: THe path to the database to modify.
    :return: Nothing
    """
    logger = logging.getLogger(__name__)
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
            .join(VehicleType)
            .filter(not_(VehicleType.name_short.in_(vehicle_types_we_want_to_keep)))
            .all()
        )
        for rotation in rotations_to_remove:
            logger.debug(
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
            charging_curve=[[0, 450], [1, 450]],
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
            charging_curve=[[0, 450], [1, 450]],
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
            charging_curve=[[0, 450], [1, 450]],
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
    logger = logging.getLogger(__name__)
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
                logger.debug(
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
    logger = logging.getLogger(__name__)
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
                        logger.debug(
                            f"Found nearby stations to merge: {station.name} and {nearby_station.name} ({percentage}%)"
                        )

        for merge_group in to_merge:
            # Print the stations to merge
            logger.debug(f"Merging stations: {[station.name for station in merge_group]}")
            # Pick the station with the shortest name as the one to keep
            station_to_keep = min(merge_group, key=lambda s: len(s.name))
            logger.debug(f"Keeping station: {station_to_keep.name}")
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
    logger = logging.getLogger(__name__)
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
                logger.debug(f"Removing route {route.name}")
                for assoc_route_station in route.assoc_route_stations:
                    session.delete(assoc_route_station)
                session.delete(route)

        # Remove all lines that have no routes
        all_lines = session.query(Line).all()
        for line in all_lines:
            if len(line.routes) == 0:
                logger.debug(f"Removing line {line.name}")
                session.delete(line)

        # Remove all stations that are not part of a route
        all_stations = session.query(Station).all()
        for station in all_stations:
            if (
                len(station.assoc_route_stations) == 0
                and len(station.routes_departing) == 0
                and len(station.routes_arriving) == 0
            ):
                logger.debug(f"Removing station {station.name}")
                session.delete(station)

        # Print the number of remaining routes, lines, and stations
        logger.debug(f"Number of remaining routes: {session.query(Route).count()}")
        logger.debug(f"Number of remaining lines: {session.query(Line).count()}")
        logger.debug(f"Number of remaining stations: {session.query(Station).count()}")
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
    logger = logging.getLogger(__name__)
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
            logger.debug(
                f"Set vehicle type {vehicle_type.name_short} battery capacity to {vehicle_type.battery_capacity} kWh, empty mass to {vehicle_type.empty_mass} kg, and allowed mass to {vehicle_type.allowed_mass} kg"
            )
        session.flush()
        session.commit()


@task(
    name="vehicle-type-and-depot-plot",
)
def vehicle_type_and_depot_plot(db_path: Path, output_path: Path):
    """
    Create a plot showing the vehicle types and depots in the database.
    :param db_path: The path to the database to read.
    :param output_path: The path to the output plot file.
    :return: Nothing
    """

    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        # Identify all the places where rotations start or end.
        all_start_end_stations = set()
        all_rotations = (
            session.query(Rotation)
            .options(joinedload(Rotation.trips).joinedload(Trip.route))
            .all()
        )
        for rotation in all_rotations:
            if len(rotation.trips) == 0:
                raise ValueError(f"Rotation {rotation.name} has no trips")
            all_start_end_stations.add(rotation.trips[0].route.departure_station)
            all_start_end_stations.add(rotation.trips[-1].route.arrival_station)

        vehicle_type_data: List[Dict[str, int | float]] = []
        for vehicle_type in session.query(VehicleType).all():
            for station in all_start_end_stations:
                depot_station_name = station.name
                depot_station_name = depot_station_name.removeprefix("Betriebshof ")
                depot_station_name = depot_station_name.removeprefix("Abstellfläche ")

                rotations = (
                    session.query(Rotation)
                    .join(Trip)
                    .join(Route)
                    .join(Station, Route.departure_station_id == Station.id)
                    .join(VehicleType)
                    .filter(
                        VehicleType.id == vehicle_type.id,
                        Station.id == station.id,
                    )
                    .all()
                )
                trips = sum([len(rotation.trips) for rotation in rotations]) - 2
                total_distance_pax = (
                    sum(
                        [
                            sum(
                                [
                                    (
                                        trip.route.distance
                                        if trip.trip_type == TripType.PASSENGER
                                        else 0
                                    )
                                    for trip in rotation.trips
                                ]
                            )
                            for rotation in rotations
                        ]
                    )
                    / 1000
                )
                total_distance = (
                    sum(
                        [
                            sum([(trip.route.distance) for trip in rotation.trips])
                            for rotation in rotations
                        ]
                    )
                    / 1000
                )
                vehicle_type_data.append(
                    {
                        "Fahrzeugtyp": vehicle_type.name_short,
                        "depot": depot_station_name,
                        "Umläufe": len(rotations),
                        "trips": trips,
                        "Nutzwagenkilometer": total_distance_pax,
                        "Wagenkilometer": total_distance,
                    }
                )

        vehicle_type_df = pd.DataFrame(vehicle_type_data)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        vehicle_type_df.to_pickle(f"{output_path}/vehicle_type_and_depot_data.pkl")
        vehicle_type_df.to_excel(f"{output_path}/vehicle_type_and_depot_data.xlsx")

        vehicle_name_translation = {
            "EN": "Single Decker",
            "GN": "Articulated Bus",
            "DD": "Double Decker",
        }

        # Now, do two stacked bar plots, one for kilometers and one for trips
        # Stack each vehicle type on top of each other
        fig, ax = plt.subplots(
            1,
            1,
        )

        # Replace the vehicle type names with the translated names
        vehicle_type_df["Fahrzeugtyp"] = vehicle_type_df["Fahrzeugtyp"].apply(
            lambda x: vehicle_name_translation[x]
        )
        vehicle_type_df["Nutzwagenkilometer"] *= 52 / 1000000

        # Change the color palette to seaborns default color palette, but start at the 3rd color
        palette = sns.color_palette("Set2")

        df2 = vehicle_type_df.pivot(
            index="depot", columns="Fahrzeugtyp", values="Nutzwagenkilometer"
        )

        # Order:Single Decker, Double Decker, Articulated Bus
        df2 = df2[["Single Decker", "Articulated Bus", "Double Decker"]]
        # Plot the data
        df2.plot(kind="bar", stacked=True, ax=ax, color=palette)
        df2.to_pickle(f"{output_path}/vehicle_type_and_depot_pivoted_data.pkl")

        ax.set_title("")
        ax.set_ylabel(r"Revenue Mileage $\left[ \frac{km \times 10^6}{a} \right]$")

        # Remove xlabel, as the depots are already labeled on the x-axis
        ax.set_xlabel("")
        # 45 degree rotation for better readability
        plt.xticks(rotation=45)

        # Name Legend
        plt.legend(title="", bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper left", ncols=3)

        plt.tight_layout()
        # plt.subplots_adjust(top=0.8)
        plt.savefig(f"{output_path}/vehicle_type_and_depot_plot.png", dpi=300)
        plt.savefig(f"{output_path}/vehicle_type_and_depot_plot.pdf")
        plt.close()


@pipeline_step(
    step_name="reduce-to-one-day-two-depots",
    code_version="1.0.1",
)
def reduce_to_one_day_two_depots(db_path: Path):
    logger = logging.getLogger(__name__)
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        # find the day with the most trips
        # We can't use the day with the fewest trips since this might be just an overflow into the wee hours of the following day
        day_counts = (
            session.query(func.strftime("%Y-%m-%d", Trip.departure_time), func.count(Trip.id))
            .group_by(func.strftime("%Y-%m-%d", Trip.departure_time))
            .all()
        )
        day_with_most_trips = max(day_counts, key=lambda x: x[1])[0]
        logger.debug(f"Day with most trips is {day_with_most_trips}")

        # Identify all the places where rotations start or end.
        all_start_end_stations: Dict[Station, int] = defaultdict(int)
        all_rotations = (
            session.query(Rotation)
            .options(joinedload(Rotation.trips).joinedload(Trip.route))
            .all()
        )
        for rotation in all_rotations:
            if len(rotation.trips) == 0:
                raise ValueError(f"Rotation {rotation.name} has no trips")
            all_start_end_stations[rotation.trips[0].route.departure_station] += 1

        # Keep only the two depots with the fewest rotations starting or ending there
        depots_to_keep = sorted(all_start_end_stations.items(), key=lambda x: x[1])[:2]
        depot_station_ids_to_keep = [depot[0].id for depot in depots_to_keep]
        logger.debug(f"Keeping depots: {[depot[0].name for depot in depots_to_keep]}")

        all_rotations = (
            session.query(Rotation)
            .options(joinedload(Rotation.trips).joinedload(Trip.route))
            .all()
        )
        to_delete: List[Base] = []
        for rotation in tqdm(all_rotations, desc="Processing rotations"):
            if len(rotation.trips) == 0:
                raise ValueError(f"Rotation {rotation.name} has no trips")
            first_station_id = rotation.trips[0].route.departure_station_id
            date = rotation.trips[0].departure_time.date().isoformat()
            if first_station_id not in depot_station_ids_to_keep or date != day_with_most_trips:
                logger.debug(
                    f"Removing rotation {rotation.name}, vehicle type {rotation.vehicle_type.name_short}, start {rotation.trips[0].route.departure_station.name}, end {rotation.trips[-1].route.arrival_station.name}, date {date}"
                )
                for trip in rotation.trips:
                    for stop_time in trip.stop_times:
                        to_delete.append(stop_time)
                    to_delete.append(trip)
                to_delete.append(rotation)
        for obj in tqdm(to_delete, desc="Deleting objects"):
            session.delete(obj)
        session.commit()


@pipeline_step(
    step_name="add-temperatures-and-consumption",
    code_version="0.1.2",
    input_files=["../../..data/input/consumption_lut_gn.xlsx"],
)
def add_temperatures_and_consumption(db_path: Path):
    """
    Add temperature data and consumption data to the database.
    :param db_path: The path to the database to modify.
    :return: Nothing
    """
    logger = logging.getLogger(__name__)
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        CONSUMPTION_LUT_PATH = Path("../../../data/input/consumption_lut_gn.xlsx")
        if not CONSUMPTION_LUT_PATH.exists():
            raise ValueError(f"Consumption LUT file {CONSUMPTION_LUT_PATH} does not exist")

        with open(CONSUMPTION_LUT_PATH, "rb") as f:
            consumption_lut_gn = pd.read_excel(f)
            # The LUT is a 2D table. The first column is the average speed.
            # The first row contains the temperatures.
            # Turn it into a multi-indexed dataframe
            emp_temperatures = np.array(consumption_lut_gn.columns[1:]).astype(np.float64)
            emp_speeds = np.array(consumption_lut_gn.iloc[:, 0]).astype(np.float64)
            emp_data = np.array(consumption_lut_gn.iloc[:, 1:]).astype(np.float64)

        vehicle_type = session.query(VehicleType).filter(VehicleType.name_short == "GN").one()

        # Create a vehicle class for the vehicle type
        vehicle_class = VehicleClass(
            scenario_id=vehicle_type.scenario_id,
            name="Consumption LUT for GN",
            vehicle_types=[vehicle_type],
        )
        session.add(vehicle_class)

        # Create a LUT for the vehicle class
        # It is first filled with dummy values, then updated below
        consumption_lut = ConsumptionLut.from_vehicle_type(vehicle_type, vehicle_class)
        session.add(consumption_lut)

        new_coordinates = []
        new_values = []

        # Update the LUT with the empirical data
        incline = 0.0
        level_of_loading = 0.5
        for i, temperature in enumerate(emp_temperatures):
            for j, speed in enumerate(emp_speeds):
                # Interpolate the empirical data to the coordinates
                consumption = emp_data[i, j]
                if not np.isnan(consumption):
                    new_coordinates.append((incline, temperature, level_of_loading, speed))
                    new_values.append(consumption)
        consumption_lut.data_points = [
            [float(value) for value in coord] for coord in new_coordinates
        ]
        consumption_lut.values = [float(value) for value in new_values]

        # For the other vehicle types, just create dummy consumption LUTs, then multiply the values by a factor
        other_vehicle_types = (
            session.query(VehicleType).filter(VehicleType.id != vehicle_type.id).all()
        )
        for other_vehicle_type in other_vehicle_types:
            factor = 2
            vehicle_class = VehicleClass(
                scenario_id=other_vehicle_type.scenario_id,
                name=f"Consumption LUT for {other_vehicle_type.name_short}",
                vehicle_types=[other_vehicle_type],
            )
            session.add(vehicle_class)
            consumption_lut = ConsumptionLut.from_vehicle_type(other_vehicle_type, vehicle_class)
            session.add(consumption_lut)
            consumption_lut.data_points = [
                [float(value) for value in coord] for coord in new_coordinates
            ]
            consumption_lut.values = [float(value * factor) for value in new_values]

        session.flush()

        # Now, add the temperature data to all trips
        tz = ZoneInfo("Europe/Berlin")
        datetimes = [datetime(1971, 1, 1, tzinfo=tz), datetime(2037, 12, 21, tzinfo=tz)]
        temps = [-12, -12]

        try:
            scenarios = session.query(Scenario).all()
            for scenario in scenarios:
                scenario_temperatures = Temperatures(
                    scenario_id=scenario.id,
                    name="-12 °C",
                    use_only_time=False,
                    datetimes=datetimes,
                    data=temps,
                )

                session.add(scenario_temperatures)
        except:
            session.rollback()
            raise
        finally:
            session.commit()


@pipeline_step(
    step_name="vehicle-scheduling",
    code_version="0.1.2",
    input_files=None,
)
def vehicle_scheduling(
    db_path: Path,
    minimum_break_time: timedelta = timedelta(seconds=0),
    maximum_schedule_duration=timedelta(hours=24),
    battery_margin=0.1,
) -> None:
    progress_artifact_id = create_progress_artifact(
        progress=0.0,
        description="Starting vehicle scheduling",
    )
    logger = logging.getLogger(__name__)
    url = f"sqlite:///{db_path}"
    engine = create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenario_q = session.query(eflips.model.Scenario)
        if scenario_q.count() != 1:
            raise ValueError(f"Expected exactly one scenario, found {scenario_q.count()}")
        scenario = scenario_q.one()

        # Create a dictionary of the energy consumption for each rotation, tehn convert it to the format
        # eflips-opt expects.
        consumption: Dict[int, ConsumptionResult] = generate_consumption_result(scenario)

        # this is of the format {trip_id: ConsumptionResult, ...}, we need to convert it to
        # {trip_is: delta_soc, ...} and also turn the delta_soc into a positive number
        delta_socs: Dict[int, float] = {}
        for trip_id, result in consumption.items():
            delta_socs[trip_id] = -1 * result.delta_soc_total / (1 - battery_margin)

        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=0.1,
            description="Generated consumption results",
        )

        # Create the graph of all possible connections between trips
        # Internally, this is done separately for each vehicle type
        # We will also do it separately for each vehicle type, so we can log some progress information
        all_vehicle_types = session.query(VehicleType).join(Rotation).distinct().all()
        if len(all_vehicle_types) == 0:
            raise ValueError("No vehicle types found in the database")
        for i, vehicle_type in enumerate(all_vehicle_types):
            trips = (
                session.query(Trip)
                .join(Rotation)
                .filter(Rotation.vehicle_type_id == vehicle_type.id)
                .filter(Trip.trip_type == TripType.PASSENGER)
                .all()
            )
            graph = create_graph(
                trips=trips,
                delta_socs=delta_socs,
                maximum_schedule_duration=maximum_schedule_duration,
                minimum_break_time=minimum_break_time,
            )
            progress = ((3 * i + 1) / (3 * len(all_vehicle_types))) * 100
            update_progress_artifact(
                artifact_id=progress_artifact_id,
                progress=progress,
                description=f"Created graph for vehicle type {vehicle_type.name_short}",
            )

            rotation_plan = solve(graph)
            progress = ((3 * i + 2) / (3 * len(all_vehicle_types))) * 100
            update_progress_artifact(
                artifact_id=progress_artifact_id,
                progress=progress,
                description=f"Solved vehicle scheduling for vehicle type {vehicle_type.name_short}",
            )

            write_back_rotation_plan(rotation_plan, session)
            progress = ((3 * i + 3) / (3 * len(all_vehicle_types))) * 100
            update_progress_artifact(
                artifact_id=progress_artifact_id,
                progress=progress,
                description=f"Wrote back rotation plan for vehicle type {vehicle_type.name_short}",
            )

            logger.info(
                f"Created graph for vehicle type {vehicle_type.name_short} with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
            )


@pipeline_step(
    step_name="depot-assignment",
    code_version="0.1.1",
    input_files=None,
)
def depot_assignment(db_path: Path):
    logger = logging.getLogger(__name__)
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        scenario = session.query(Scenario).one()

        optimize_scenario(scenario, session)
        session.commit()


@pipeline_step(
    step_name="is-station-electrification-possible",
    code_version="0.1.0",
    input_files=None,
)
def is_station_electrification_possible(db_path: Path):
    logger = logging.getLogger(__name__)
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        scenario = session.query(Scenario).one()
        update_trip_loaded_masses(scenario, session)
        make_depot_stations_electrified(scenario, session)

        # Electrify ALL stations
        station_q = session.query(Station).filter(Station.scenario_id == scenario.id)
        station_q.update(
            {
                "is_electrified": True,
                "amount_charging_places": 100,
                "power_per_charger": 450,
                "power_total": 100 * 450,
                "charge_type": ChargeType.OPPORTUNITY,
                "voltage_level": VoltageLevel.MV,
            }
        )

        clear_previous_simulation_results(scenario, session)
        consumption_results = generate_consumption_result(scenario)
        simple_consumption_simulation(
            scenario, initialize_vehicles=True, consumption_result=consumption_results
        )

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
            f"Minimum SOC at end of day: {min_soc_end}, Electrified termini: {count_of_electrified_termini}"
        )
        if min_soc_end < 0:

            raise ValueError(
                "Scenario has rotations with SOC below 0% even with all stations electrified."
            )


@pipeline_step(
    step_name="do-station-electrification",
    code_version="0.1.0",
    input_files=None,
)
def do_station_electrification(db_path: Path):
    logger = logging.getLogger(__name__)
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        scenario = session.query(Scenario).one()

        # Set the loaded masses for correct consumption calculation
        update_trip_loaded_masses(scenario, session)

        # Make the depots electrified
        make_depot_stations_electrified(scenario, session)

        # Now, for the roations that do not need terminus charging, we also do not make them electirfiable
        # This way, our chargers are not hogged biy buses that do not need them
        remove_terminus_charging_from_okay_rotations(scenario, session)

        consumption_results = generate_consumption_result(scenario)

        while number_of_rotations_below_zero(scenario, session) > 0:
            # Log the current state and write it to an excel table for monitoring
            number_of_eletrified_termini = (
                session.query(Station)
                .filter(Station.scenario_id == scenario.id)
                .filter(Station.is_electrified == True)
                .count()
            )
            logger.info(
                f"Number of rotations with SOC below 0%: {number_of_rotations_below_zero(scenario, session)}, Number of electrified termini: {number_of_eletrified_termini}"
            )

            electrified_station_id = add_charging_station(scenario, session, power=450)
            logger.info(
                f"Added charging station {session.query(Station).filter(Station.id == electrified_station_id).one().name}"
                f" ({electrified_station_id}) to scenario"
            )

            # Run the consumption simulation again
            clear_previous_simulation_results(scenario, session)

            consumption_results = generate_consumption_result(scenario)
            simple_consumption_simulation(
                scenario, initialize_vehicles=True, consumption_result=consumption_results
            )

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
                f"Minimum SOC at end of day: {min_soc_end}, Electrified termini: {count_of_electrified_termini}"
            )
        session.commit()


@pipeline_step(
    step_name="run-simulation",
    code_version="0.1.0",
    input_files=None,
)
def run_simulation(db_path: Path):
    logger = logging.getLogger(__name__)
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        scenario = session.query(Scenario).one()

        #### Step 0: Update loaded masses and Make depot stations electrified #####
        update_trip_loaded_masses(scenario, session)
        make_depot_stations_electrified(scenario, session)

        ##### Step 1: Consumption simulation
        clear_previous_simulation_results(scenario, session)
        consumption_results = generate_consumption_result(scenario)
        simple_consumption_simulation(
            scenario, initialize_vehicles=True, consumption_result=consumption_results
        )

        ##### Step 2: Generate the depot layout
        generate_depot_optimal_size(
            scenario=scenario,
            charging_power=120,
            delete_existing_depot=True,
            use_consumption_lut=True,
        )

        ##### Step 3: Run the simulation
        # This can be done using eflips.api.run_simulation. Here, we use the three steps of
        # eflips.api.init_simulation, eflips.api.run_simulation, and eflips.api.add_evaluation_to_database
        # in order to show what happens "under the hood".
        simulation_host = init_simulation(
            scenario=scenario,
            session=session,
            repetition_period=None,
        )
        depot_evaluations = eflips.depot.api.run_simulation(simulation_host)

        add_evaluation_to_database(scenario, depot_evaluations, session)
        session.flush()
        session.expire_all()

        ##### Step 4: Consumption simulation
        consumption_results = generate_consumption_result(scenario)
        simple_consumption_simulation(
            scenario, initialize_vehicles=False, consumption_result=consumption_results
        )

        ##### Step 4.5: Insert dummy standby departure events
        # This is something we need to do due to an unclear reason. The depot simulation sometimes does not
        # generate the correct departure events for the vehicles. Therefore, we insert dummy standby departure events
        # for depot in session.query(Depot).filter(Depot.scenario_id == scenario.id).all():
        #    insert_dummy_standby_departure_events(
        #        depot.id, session, sim_time_end=END_OF_SIMULATION
        #    )

        ##### Step 5: Apply even smart charging
        # This step is optional. It can be used to apply even smart charging to the vehicles, reducing the peak power
        # consumption. This is done by shifting the charging times of the vehicles. The method is called
        # apply_even_smart_charging and is part of the eflips.depot.api module.
        apply_even_smart_charging(scenario)
        session.commit()

        print(f"Simulation complete for scenario {scenario.name_short}.")


@pipeline_step(
    step_name="tco-calculation",
    code_version="0.1.0",
    input_files=None,
)
def calculate_tco(db_path: Path):
    logger = logging.getLogger(__name__)
    url = f"sqlite:///{db_path}"
    engine = eflips.model.create_engine(url)
    with Session(engine) as session:
        # make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        scenario = session.query(Scenario).one()
        dict_tco_params = tco_parameters(scenario, session)
        init_tco_parameters(
            scenario=scenario,
            scenario_tco_parameters=dict_tco_params["scenario_tco_parameters"],
            vehicle_types=dict_tco_params["vehicle_types"],
            battery_types=dict_tco_params["battery_types"],
            charging_point_types=dict_tco_params["charging_point_types"],
            charging_infrastructure=dict_tco_params["charging_infrastructure"],
        )
        tco_calculator = TCOCalculator(scenario=scenario, energy_consumption_mode="constant")
        tco_calculator.calculate()
        result = tco_calculator.tco_by_type
        result["INFRASTRUCTURE"] += result.get("CHARGING_POINT", 0.0)
        result.pop("CHARGING_POINT", None)
        # Write the result to json file

        import json

        with open("tco_result.json", "w") as f:
            json.dump(result, f, indent=4)

        logger.info(f"Total Cost of Ownership (TCO): {tco_calculator.tco_unit_distance} EUR / km")
