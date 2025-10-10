import logging
import socket
import warnings
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Any, List, Tuple

import sqlalchemy.orm.session
from eflips.ingest.legacy.bvgxml import (
    load_and_validate_xml,
    create_stations,
    TimeProfile,
    create_routes_and_time_profiles,
    create_trip_prototypes,
    create_trips_and_vehicle_schedules,
    merge_identical_stations,
    merge_identical_rotations,
    identify_and_delete_overlapping_rotations,
    recenter_station,
)
from eflips.ingest.legacy.xmldata import Linienfahrplan
from eflips.model import Scenario, Route, ConsistencyWarning, Station, AssocRouteStation
from geoalchemy2.functions import ST_Distance
from prefect.artifacts import create_progress_artifact, update_progress_artifact
from sqlalchemy import func, text
from tqdm.auto import tqdm

from eflips.x.framework import Generator


class BVGXMLIngester(Generator):
    def __init__(
        self,
        input_files: List[Path],
        code_version: str = "v1",
        cache_enabled: bool = True,
    ):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)
        self.input_files = input_files

        if not all(isinstance(f, Path) for f in self.input_files):
            raise ValueError("All input_files must be of type pathlib.Path")
        if not all(f.exists() for f in self.input_files):
            missing_files = [str(f) for f in self.input_files if not f.exists()]
            raise ValueError(f"The following input files do not exist: {missing_files}")

    def document_params(self) -> Dict[str, str]:
        """
        This method documents the parameters of the generator. It returns a dictionary where the keys are the parameter
        and the values are a description of the parameter. The values may use markdown formatting. They may be
        multi-line strings.
        If the parameters are specific to a subclass, the key should be prefixed with the class name and a dot.
        For example, if the class is MyGenerator and the parameter is my_param, the key should be MyGenerator.my_param.
        :return: A dictionary documenting the parameters of the generator.
        """
        return {
            "log_level": "Logging level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.",
            f"{self.__class__.__name__}.multithreading": "Whether to use multithreading. Default is True.",
        }

    def generate(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Path:
        match params["log_level"] or "INFO":
            case "DEBUG":
                logging.basicConfig(level=logging.DEBUG)
            case "INFO":
                logging.basicConfig(level=logging.INFO)
            case "WARNING":
                logging.basicConfig(level=logging.WARNING)
            case "ERROR":
                logging.basicConfig(level=logging.ERROR)
            case "CRITICAL":
                logging.basicConfig(level=logging.CRITICAL)
            case _:
                raise ValueError(
                    "Invalid log level. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL"
                )
        logger = logging.getLogger(__name__)

        # Set up an empty progress artifact
        progress_artifact_id = create_progress_artifact(
            progress=0.0, key=self.__class__.__name__.lower()
        )

        TOTAL_STEPS = 11
        current_step = 0

        ### STEP 1: Load the XML files into memory
        # First, we go through all the files and load them into memory
        multithreading = params[f"{self.__class__.__name__}.multithreading"]
        logger.info(f"Using multithreading: {multithreading}")
        schedules = []

        if multithreading:
            with Pool() as pool:
                for schedule in tqdm(
                    pool.imap_unordered(load_and_validate_xml, self.input_files),
                    total=len(self.input_files),
                    desc=f"(1/{TOTAL_STEPS}) Loading XML files",
                ):
                    schedules.append(schedule)
        else:
            schedules = []
            for path in tqdm(self.input_files, desc=f"(1/{TOTAL_STEPS}) Loading XML files"):
                schedules.append(load_and_validate_xml(path))

        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Loaded XML files",
        )
        logger.info(f"Loaded {len(schedules)} schedules from XML files")

        ### STEP 1.5: Create the scenario
        scenario = Scenario(
            name=f"Created by BVG-XML Ingestion on {socket.gethostname()} at {datetime.now().isoformat()}"
        )
        session.add(scenario)
        session.flush()
        scenario_id = scenario.id

        ### STEP 2: Create the stations
        # Now, we go through the schedules and create the stations
        # No multithreading, because that would just create duplicate stations
        for schedule in tqdm(schedules, desc=f"(2/{TOTAL_STEPS}) Creating stations"):
            create_stations(schedule, scenario_id, session)
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Created stations",
        )
        logger.info("Created stations")

        ### STEP 3: Create the routes and save some data for later
        # Again no multithreading
        create_route_results: List[
            Tuple[
                Linienfahrplan,
                Dict[int, Dict[int, List[TimeProfile.TimeProfilePoint]]],
                Dict[int, None | Route],
            ]
        ] = []
        for schedule in tqdm(schedules, desc=f"(3/{TOTAL_STEPS}) Creating routes"):
            trip_time_profiles, db_routes_by_lfd_nr = create_routes_and_time_profiles(
                schedule, scenario_id, session
            )
            create_route_results.append((schedule, trip_time_profiles, db_routes_by_lfd_nr))
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Created routes",
        )
        logger.info("Created routes")

        ### STEP 4: Create the trip prototypes
        # This can be done in parallel, but we don't need to do it, it's fast enough
        all_trip_protoypes: List[Dict[int, None | TimeProfile]] = []
        for create_route_result in tqdm(
            create_route_results, desc=f"(4/{TOTAL_STEPS}) Creating trip prototypes"
        ):
            trip_prototypes = create_trip_prototypes(
                create_route_result[0], create_route_result[1], create_route_result[2]
            )
            all_trip_protoypes.append(trip_prototypes)

        # Unify the dictionaries, making sure the contents are the same if there is a duplicate key
        trip_prototypes = {}
        for the_dict in all_trip_protoypes:
            for fahrt_id, time_profile in the_dict.items():
                if fahrt_id in trip_prototypes:
                    if trip_prototypes[fahrt_id] != time_profile:
                        raise ValueError(
                            f"Trip {fahrt_id} has two different time profiles in different schedules"
                        )
                else:
                    trip_prototypes[fahrt_id] = time_profile
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Created trip prototypes",
        )
        logger.info("Created trip prototypes")

        ### STEP 5: Create the trips and vehicle schedules
        for schedule in tqdm(
            schedules, desc=f"(5/{TOTAL_STEPS}) Creating trips and vehicle schedules"
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConsistencyWarning)
                create_trips_and_vehicle_schedules(schedule, trip_prototypes, scenario_id, session)
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Created trips and vehicle schedules",
        )
        logger.info("Created trips and vehicle schedules")

        ### STEP 6: Set the geom of the stations
        # No multithreading, because it should be fast enough
        stations_without_geom_q = (
            session.query(Station)
            .join(AssocRouteStation)
            .filter(Station.scenario_id == scenario_id)
            .distinct(Station.id)
        )
        for station in tqdm(
            stations_without_geom_q,
            desc=f"(6/{TOTAL_STEPS}) Setting station geom",
            total=stations_without_geom_q.count(),
        ):
            # Get the median of the assoc_route_stations
            recenter_station(station, session)

        # Flush the session to convert the geoms from string to binary
        session.flush()
        session.expire_all()
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Set station geom",
        )
        logger.info("Set station geom")

        ### STEP 7: Fix the routes with very large distances:
        # There are some routes which have a distance of zero even once the last point is reached
        # We set their distance to a very large number. Now we set it to the geometric distance between the first and last
        # point
        long_route_q = (
            session.query(Route)
            .filter(Route.scenario_id == scenario_id)
            .filter(Route.distance >= 1e6 * 1000)
        )
        for route in tqdm(
            long_route_q, desc=f"(7/{TOTAL_STEPS}) Fixing long routes", total=long_route_q.count()
        ):
            first_point = route.departure_station.geom
            last_point = route.arrival_station.geom

            first_point_soldner = func.ST_Transform(first_point, 3068)
            last_point_soldner = func.ST_Transform(last_point, 3068)
            dist_q = ST_Distance(first_point_soldner, last_point_soldner)

            dist = session.query(dist_q).one()[0]

            with session.no_autoflush:
                route.distance = dist
                route.assoc_route_stations[-1].elapsed_distance = dist
            route.name = "CHECK DISTANCE: " + route.name

        session.flush()
        session.expire_all()
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Fixed long routes",
        )
        logger.info("Fixed long routes")

        # STEP 8: Merge identical stations
        print(f"(8/{TOTAL_STEPS}) Merging identical stations")
        merge_identical_stations(scenario_id, session)

        session.flush()
        session.expire_all()
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Merged identical stations",
        )

        # STEP 9: Combine rotations with the same name
        print(f"(9/{TOTAL_STEPS}) Merging identical rotations")
        merge_identical_rotations(scenario_id, session)
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Merged identical rotations",
        )
        logger.info("Merged identical rotations")

        # STEP 10: Identify overlapping rotations
        print(f"(10/{TOTAL_STEPS}) Identifying and deleting overlapping rotations")
        identify_and_delete_overlapping_rotations(scenario_id, session)
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Identified and deleted overlapping rotations",
        )
        logger.info("Identified and deleted overlapping rotations")

        # STEP 11: Fix the max sequence numbers
        print(f"(11/{TOTAL_STEPS}) Fixing max sequence numbers")
        TABLES = [
            "Scenario",
            "Plan",
            "Process",
            "BatteryType",
            "VehicleClass",
            "Line",
            "Station",
            "Depot",
            "AssocPlanProcess",
            "VehicleType",
            "Route",
            "Area",
            "Vehicle",
            "AssocVehicleTypeVehicleClass",
            "AssocRouteStation",
            "AssocAreaProcess",
            "Rotation",
            "Trip",
            "Event",
            "StopTime",
        ]

        for table_name in TABLES:
            # Read the maximum id from the table
            result = session.execute(text(f'SELECT MAX(id) FROM "{table_name}"'))
            max_id = result.scalar()  # scalar() returns None if no rows or if MAX(id) is NULL

            if max_id is not None:
                # Update the sqlite_sequence table
                session.execute(
                    text(
                        f'UPDATE sqlite_sequence SET seq = {max_id+1} WHERE name = "{table_name}"'
                    )
                )
        current_step += 1
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=current_step / (TOTAL_STEPS / 100),
            description="Fixed max sequence numbers",
        )
        logger.info("Fixed max sequence numbers")
