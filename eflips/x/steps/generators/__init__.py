import logging
import socket
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Any, List, Tuple
from uuid import UUID

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

import gtfs_kit as gk  # type: ignore[import-untyped]
from eflips.ingest.gtfs import GtfsIngester as EflipsIngestGtfsIngester

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

    @classmethod
    def document_params(cls) -> Dict[str, str]:
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
            f"{cls.__name__}.multithreading": "Whether to use multithreading. Default is True.",
        }

    def generate(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
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
        assert isinstance(progress_artifact_id, UUID)

        TOTAL_STEPS = 11
        current_step = 0

        ### STEP 1: Load the XML files into memory
        # First, we go through all the files and load them into memory
        multithreading = params[f"{self.__class__.__name__}.multithreading"]
        logger.info(f"Using multithreading: {multithreading}")

        if multithreading:
            with Pool() as pool:
                schedules = pool.map(load_and_validate_xml, self.input_files)
        else:
            schedules = []
            for path in self.input_files:
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
        for schedule in schedules:
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
        for schedule in schedules:
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
        for create_route_result in create_route_results:
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
        for schedule in schedules:
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
        for station in stations_without_geom_q:
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
        for route in long_route_q:
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


class GTFSIngester(Generator):
    """
    Generator that ingests GTFS (General Transit Feed Specification) data into the eflips database.

    This class wraps the eflips.ingest.gtfs.GtfsIngester to provide integration with the eflips-x
    framework. It supports:
    - Single or multi-agency GTFS feeds
    - Automatic or manual date selection
    - Filtering by route type (bus only or all transit types)
    - DAY or WEEK duration imports
    """

    def __init__(
        self,
        input_files: List[Path],
        code_version: str = "v1",
        cache_enabled: bool = True,
    ):
        """
        Initialize the GTFS Ingester.

        :param input_files: List containing exactly one Path to a GTFS zip file
        :param code_version: Version string for cache invalidation
        :param cache_enabled: Whether to enable caching
        """
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)
        self.input_files = input_files

        if not all(isinstance(f, Path) for f in self.input_files):
            raise ValueError("All input_files must be of type pathlib.Path")
        if len(self.input_files) != 1:
            raise ValueError(
                f"GTFSIngester requires exactly one GTFS zip file, got {len(self.input_files)}"
            )
        if not all(f.exists() for f in self.input_files):
            missing_files = [str(f) for f in self.input_files if not f.exists()]
            raise ValueError(f"The following input files do not exist: {missing_files}")

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters accepted by this generator.

        :return: Dictionary mapping parameter names to descriptions
        """
        return {
            "log_level": "Logging level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO.",
            f"{cls.__name__}.agency_name": (
                "Name of the agency to import (required for multi-agency feeds). "
                "If the GTFS feed contains multiple agencies, you must specify which one to import. "
                "If not specified for a single-agency feed, the sole agency will be used automatically."
            ),
            f"{cls.__name__}.start_date": (
                "Start date for import in ISO 8601 format (YYYY-MM-DD). "
                "If not specified, a Monday in the middle of the feed's validity period will be automatically selected."
            ),
            f"{cls.__name__}.duration": (
                "Duration of import period. Either 'DAY' (import one day) or 'WEEK' (import one week). "
                "Default is 'WEEK'."
            ),
            f"{cls.__name__}.bus_only": (
                "If True (default), only import bus routes (route_type 3 or 700-799). "
                "If False, import all transit types (rail, subway, bus, ferry, etc.)."
            ),
        }

    @staticmethod
    def _auto_select_start_date(gtfs_zip_file: Path) -> str:
        """
        Auto-select a start date (Monday) in the middle of the GTFS feed validity period.

        This method:
        1. Reads the GTFS feed to determine its validity period
        2. Calculates the midpoint date
        3. Finds the nearest Monday on or after the midpoint
        4. Returns the date in ISO 8601 format (YYYY-MM-DD)

        :param gtfs_zip_file: Path to the GTFS zip file
        :return: ISO 8601 formatted date string (YYYY-MM-DD)
        :raises ValueError: If the feed has no calendar data or is invalid
        """
        # Read the GTFS feed
        feed = gk.read_feed(gtfs_zip_file, dist_units="m")

        # Get validity period
        validity = EflipsIngestGtfsIngester.get_feed_validity_period(feed)
        if validity is None:
            raise ValueError(
                "Cannot auto-select date: GTFS feed has no calendar data. "
                "Please specify start_date manually."
            )

        start_date_str, end_date_str = validity

        # Parse GTFS dates (YYYYMMDD format)
        start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
        end_date = datetime.strptime(end_date_str, "%Y%m%d").date()

        # Calculate midpoint
        midpoint = start_date + (end_date - start_date) / 2

        # Find the nearest Monday on or after the midpoint
        # weekday() returns 0 for Monday, 6 for Sunday
        days_until_monday = (7 - midpoint.weekday()) % 7
        monday_date = midpoint + timedelta(days=days_until_monday)

        # Make sure the selected Monday + 6 days (for a week) is within the validity period
        week_end = monday_date + timedelta(days=6)
        if week_end > end_date:
            # If the week extends beyond the validity period, move back to an earlier Monday
            days_to_move_back = (week_end - end_date).days
            # Round up to the nearest week
            weeks_to_move_back = (days_to_move_back + 6) // 7
            monday_date = monday_date - timedelta(weeks=weeks_to_move_back)

        # Ensure the Monday is not before the start date
        if monday_date < start_date:
            # Find the first Monday on or after the start date
            days_until_monday = (7 - start_date.weekday()) % 7
            monday_date = start_date + timedelta(days=days_until_monday)

        return monday_date.strftime("%Y-%m-%d")

    def generate(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """
        Generate database content from GTFS data.

        This method:
        1. Extracts parameters from the params dict
        2. Auto-selects start date if not provided
        3. Creates an instance of the eflips-ingest GtfsIngester
        4. Calls prepare() to validate and prepare the data
        5. Calls ingest() to load the data into the database

        :param session: SQLAlchemy session for database operations
        :param params: Dictionary of parameters (see document_params for details)
        :raises ValueError: If parameters are invalid or preparation fails
        """
        # Configure logging
        log_level = params.get("log_level", "INFO")
        match log_level:
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

        # Set up a progress artifact for tracking
        progress_artifact_id = create_progress_artifact(
            progress=0.0, key=self.__class__.__name__.lower()
        )
        assert isinstance(progress_artifact_id, UUID)

        # Get GTFS zip file
        gtfs_zip_file = self.input_files[0]
        logger.info(f"Processing GTFS file: {gtfs_zip_file}")

        # Get parameters
        agency_name = params.get(f"{self.__class__.__name__}.agency_name", "")
        start_date = params.get(f"{self.__class__.__name__}.start_date")
        duration = params.get(f"{self.__class__.__name__}.duration", "WEEK")
        bus_only = params.get(f"{self.__class__.__name__}.bus_only", True)

        # Auto-select start date if not provided
        if not start_date or start_date == "":
            logger.info(
                "No start_date specified, auto-selecting Monday in middle of validity period"
            )
            start_date = self._auto_select_start_date(gtfs_zip_file)
            logger.info(f"Auto-selected start_date: {start_date}")

        # Extract database URL from session
        if session.bind is None:
            raise ValueError("Session has no bound engine or connection")
        # Handle both Engine and Connection types
        from sqlalchemy.engine import Engine

        bind = session.bind
        if isinstance(bind, Engine):
            db_url = str(bind.url)
        else:
            # Connection type - get URL from the engine
            db_url = str(bind.engine.url)
        logger.debug(f"Database URL: {db_url}")

        # Create eflips-ingest GtfsIngester instance
        gtfs_ingester = EflipsIngestGtfsIngester(database_url=db_url)

        # Create progress callbacks that map to 0-50% for prepare, 50-100% for ingest
        def prepare_progress_callback(progress: float) -> None:
            """Map prepare progress (0-1) to overall progress (0-50%)."""
            overall_progress = progress * 50.0
            update_progress_artifact(
                artifact_id=progress_artifact_id,
                progress=overall_progress,
                description=f"Preparing GTFS data: {overall_progress:.1f}%",
            )

        def ingest_progress_callback(progress: float) -> None:
            """Map ingest progress (0-1) to overall progress (50-100%)."""
            overall_progress = 50.0 + (progress * 50.0)
            update_progress_artifact(
                artifact_id=progress_artifact_id,
                progress=overall_progress,
                description=f"Ingesting GTFS data: {overall_progress:.1f}%",
            )

        # Prepare the data
        logger.info("Calling prepare() on GtfsIngester")
        success, result = gtfs_ingester.prepare(
            gtfs_zip_file=gtfs_zip_file,
            start_date=start_date,
            progress_callback=prepare_progress_callback,
            duration=duration,
            agency_name=agency_name,
            bus_only=bus_only,
        )

        if not success:
            # prepare() returned errors
            assert isinstance(result, dict)
            error_messages = "\n".join(f"  - {key}: {msg}" for key, msg in result.items())
            raise ValueError(f"GTFS preparation failed:\n{error_messages}")

        # Get the UUID from prepare
        ingestion_uuid = result
        assert isinstance(ingestion_uuid, UUID)
        logger.info(f"Preparation successful. UUID: {ingestion_uuid}")

        # Ingest the data
        logger.info("Calling ingest() on GtfsIngester")
        gtfs_ingester.ingest(
            uuid=ingestion_uuid, always_flush=False, progress_callback=ingest_progress_callback
        )

        # Mark as 100% complete
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=100.0,
            description="GTFS ingestion completed successfully",
        )
        logger.info("GTFS ingestion completed successfully")
