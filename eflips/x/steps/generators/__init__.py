from __future__ import annotations

import logging
import shutil
import tempfile
import warnings
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple
from uuid import UUID

import gtfs_kit as gk  # type: ignore[import-untyped]
import sqlalchemy.orm.session
from eflips.ingest.bvgxml import BvgxmlIngester
from eflips.ingest.gtfs import GtfsIngester as EflipsIngestGtfsIngester
from eflips.model import (
    Scenario,
    Route,
    ConsistencyWarning,
    Trip,
    Rotation,
)
from prefect.artifacts import create_progress_artifact, update_progress_artifact

from eflips.x.framework import Generator

if TYPE_CHECKING:
    from eflips.x.framework import PipelineContext


class BVGXMLIngester(Generator):
    """
    Generator that ingests BVG-XML ``Linienfahrplan`` files into a new database.

    Thin wrapper around :class:`eflips.ingest.bvgxml.BvgxmlIngester`. eflips-ingest 2.x
    replaced the loose ``eflips.ingest.legacy.bvgxml`` functions this step used to
    orchestrate itself with a two-phase ingester built on the standard ingester API:

    - ``prepare()`` takes a **zip** of ``*.xml`` files, parses and validates each one,
      and merges them into a single corpus so that contradictions between files surface
      before anything is written. It returns either a UUID naming the staged data or a
      dict of field errors.
    - ``ingest()`` resolves that corpus (network, routes, rotations, schedule) and writes
      it to the database in one insert-only pass, then fixes the id sequences.

    The merging, deduplication and station recentering that this step used to drive
    itself are part of that resolution now, so they are no longer done here.

    Input files may be given either as individual ``*.xml`` paths (they are zipped into
    a temporary archive) or as a single ``*.zip``, which is passed through untouched.
    """

    def __init__(
        self,
        input_files: List[Path],
        code_version: str = "v2",
        cache_enabled: bool = True,
    ):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)
        self.input_files = input_files

        if not all(isinstance(f, Path) for f in self.input_files):
            raise ValueError("All input_files must be of type pathlib.Path")
        if not all(f.exists() for f in self.input_files):
            missing_files = [str(f) for f in self.input_files if not f.exists()]
            raise ValueError(f"The following input files do not exist: {missing_files}")
        if not self.input_files:
            raise ValueError("At least one input file must be provided")

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters of this generator.

        The ``multithreading`` parameter is gone: eflips-ingest 2.x parallelises the
        parse internally, so there is nothing left for the caller to choose.

        :return: A dictionary documenting the parameters of the generator.
        """
        return {
            "log_level": (
                "Logging level. One of DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO."
            ),
        }

    def _zip_for_ingest(self, work_dir: Path) -> Path:
        """Return a zip of the input files, creating one if they are loose XML files.

        ``BvgxmlIngester.prepare`` only accepts a zip, and rejects a zip that contains
        another zip rather than unpacking it, so a single ``*.zip`` input is passed
        straight through instead of being re-wrapped.
        """
        if len(self.input_files) == 1 and self.input_files[0].suffix.lower() == ".zip":
            return self.input_files[0]

        zip_path = work_dir / "bvgxml_input.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for xml_file in self.input_files:
                # Flatten into the archive root: prepare() globs recursively, but a flat
                # layout keeps the extracted tree predictable.
                archive.write(xml_file, arcname=xml_file.name)
        return zip_path

    def generate(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """
        Ingest the BVG-XML input files into the session's database.

        :param session: Session on the database to populate. Note that the ingester opens
            its *own* connection to the same database and commits there, so this session
            is rolled back first to make sure it is not holding a transaction, and
            expired afterwards so it sees the newly written rows.
        :param params: Pipeline parameters.
        :raises ValueError: If the input files cannot be prepared for ingestion.
        """
        # PipelineStep.set_log_level() (called from PipelineStep.execute()) has
        # already configured logging from params["log_level"] before generate()
        # runs. Still validate the value so direct callers (e.g. tests) get the
        # same error contract.
        self._validate_log_level_param(params)
        logger = logging.getLogger(__name__)

        progress_artifact_id = create_progress_artifact(
            progress=0.0, key=self.__class__.__name__.lower()
        )
        assert isinstance(progress_artifact_id, UUID)

        # get_bind() is typed as Engine | Connection; .engine narrows both to the Engine.
        database_url = session.get_bind().engine.url.render_as_string(hide_password=False)

        # The ingester writes through its own session and commits. Release anything this
        # session may be holding so the two do not contend for the SQLite write lock.
        session.rollback()

        def report(offset: float, description: str) -> Callable[[float], None]:
            """Map an ingester progress fraction onto half of the Prefect progress bar."""

            def callback(fraction: float) -> None:
                update_progress_artifact(
                    artifact_id=progress_artifact_id,
                    progress=offset + 50.0 * fraction,
                    description=description,
                )

            return callback

        ingester = BvgxmlIngester(database_url)

        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = self._zip_for_ingest(Path(temp_dir))
            logger.info("Preparing %d BVG-XML input file(s) for ingestion", len(self.input_files))

            success, result = ingester.prepare(
                zip_path,
                progress_callback=report(0.0, "Parsing and merging XML files"),
            )
            if not success:
                raise ValueError(f"BVG-XML preparation failed: {result}")
            assert isinstance(result, UUID)

            try:
                logger.info("Ingesting prepared BVG-XML corpus %s", result)
                ingester.ingest(
                    result,
                    progress_callback=report(50.0, "Writing scenario to the database"),
                )
            finally:
                # prepare() stages a pickle under the system temp dir keyed by UUID and
                # nothing else cleans it up; for a full-city import that is hundreds of MB.
                shutil.rmtree(ingester.path_for_uuid(result), ignore_errors=True)

        # The rows were written on the ingester's connection, so drop anything this
        # session has cached before the framework commits and hands it on.
        session.expire_all()

        scenario = session.query(Scenario).one()
        logger.info("Ingested BVG-XML data into scenario '%s'", scenario.name)

        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=100.0,
            description="Ingestion complete",
        )


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
        code_version: str = "v4",
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
                "Name of the agency to import, or a list of names to combine into one import "
                "(required for multi-agency feeds unless agency_ids is given). "
                "Exact match against agency.txt is required."
            ),
            f"{cls.__name__}.agency_ids": (
                "agency_id (or list of agency_ids) to import from a multi-agency feed. "
                "Preferred over agency_name because ids are unambiguous. When both are given, "
                "the resulting set is the union."
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
    def _coerce_str_list(value: str | Iterable[str] | None) -> List[str]:
        """Normalise a scalar/iterable/None into a list of non-empty strings.

        Accepts the flexible shapes our params and eflips-ingest both use:
        ``""`` / ``None`` → ``[]``; a single string → ``[s]``; any iterable of
        strings → a list (with empty/None entries dropped).
        """
        if value is None or value == "":
            return []
        if isinstance(value, str):
            return [value]
        return [str(v) for v in value if v not in (None, "")]

    @staticmethod
    def _resolve_agency_ids(
        feed: "gk.Feed", agency_names: List[str], agency_ids: List[str]
    ) -> List[str]:
        """
        Resolve the given names and ids against ``feed.agency`` and return the
        union as a list of agency_id strings. Raises ``ValueError`` if any
        requested name/id is not present in the feed.
        """
        if feed.agency is None or len(feed.agency) == 0:
            raise ValueError("GTFS feed has no agency table")

        feed_names = set(feed.agency["agency_name"].astype(str))
        feed_ids = set(feed.agency["agency_id"].astype(str))

        missing_names = [n for n in agency_names if n not in feed_names]
        missing_ids = [i for i in agency_ids if i not in feed_ids]
        if missing_names or missing_ids:
            raise ValueError(
                f"Agency not found in GTFS feed. "
                f"Missing names: {missing_names}, missing ids: {missing_ids}. "
                f"Available agencies: {', '.join(sorted(feed_names))}"
            )

        resolved: set[str] = set(agency_ids)
        if agency_names:
            by_name = feed.agency[feed.agency["agency_name"].isin(agency_names)]
            resolved.update(by_name["agency_id"].astype(str).tolist())
        return sorted(resolved)

    @staticmethod
    def _compute_agency_active_dates(
        feed: "gk.Feed", agency_ids: List[str], bus_only: bool
    ) -> List[date]:
        """
        Build the sorted list of dates on which the given agency has at least one
        active service, considering both ``calendar.txt`` (with day-of-week filtering)
        and ``calendar_dates.txt`` exceptions.

        When ``bus_only`` is True, routes are restricted to bus route types
        (``route_type == 3`` or ``700 <= route_type <= 799``) *before* computing
        service_ids. This mirrors eflips-ingest's ``filter_feed_by_route_type`` and
        is essential for mixed-mode feeds where an agency operates both rail and
        bus service: without the filter we might pick a date on which only the
        (later-filtered-out) rail service runs.

        Handles the common edge cases:

        - ``calendar.txt`` row with all day-of-week flags = 0 contributes no dates
          (e.g. placeholder rows for agencies whose service is defined entirely via
          ``calendar_dates.txt`` additions).
        - Missing ``calendar.txt`` or missing ``calendar_dates.txt`` is fine — only
          the present table contributes.
        - ``exception_type=2`` removes a date that ``calendar.txt`` would otherwise
          have generated.

        :param feed: A gtfs_kit Feed object
        :param agency_ids: The agency_ids (as strings) to filter on. Assumed to
            already be validated against ``feed.agency``.
        :param bus_only: If True, restrict to bus route types before computing
            active dates. Must match the ``bus_only`` flag passed to eflips-ingest
            downstream, otherwise the selected date may fall on routes that will
            later be filtered out.
        :return: Sorted list of unique active dates (empty if none)
        """
        if feed.agency is None or feed.routes is None or feed.trips is None:
            return []

        # Routes → service_ids belonging to this agency set.
        # Compare as strings because agency_id columns in GTFS may be int- or
        # str-typed depending on the feed.
        agency_routes = feed.routes[feed.routes["agency_id"].astype(str).isin(agency_ids)]
        if bus_only:
            # Same predicate as eflips-ingest's filter_feed_by_route_type:
            # route_type 3 (standard bus) or 700-799 (extended bus services).
            rt = agency_routes["route_type"]
            agency_routes = agency_routes[(rt == 3) | ((rt >= 700) & (rt <= 799))]
        route_ids = set(agency_routes["route_id"])
        if not route_ids:
            return []

        agency_trips = feed.trips[feed.trips["route_id"].isin(route_ids)]
        service_ids = set(agency_trips["service_id"].unique())
        if not service_ids:
            return []

        active: set[date] = set()
        dow_cols = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]

        # Step 1: expand calendar.txt rows that have at least one active day-of-week
        if feed.calendar is not None and not feed.calendar.empty:
            agency_cal = feed.calendar[feed.calendar["service_id"].isin(service_ids)]
            for _, row in agency_cal.iterrows():
                # Skip placeholder rows where every day-of-week flag is 0
                day_flags = [int(row[c]) for c in dow_cols if c in row.index]
                if not any(day_flags):
                    continue
                cal_start = datetime.strptime(str(int(row["start_date"])), "%Y%m%d").date()
                cal_end = datetime.strptime(str(int(row["end_date"])), "%Y%m%d").date()
                # Iterate days in the range and add ones whose weekday is enabled
                d = cal_start
                while d <= cal_end:
                    if day_flags[d.weekday()] == 1:
                        active.add(d)
                    d += timedelta(days=1)

        # Step 2: apply calendar_dates.txt exceptions
        if feed.calendar_dates is not None and not feed.calendar_dates.empty:
            agency_cd = feed.calendar_dates[feed.calendar_dates["service_id"].isin(service_ids)]
            for _, row in agency_cd.iterrows():
                d = datetime.strptime(str(int(row["date"])), "%Y%m%d").date()
                exc = int(row["exception_type"])
                if exc == 1:
                    active.add(d)
                elif exc == 2:
                    active.discard(d)

        return sorted(active)

    @staticmethod
    def _auto_select_start_date(
        gtfs_zip_file: Path,
        *,
        agency_name: str | Iterable[str] = "",
        agency_ids: str | Iterable[str] = "",
        bus_only: bool = True,
    ) -> str:
        """
        Auto-select a Monday near the middle of the agency set's active service window.

        When ``agency_name`` or ``agency_ids`` is non-empty, the validity period is
        computed from only the services actually used by the selected agencies'
        trips, considering both ``calendar.txt`` (filtered by day-of-week) and
        ``calendar_dates.txt`` exceptions. When ``bus_only`` is True, only bus
        route types (3 or 700-799) are considered, matching the downstream
        eflips-ingest filter.

        When neither is provided, falls back to the global feed validity period.

        :param gtfs_zip_file: Path to the GTFS zip file
        :param agency_name: Agency name or list of names to scope to
        :param agency_ids: Agency id or list of ids to scope to
        :param bus_only: Whether bus-only filtering will be applied downstream.
        :return: ISO 8601 formatted date string (YYYY-MM-DD)
        :raises ValueError: If the feed has no calendar data, any requested
            agency is missing, or the resolved agency set has no active service.
        """
        names = GTFSIngester._coerce_str_list(agency_name)
        ids = GTFSIngester._coerce_str_list(agency_ids)

        feed = gk.read_feed(gtfs_zip_file, dist_units="m")

        start_date: date
        end_date: date
        target: date
        active_set: set[date] | None

        if names or ids:
            # Agency-aware path. _resolve_agency_ids raises a clear
            # "agency not found" error if any requested name/id is missing,
            # distinct from the "no active services" case below.
            resolved_ids = GTFSIngester._resolve_agency_ids(feed, names, ids)
            active_dates = GTFSIngester._compute_agency_active_dates(
                feed, resolved_ids, bus_only=bus_only
            )
            if not active_dates:
                scope = "bus routes" if bus_only else "routes"
                raise ValueError(
                    f"Cannot auto-select date: agency/agencies {names or ids} have no "
                    f"active service dates for their {scope} in the GTFS feed (checked "
                    "both calendar.txt and calendar_dates.txt). "
                    "Please specify start_date manually."
                )
            start_date = active_dates[0]
            end_date = active_dates[-1]
            # Median date — robust against sparse / clustered service patterns
            target = active_dates[len(active_dates) // 2]
            active_set = set(active_dates)
        else:
            # Fallback: global feed validity period
            validity = EflipsIngestGtfsIngester.get_feed_validity_period(feed)
            if validity is None:
                raise ValueError(
                    "Cannot auto-select date: GTFS feed has no calendar data. "
                    "Please specify start_date manually."
                )
            start_date_str, end_date_str = validity
            start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
            end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
            target = start_date + (end_date - start_date) / 2
            active_set = None

        # Find the Monday on or before the target date
        monday_date: date = target - timedelta(days=target.weekday())

        # Make sure Mon-Sun fits within the validity period
        week_end = monday_date + timedelta(days=6)
        if week_end > end_date:
            weeks_to_move_back = ((week_end - end_date).days + 6) // 7
            monday_date -= timedelta(weeks=weeks_to_move_back)
        if monday_date < start_date:
            days_until_monday = (7 - start_date.weekday()) % 7
            monday_date = start_date + timedelta(days=days_until_monday)

        # Defensive: in the agency-aware path, ensure the chosen week intersects
        # the active-dates set. Walk backward by one week if not (shouldn't trigger
        # for normal feeds, since target itself is an active date).
        if active_set is not None:
            set_for_closure = active_set  # narrow for nested function

            def week_has_service(mon: date) -> bool:
                return any((mon + timedelta(days=i)) in set_for_closure for i in range(7))

            safety = 0
            while not week_has_service(monday_date) and monday_date >= start_date:
                monday_date -= timedelta(weeks=1)
                safety += 1
                if safety > 520:  # 10 years of weeks — should never happen
                    break

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
        # PipelineStep.set_log_level() (called from PipelineStep.execute()) has
        # already configured logging from params["log_level"] before generate()
        # runs, so no per-step match/case is needed here. Still validate the
        # value so direct callers (e.g. tests) get the same error contract.
        self._validate_log_level_param(params)
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
        agency_ids = params.get(f"{self.__class__.__name__}.agency_ids", "")
        start_date = params.get(f"{self.__class__.__name__}.start_date")
        duration = params.get(f"{self.__class__.__name__}.duration", "WEEK")
        bus_only = params.get(f"{self.__class__.__name__}.bus_only", True)

        # Auto-select start date if not provided
        if not start_date or start_date == "":
            logger.info(
                "No start_date specified, auto-selecting Monday in middle of validity period"
            )
            start_date = self._auto_select_start_date(
                gtfs_zip_file,
                agency_name=agency_name,
                agency_ids=agency_ids,
                bus_only=bus_only,
            )
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
            agency_id=agency_ids,
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

        # Verify that the ingestion actually produced trips
        trip_count = session.query(Trip).count()
        rotation_count = session.query(Rotation).count()
        if trip_count == 0:
            raise ValueError(
                f"GTFS ingestion produced 0 trips for agency name(s) {agency_name!r} "
                f"id(s) {agency_ids!r} with start_date={start_date}, duration={duration}. "
                f"The selected date range may fall outside this agency's active service period. "
                f"({rotation_count} empty rotations were created.)"
            )

        # Mark as 100% complete
        update_progress_artifact(
            artifact_id=progress_artifact_id,
            progress=100.0,
            description="GTFS ingestion completed successfully",
        )
        logger.info(
            f"GTFS ingestion completed successfully: {trip_count} trips, {rotation_count} rotations"
        )


class CopyCreator(Generator):
    """
    Lightweight generator that copies an existing database file to create a new pipeline step.

    This generator is useful for branching workflows where you want to start from an existing
    database without using a Modifier (which would require cache invalidation based on the
    input database hash). Instead, CopyCreator treats the source database as an input file,
    similar to how other generators treat XML or GTFS files.

    The copy operation bypasses the normal Generator session creation to avoid unnecessary
    overhead when no actual generation logic needs to run.
    """

    def __init__(
        self,
        input_files: List[Path],
        code_version: str = "v1",
        cache_enabled: bool = True,
    ):
        """
        Initialize the CopyCreator.

        :param input_files: List containing exactly one Path to a database file to copy
        :param code_version: Version string for cache invalidation
        :param cache_enabled: Whether to enable caching
        """
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)
        self.input_files = input_files

        if not all(isinstance(f, Path) for f in self.input_files):
            raise ValueError("All input_files must be of type pathlib.Path")
        if len(self.input_files) != 1:
            raise ValueError(
                f"CopyCreator requires exactly one database file, got {len(self.input_files)}"
            )
        if not all(f.exists() for f in self.input_files):
            missing_files = [str(f) for f in self.input_files if not f.exists()]
            raise ValueError(f"The following input files do not exist: {missing_files}")

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters accepted by this generator.

        :return: Dictionary mapping parameter names to descriptions (empty for CopyCreator)
        """
        return {}

    def execute_impl(self, context: "PipelineContext", output_db: Path) -> None:
        """
        Override execute_impl to perform a simple file copy without opening a session.

        This is more efficient than the default Generator.execute_impl() which creates
        a new database and opens a session.

        :param context: PipelineContext (unused, but required by interface)
        :param output_db: Path to the output database file
        """
        self._unlink_stale_output_db(output_db)

        source_db = self.input_files[0]

        # Ensure parent directory exists
        output_db.parent.mkdir(parents=True, exist_ok=True)

        # Copy the database file
        shutil.copy2(source_db, output_db)
        self.logger.info(f"Copied database from {source_db} to {output_db}")

    def generate(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """
        Generate method (not used since execute_impl is overridden).

        This method is required by the Generator abstract base class but is never called
        because we override execute_impl().

        :param session: SQLAlchemy session (unused)
        :param params: Pipeline parameters (unused)
        """
        raise NotImplementedError(
            "CopyCreator.generate() should never be called. execute_impl() is overridden to bypass this."
        )
