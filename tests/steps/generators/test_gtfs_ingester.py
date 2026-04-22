"""Tests for GTFSIngester generator step."""

from pathlib import Path
from typing import List

import pytest
from eflips.model import Scenario, Station, Route, Trip, Rotation, Line
from sqlalchemy.orm import Session

from eflips.x.steps.generators import GTFSIngester


class TestGTFSIngester:
    """Test suite for GTFSIngester."""

    @pytest.fixture
    def gtfs_files(self, gtfs_test_data_dir: Path) -> List[Path]:
        """Get list of GTFS zip files from test data directory."""
        gtfs_files = sorted(gtfs_test_data_dir.glob("*.zip"))
        assert len(gtfs_files) > 0, f"No GTFS zip files found in {gtfs_test_data_dir}"
        return gtfs_files

    @pytest.fixture
    def sample_gtfs_file(self, gtfs_test_data_dir: Path) -> Path:
        """Get a single GTFS file for testing."""
        sample_file = gtfs_test_data_dir / "sample-feed-1.zip"
        if not sample_file.exists():
            pytest.skip(f"Sample GTFS file not found: {sample_file}")
        return sample_file

    @pytest.fixture
    def ingester(self, sample_gtfs_file: Path) -> GTFSIngester:
        """Create a GTFSIngester instance with test file."""
        return GTFSIngester(input_files=[sample_gtfs_file], cache_enabled=False)

    def test_init(self, sample_gtfs_file: Path):
        """Test that GTFSIngester initializes correctly."""
        ingester = GTFSIngester(input_files=[sample_gtfs_file])
        assert len(ingester.input_files) == 1
        assert isinstance(ingester.input_files[0], Path)
        assert ingester.input_files[0] == sample_gtfs_file

    def test_init_requires_single_file(self, gtfs_files: List[Path]):
        """Test that GTFSIngester requires exactly one GTFS file."""
        # Should fail with multiple files
        if len(gtfs_files) > 1:
            with pytest.raises(ValueError, match="exactly one GTFS zip file"):
                GTFSIngester(input_files=gtfs_files)

        # Should fail with no files
        with pytest.raises(ValueError, match="exactly one GTFS zip file"):
            GTFSIngester(input_files=[])

    def test_init_requires_existing_file(self):
        """Test that GTFSIngester validates file existence."""
        non_existent = Path("/tmp/does_not_exist.zip")
        with pytest.raises(ValueError, match="do not exist"):
            GTFSIngester(input_files=[non_existent])

    def test_document_params(self, ingester: GTFSIngester):
        """Test that document_params returns expected parameters."""
        params = ingester.document_params()
        assert "log_level" in params
        assert f"{ingester.__class__.__name__}.agency_name" in params
        assert f"{ingester.__class__.__name__}.start_date" in params
        assert f"{ingester.__class__.__name__}.duration" in params
        assert f"{ingester.__class__.__name__}.bus_only" in params

    def test_auto_select_start_date(self, sample_gtfs_file: Path):
        """Test the auto date selection method (global fallback path)."""
        start_date = GTFSIngester._auto_select_start_date(
            sample_gtfs_file, agency_name="", bus_only=False
        )

        # Should return a valid ISO 8601 date string
        assert isinstance(start_date, str)
        from datetime import datetime

        parsed_date = datetime.strptime(start_date, "%Y-%m-%d")

        # Should be a Monday (weekday() returns 0 for Monday)
        assert parsed_date.weekday() == 0, f"Selected date {start_date} is not a Monday"

    def test_generate_with_auto_date(self, db_session: Session, ingester: GTFSIngester):
        """Test generation with auto-selected date."""
        params = {
            "log_level": "WARNING",
            f"{ingester.__class__.__name__}.duration": "WEEK",
            f"{ingester.__class__.__name__}.bus_only": False,  # sample-feed-1 requires bus_only=False
        }

        # Run the generator - should auto-select date
        ingester.generate(db_session, params)

        # Check that a scenario was created
        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1
        assert "GTFS Import" in scenarios[0].name

    def test_generate_with_manual_date(self, db_session: Session, ingester: GTFSIngester):
        """Test generation with manually specified date."""
        params = {
            "log_level": "WARNING",
            f"{ingester.__class__.__name__}.start_date": "2007-01-01",  # sample-feed-1 validity
            f"{ingester.__class__.__name__}.duration": "DAY",
            f"{ingester.__class__.__name__}.bus_only": False,  # sample-feed-1 requires bus_only=False
        }

        # Run the generator
        ingester.generate(db_session, params)

        # Check that a scenario was created
        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1

    def test_generate_creates_valid_scenario(self, db_session: Session, ingester: GTFSIngester):
        """Test that generate() creates a scenario with all required objects."""
        params = {
            "log_level": "WARNING",
            f"{ingester.__class__.__name__}.start_date": "2007-01-01",
            f"{ingester.__class__.__name__}.duration": "WEEK",
            f"{ingester.__class__.__name__}.bus_only": False,  # sample-feed-1 requires bus_only=False
        }

        # Run the generator
        ingester.generate(db_session, params)

        # Check that a scenario was created
        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1
        scenario = scenarios[0]

        # Check that stations were created
        stations = db_session.query(Station).filter(Station.scenario_id == scenario.id).all()
        assert len(stations) > 0, "No stations were created"

        # Check station attributes
        for station in stations[:5]:
            assert station.name is not None
            assert station.scenario_id == scenario.id

        # Check that lines were created
        lines = db_session.query(Line).filter(Line.scenario_id == scenario.id).all()
        assert len(lines) > 0, "No lines were created"

        # Check that routes were created
        routes = db_session.query(Route).filter(Route.scenario_id == scenario.id).all()
        assert len(routes) > 0, "No routes were created"

        # Check route attributes
        for route in routes[:5]:
            assert route.name is not None
            assert route.scenario_id == scenario.id
            assert route.departure_station_id is not None
            assert route.arrival_station_id is not None
            assert route.line_id is not None

        # Check that trips were created
        trips = db_session.query(Trip).filter(Trip.scenario_id == scenario.id).all()
        assert len(trips) > 0, "No trips were created"

        # Check trip attributes
        for trip in trips[:5]:
            assert trip.route_id is not None
            assert trip.scenario_id == scenario.id
            assert trip.rotation_id is not None
            assert trip.departure_time is not None
            assert trip.arrival_time is not None

        # Check that rotations were created
        rotations = db_session.query(Rotation).filter(Rotation.scenario_id == scenario.id).all()
        assert len(rotations) > 0, "No rotations were created"

    def test_generate_with_invalid_log_level(self, db_session: Session, ingester: GTFSIngester):
        """Test that generate() raises error with invalid log level."""
        params = {
            "log_level": "INVALID",
            f"{ingester.__class__.__name__}.start_date": "2007-01-01",
        }

        with pytest.raises(ValueError, match="Invalid log level"):
            ingester.generate(db_session, params)

    @pytest.mark.parametrize("log_level", ["DEBUG", "ERROR"])
    def test_generate_with_different_log_levels(
        self, db_session: Session, ingester: GTFSIngester, log_level: str
    ):
        """Test that generate() works with different log levels."""
        params = {
            "log_level": log_level,
            f"{ingester.__class__.__name__}.start_date": "2007-01-01",
            f"{ingester.__class__.__name__}.duration": "DAY",
            f"{ingester.__class__.__name__}.bus_only": False,  # sample-feed-1 requires bus_only=False
        }

        # Should not raise an exception
        ingester.generate(db_session, params)

        # Basic sanity check
        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1

    @pytest.mark.parametrize("duration", ["DAY", "WEEK"])
    def test_generate_with_different_durations(
        self, db_session: Session, ingester: GTFSIngester, duration: str
    ):
        """Test that generate() works with different duration values."""
        params = {
            "log_level": "WARNING",
            f"{ingester.__class__.__name__}.start_date": "2007-01-01",
            f"{ingester.__class__.__name__}.duration": duration,
            f"{ingester.__class__.__name__}.bus_only": False,  # sample-feed-1 requires bus_only=False
        }

        # Should not raise an exception
        ingester.generate(db_session, params)

        # Basic sanity check
        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1

    def test_generate_with_invalid_date(self, db_session: Session, ingester: GTFSIngester):
        """Test that generate() handles invalid dates appropriately."""
        params = {
            "log_level": "WARNING",
            f"{ingester.__class__.__name__}.start_date": "2099-01-01",  # Far future date
            f"{ingester.__class__.__name__}.duration": "WEEK",
        }

        # Should raise ValueError due to date being outside validity period
        with pytest.raises(ValueError, match="GTFS preparation failed"):
            ingester.generate(db_session, params)

    def test_gtfs_files_exist(self, gtfs_files: List[Path]):
        """Test that the GTFS test files exist and are readable."""
        assert len(gtfs_files) > 0
        for gtfs_file in gtfs_files:
            assert gtfs_file.exists()
            assert gtfs_file.is_file()
            assert gtfs_file.suffix == ".zip"

    def test_generate_with_swu_file_bus_only(self, db_session: Session, gtfs_test_data_dir: Path):
        """Test generation with SWU.zip file using bus_only=True."""
        swu_file = gtfs_test_data_dir / "SWU.zip"
        if not swu_file.exists():
            pytest.skip(f"SWU GTFS file not found: {swu_file}")

        ingester = GTFSIngester(input_files=[swu_file], cache_enabled=False)
        params = {
            "log_level": "WARNING",
            f"{ingester.__class__.__name__}.duration": "DAY",
            f"{ingester.__class__.__name__}.bus_only": True,  # SWU supports bus_only=True
        }

        # Run the generator - should auto-select date
        ingester.generate(db_session, params)

        # Check that a scenario was created
        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1
        assert "GTFS Import" in scenarios[0].name

        # Check that at least some data was imported
        stations = db_session.query(Station).all()
        assert len(stations) > 0, "No stations were created"

        routes = db_session.query(Route).all()
        assert len(routes) > 0, "No routes were created"


class TestAutoSelectStartDateAgencyGate:
    """Cover the single-vs-multi-agency gate in ``_auto_select_start_date``.

    ``sample-feed-1.zip`` contains exactly one agency (``DTA`` / "Demo Transit
    Authority"), so passing a matching ``agency_name`` must NOT take the
    agency-aware path — the feed's ``len(feed.agency) > 1`` clause should
    short-circuit it to the global-validity fallback. Regression guard for
    the single-agency feed case that previously hit ``feed.routes["agency_id"]``
    in ``_compute_agency_active_dates`` even when the column was missing or
    redundant.
    """

    @pytest.fixture
    def sample_gtfs_file(self, gtfs_test_data_dir: Path) -> Path:
        f = gtfs_test_data_dir / "sample-feed-1.zip"
        if not f.exists():
            pytest.skip(f"Sample GTFS file not found: {f}")
        return f

    def test_single_agency_with_matching_name_uses_fallback(self, sample_gtfs_file: Path):
        """With one agency in the feed, a matching agency_name still picks a Monday
        from the global validity period rather than raising."""
        result = GTFSIngester._auto_select_start_date(
            sample_gtfs_file,
            agency_name="Demo Transit Authority",
            bus_only=False,
        )
        from datetime import datetime

        parsed = datetime.strptime(result, "%Y-%m-%d")
        assert parsed.weekday() == 0  # Monday

    def test_single_agency_with_matching_id_uses_fallback(self, sample_gtfs_file: Path):
        """Same as above but via agency_ids."""
        result = GTFSIngester._auto_select_start_date(
            sample_gtfs_file,
            agency_ids="DTA",
            bus_only=False,
        )
        from datetime import datetime

        parsed = datetime.strptime(result, "%Y-%m-%d")
        assert parsed.weekday() == 0

    def test_single_agency_unknown_name_is_ignored(self, sample_gtfs_file: Path):
        """Consequence of the ``len(feed.agency) > 1`` gate: on a single-agency feed
        an unknown ``agency_name`` does not trigger agency resolution at all — we
        silently fall back to global feed validity. Documenting this as intentional
        behavior (the single agency IS the whole feed, so the typo is harmless),
        but worth a regression guard."""
        from datetime import datetime

        result = GTFSIngester._auto_select_start_date(
            sample_gtfs_file,
            agency_name="Nonexistent Authority",
            bus_only=False,
        )
        parsed = datetime.strptime(result, "%Y-%m-%d")
        assert parsed.weekday() == 0


class TestValidateRoutesMatchAgencies:
    """Cover ``GTFSIngester._validate_routes_match_agencies``.

    ``sample-feed-1.zip`` has a single agency (DTA) owning five routes
    (AB, BFC, STBA, CITY, AAMV), and ``routes.txt`` carries the ``agency_id``
    column. This exercises the happy paths and the no-op early returns;
    the "routes owned by agency outside requested set" branch requires a
    multi-agency feed and is not covered here.
    """

    @pytest.fixture
    def sample_gtfs_file(self, gtfs_test_data_dir: Path) -> Path:
        f = gtfs_test_data_dir / "sample-feed-1.zip"
        if not f.exists():
            pytest.skip(f"Sample GTFS file not found: {f}")
        return f

    @pytest.fixture
    def ingester(self, sample_gtfs_file: Path) -> GTFSIngester:
        return GTFSIngester(input_files=[sample_gtfs_file], cache_enabled=False)

    def test_noop_when_route_ids_empty(self, ingester: GTFSIngester, sample_gtfs_file: Path):
        """Empty route_ids ⇒ nothing to validate, must not raise even with agencies set."""
        ingester._validate_routes_match_agencies(
            sample_gtfs_file, route_ids="", agency_ids="DTA", agency_name=""
        )
        ingester._validate_routes_match_agencies(
            sample_gtfs_file, route_ids=[], agency_ids="", agency_name="Demo Transit Authority"
        )

    def test_noop_when_no_agency_scoping(self, ingester: GTFSIngester, sample_gtfs_file: Path):
        """With route_ids but no agency/name, there is no agency intent to cross-check."""
        ingester._validate_routes_match_agencies(
            sample_gtfs_file, route_ids=["AB"], agency_ids="", agency_name=""
        )

    def test_happy_path_route_belongs_to_requested_agency(
        self, ingester: GTFSIngester, sample_gtfs_file: Path
    ):
        """Route AB is owned by DTA — requesting DTA must not raise."""
        ingester._validate_routes_match_agencies(
            sample_gtfs_file, route_ids=["AB", "CITY"], agency_ids="DTA", agency_name=""
        )
        ingester._validate_routes_match_agencies(
            sample_gtfs_file,
            route_ids="AB",
            agency_ids="",
            agency_name="Demo Transit Authority",
        )

    def test_nonexistent_route_id_is_not_this_layers_job(
        self, ingester: GTFSIngester, sample_gtfs_file: Path
    ):
        """After the trim, route-membership is validated on the eflips-ingest side.
        The x-side cross-check must no longer raise on unknown route_ids."""
        ingester._validate_routes_match_agencies(
            sample_gtfs_file,
            route_ids=["DEFINITELY_NOT_A_ROUTE"],
            agency_ids="DTA",
            agency_name="",
        )

    def test_unknown_agency_raises(self, ingester: GTFSIngester, sample_gtfs_file: Path):
        """Unknown agency bubbles up from ``_resolve_agency_ids`` with a clear message."""
        with pytest.raises(ValueError, match="Agency not found"):
            ingester._validate_routes_match_agencies(
                sample_gtfs_file,
                route_ids=["AB"],
                agency_ids="NO_SUCH_ID",
                agency_name="",
            )
        with pytest.raises(ValueError, match="Agency not found"):
            ingester._validate_routes_match_agencies(
                sample_gtfs_file,
                route_ids=["AB"],
                agency_ids="",
                agency_name="Imaginary Transit",
            )

    def test_end_to_end_route_ids_restrict_ingest(
        self, db_session: Session, sample_gtfs_file: Path
    ):
        """End-to-end: run ``GTFSIngester.generate`` with ``route_ids`` set to two
        of sample-feed-1's five routes and verify the final eflips-model scenario
        only contains data for those two routes.

        sample-feed-1 route_ids: AB, BFC, STBA, CITY, AAMV (all owned by DTA).
        Restrict to {"AB", "CITY"} and confirm exactly those survived into the DB.
        """
        ingester = GTFSIngester(input_files=[sample_gtfs_file], cache_enabled=False)
        params = {
            "log_level": "WARNING",
            f"{ingester.__class__.__name__}.start_date": "2007-01-01",
            f"{ingester.__class__.__name__}.duration": "WEEK",
            f"{ingester.__class__.__name__}.bus_only": False,
            f"{ingester.__class__.__name__}.route_ids": ["AB", "CITY"],
        }

        ingester.generate(db_session, params)

        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1
        scenario_id = scenarios[0].id

        # sample-feed-1 has 5 routes total; restricting to {"AB", "CITY"} must
        # cut that to exactly 2 Lines in the scenario.
        lines = db_session.query(Line).filter(Line.scenario_id == scenario_id).all()
        assert len(lines) == 2, (
            f"Expected 2 Lines (for route_ids AB and CITY), got {len(lines)}: "
            f"{[l.name for l in lines]}"
        )

        # AB has route_short_name 10, CITY has 40 — confirm those survived
        # (rather than e.g. BFC=20 / STBA=30 / AAMV=50).
        line_short_codes = {line.name.split(":", 1)[0].strip() for line in lines}
        assert line_short_codes == {"10", "40"}, (
            f"Lines match wrong route_ids: {[l.name for l in lines]}"
        )

        # Trips must reference only the surviving lines.
        trips = db_session.query(Trip).filter(Trip.scenario_id == scenario_id).all()
        assert len(trips) > 0, "route_id filter yielded no trips"
        trip_line_ids = {t.route.line_id for t in trips}
        assert trip_line_ids == {line.id for line in lines}, (
            "Trips reference lines outside the restricted set"
        )
