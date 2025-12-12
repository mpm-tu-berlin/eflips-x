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
        """Test the auto date selection method."""
        start_date = GTFSIngester._auto_select_start_date(sample_gtfs_file)

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

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
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
            f"{ingester.__class__.__name__}.duration": "WEEK",
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
