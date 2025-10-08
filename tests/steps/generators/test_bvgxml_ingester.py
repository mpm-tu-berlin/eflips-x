"""Tests for BVGXMLIngester generator step."""

from pathlib import Path
from typing import List

import pytest
from eflips.model import Scenario, Station, Route, Trip, Rotation
from geoalchemy2.shape import to_shape
from sqlalchemy.orm import Session

from eflips.x.steps.generators import BVGXMLIngester


class TestBVGXMLIngester:
    """Test suite for BVGXMLIngester."""

    @pytest.fixture
    def xml_files(self, test_data_dir: Path) -> List[Path]:
        """Get list of XML files from test data directory."""
        xml_files = sorted(test_data_dir.glob("*.xml"))
        assert len(xml_files) > 0, f"No XML files found in {test_data_dir}"
        return xml_files

    @pytest.fixture
    def ingester(self, xml_files: List[Path]) -> BVGXMLIngester:
        """Create a BVGXMLIngester instance with test files."""
        return BVGXMLIngester(input_files=xml_files, cache_enabled=False)

    def test_init(self, xml_files: List[Path]):
        """Test that BVGXMLIngester initializes correctly."""
        ingester = BVGXMLIngester(
            input_files=xml_files,
        )
        assert len(ingester.input_files) == len(xml_files)
        assert all(isinstance(f, Path) for f in ingester.input_files)

    def test_document_params(self, ingester: BVGXMLIngester):
        """Test that document_params returns expected parameters."""
        params = ingester.document_params()
        assert "log_level" in params
        assert f"{ingester.__class__.__name__}.multithreading" in params

    def test_generate(self, db_session: Session, ingester: BVGXMLIngester):
        """Test that generate() creates a scenario in the database."""
        params = {"log_level": "INFO", f"{ingester.__class__.__name__}.multithreading": False}

        # Run the generator
        result = ingester.generate(db_session, params)

        # Check that a scenario was created
        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1
        assert "BVG-XML Ingestion" in scenarios[0].name

        # Check that stations were created
        stations = db_session.query(Station).all()
        assert len(stations) > 0
        # Check that stations have required attributes
        for station in stations[:5]:  # Check first 5
            assert station.name is not None
            assert station.scenario_id is not None
            # Make sure geometry is valid
            shapely_geom = to_shape(station.geom)
            assert shapely_geom.is_valid

        # Check that routes were created
        routes = db_session.query(Route).all()
        assert len(routes) > 0
        # Check that routes have required attributes
        for route in routes[:5]:  # Check first 5
            assert route.name is not None
            assert route.scenario_id is not None
            assert route.departure_station_id is not None
            assert route.arrival_station_id is not None

        # Check that trips were created
        trips = db_session.query(Trip).all()
        assert len(trips) > 0
        # Check that trips have required attributes
        for trip in trips[:5]:  # Check first 5
            assert trip.route_id is not None
            assert trip.scenario_id is not None

        # Check that rotations were created
        rotations = db_session.query(Rotation).all()
        assert len(rotations) > 0

    def test_generate_with_multithreading(self, db_session: Session, ingester: BVGXMLIngester):
        """Test that generate() works with multithreading enabled."""
        params = {"log_level": "WARNING", f"{ingester.__class__.__name__}.multithreading": True}

        # Should not raise an exception
        result = ingester.generate(db_session, params)

        # Basic sanity check
        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1

    def test_generate_with_invalid_log_level(self, db_session: Session, ingester: BVGXMLIngester):
        """Test that generate() raises error with invalid log level."""
        params = {"log_level": "INVALID", f"{ingester.__class__.__name__}.multithreading": False}

        with pytest.raises(ValueError, match="Invalid log level"):
            ingester.generate(db_session, params)

    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_generate_with_different_log_levels(
        self, db_session: Session, ingester: BVGXMLIngester, log_level: str
    ):
        """Test that generate() works with different log levels."""
        params = {"log_level": log_level, f"{ingester.__class__.__name__}.multithreading": False}

        # Shorten the input files for faster testing
        ingester.input_files = ingester.input_files[:2]

        # Should not raise an exception
        ingester.generate(db_session, params)

        # Basic sanity check
        scenarios = db_session.query(Scenario).all()
        assert len(scenarios) == 1

    def test_xml_files_exist(self, xml_files: List[Path]):
        """Test that the XML test files exist and are readable."""
        assert len(xml_files) > 0
        for xml_file in xml_files:
            assert xml_file.exists()
            assert xml_file.is_file()
            assert xml_file.suffix == ".xml"
