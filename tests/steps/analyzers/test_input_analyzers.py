"""Tests for input analyzers."""

from pathlib import Path
from zoneinfo import ZoneInfo

import dash_cytoscape as cyto
import folium
import pandas as pd
import plotly.graph_objs as go
import pytest
from eflips.model import Scenario
from sqlalchemy.orm import Session
from tests.util import multi_depot_scenario

from eflips.x.steps.analyzers.input_analyzers import (
    GeographicTripPlotAnalyzer,
    RotationInfoAnalyzer,
    SingleRotationInfoAnalyzer,
)


@pytest.fixture
def test_scenario(db_session: Session) -> Scenario:
    """Create a test scenario with multi-depot network."""
    return multi_depot_scenario(
        db_session,
        num_depots=2,
        lines_per_depot=4,
        trips_per_line=10,
    )


class TestRotationInfoAnalyzer:
    """Test suite for RotationInfoAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a RotationInfoAnalyzer instance."""
        return RotationInfoAnalyzer()

    def test_analyze_all_rotations(
        self, analyzer: RotationInfoAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test analyzing all rotations without filtering."""
        # Save database and create path
        db_session.commit()

        result = analyzer.analyze(temp_db, {})

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify DataFrame has expected columns
        expected_columns = [
            "rotation_id",
            "rotation_name",
            "vehicle_type_id",
            "vehicle_type_name",
            "total_distance",
            "line_name",
            "line_is_unified",
            "time_start",
            "time_end",
            "start_station",
            "end_station",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Verify we have data
        assert len(result) > 0, "Should have at least one rotation"

    def test_analyze_filtered_rotations(
        self, analyzer: RotationInfoAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test analyzing with rotation_ids filter."""
        db_session.commit()

        # Get first rotation ID
        from eflips.model import Rotation
        first_rotation = db_session.query(Rotation).first()
        rotation_id = first_rotation.id

        # Analyze with filter
        result = analyzer.analyze(
            temp_db, {"RotationInfoAnalyzer.rotation_ids": rotation_id}
        )

        # Verify we only get one rotation
        assert len(result) == 1
        assert result.iloc[0]["rotation_id"] == rotation_id

    def test_analyze_multiple_filtered_rotations(
        self, analyzer: RotationInfoAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test analyzing with list of rotation_ids."""
        db_session.commit()

        # Get first two rotation IDs
        from eflips.model import Rotation
        rotations = db_session.query(Rotation).limit(2).all()
        rotation_ids = [r.id for r in rotations]

        # Analyze with filter
        result = analyzer.analyze(
            temp_db, {"RotationInfoAnalyzer.rotation_ids": rotation_ids}
        )

        # Verify we only get two rotations
        assert len(result) == 2
        assert set(result["rotation_id"]) == set(rotation_ids)

    def test_visualize(
        self, analyzer: RotationInfoAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test visualization method."""
        db_session.commit()

        result = analyzer.analyze(temp_db, {})

        # Test visualization
        fig = RotationInfoAnalyzer.visualize(result)

        # Verify it returns a plotly figure
        assert isinstance(fig, go.Figure)

    def test_visualize_with_timezone(
        self, analyzer: RotationInfoAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test visualization with custom timezone."""
        db_session.commit()

        result = analyzer.analyze(temp_db, {})

        # Test visualization with UTC timezone
        fig = RotationInfoAnalyzer.visualize(result, timezone=ZoneInfo("UTC"))

        assert isinstance(fig, go.Figure)

    def test_document_params(self, analyzer: RotationInfoAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)
        assert "RotationInfoAnalyzer.rotation_ids" in docs


class TestGeographicTripPlotAnalyzer:
    """Test suite for GeographicTripPlotAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a GeographicTripPlotAnalyzer instance."""
        return GeographicTripPlotAnalyzer()

    @pytest.mark.xfail(
        reason="eflips-eval geographic_trip_plot doesn't support route geometries yet"
    )
    def test_analyze_all_trips(
        self, analyzer: GeographicTripPlotAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test analyzing all trips without filtering."""
        db_session.commit()

        result = analyzer.analyze(temp_db, {})

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify DataFrame has expected columns
        expected_columns = [
            "rotation_id",
            "rotation_name",
            "vehicle_type_id",
            "vehicle_type_name",
            "originating_depot_id",
            "originating_depot_name",
            "distance",
            "coordinates",
            "line_name",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Verify we have data
        assert len(result) > 0, "Should have at least one trip"

    @pytest.mark.xfail(
        reason="eflips-eval geographic_trip_plot doesn't support route geometries yet"
    )
    def test_analyze_filtered_trips(
        self, analyzer: GeographicTripPlotAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test analyzing with rotation_ids filter."""
        db_session.commit()

        # Get first rotation ID
        from eflips.model import Rotation
        first_rotation = db_session.query(Rotation).first()
        rotation_id = first_rotation.id

        # Count trips in this rotation
        expected_trip_count = len(first_rotation.trips)

        # Analyze with filter
        result = analyzer.analyze(
            temp_db, {"GeographicTripPlotAnalyzer.rotation_ids": rotation_id}
        )

        # Verify we only get trips from one rotation
        assert len(result) == expected_trip_count
        assert all(result["rotation_id"] == rotation_id)

    @pytest.mark.xfail(
        reason="eflips-eval geographic_trip_plot doesn't support route geometries yet"
    )
    def test_visualize(
        self, analyzer: GeographicTripPlotAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test visualization method."""
        db_session.commit()

        result = analyzer.analyze(temp_db, {})

        # Test visualization
        map_obj = GeographicTripPlotAnalyzer.visualize(result)

        # Verify it returns a folium map
        assert isinstance(map_obj, folium.Map)

    def test_document_params(self, analyzer: GeographicTripPlotAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)
        assert "GeographicTripPlotAnalyzer.rotation_ids" in docs


class TestSingleRotationInfoAnalyzer:
    """Test suite for SingleRotationInfoAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a SingleRotationInfoAnalyzer instance."""
        return SingleRotationInfoAnalyzer()

    def test_analyze_single_rotation(
        self, analyzer: SingleRotationInfoAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test analyzing a single rotation."""
        db_session.commit()

        # Get first rotation ID
        from eflips.model import Rotation
        first_rotation = db_session.query(Rotation).first()
        rotation_id = first_rotation.id
        trip_count = len(first_rotation.trips)

        # Analyze
        result = analyzer.analyze(
            temp_db, {"SingleRotationInfoAnalyzer.rotation_id": rotation_id}
        )

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify DataFrame has expected columns
        expected_columns = [
            "trip_id",
            "trip_type",
            "line_name",
            "route_name",
            "distance",
            "departure_time",
            "arrival_time",
            "departure_station_name",
            "departure_station_id",
            "arrival_station_name",
            "arrival_station_id",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Verify we have the right number of trips
        assert len(result) == trip_count

    def test_analyze_missing_rotation_id(
        self, analyzer: SingleRotationInfoAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test that missing rotation_id raises ValueError."""
        db_session.commit()

        with pytest.raises(ValueError, match="Required parameter.*rotation_id.*not provided"):
            analyzer.analyze(temp_db, {})

    def test_visualize(
        self, analyzer: SingleRotationInfoAnalyzer, temp_db: Path, test_scenario: Scenario, db_session: Session
    ):
        """Test visualization method."""
        db_session.commit()

        # Get first rotation ID
        from eflips.model import Rotation
        first_rotation = db_session.query(Rotation).first()
        rotation_id = first_rotation.id

        result = analyzer.analyze(
            temp_db, {"SingleRotationInfoAnalyzer.rotation_id": rotation_id}
        )

        # Test visualization
        cyto_obj = SingleRotationInfoAnalyzer.visualize(result)

        # Verify it returns a Cytoscape object
        assert isinstance(cyto_obj, cyto.Cytoscape)

    def test_document_params(self, analyzer: SingleRotationInfoAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)
        assert "SingleRotationInfoAnalyzer.rotation_id" in docs
