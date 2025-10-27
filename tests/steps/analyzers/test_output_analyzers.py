"""Tests for output analyzers."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import matplotlib.animation as animation
import pandas as pd
import plotly.graph_objs as go
import pytest
from eflips.model import Area, ChargeType, Depot, Event, Rotation, Scenario, Vehicle, VehicleType
from matplotlib.figure import Figure
from sqlalchemy.orm import Session
from tests.util import multi_depot_scenario

from eflips.x.steps.analyzers.output_analyzers import (
    DepartureArrivalSocAnalyzer,
    DepotActivityAnalyzer,
    DepotEventAnalyzer,
    DepotLayoutAnalyzer,
    PowerAndOccupancyAnalyzer,
    SpecificEnergyConsumptionAnalyzer,
    VehicleSocAnalyzer,
)
from eflips.x.steps.modifiers.scheduling import DepotAssignment, VehicleScheduling
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation


@pytest.fixture
def simulated_scenario(db_session: Session, tmp_path: Path) -> Scenario:
    """
    Create a scenario with simulation results.

    Runs vehicle scheduling, depot assignment, depot generation, and simulation.
    """
    scenario = multi_depot_scenario(
        db_session,
        num_depots=2,
        lines_per_depot=4,
        trips_per_line=30,  # High number to force rotation splitting
    )

    # Step 1: Vehicle Scheduling in DEPOT mode
    vehicle_scheduler = VehicleScheduling()
    vs_params = {
        "VehicleScheduling.charge_type": ChargeType.DEPOT,
        "VehicleScheduling.battery_margin": 0.1,
    }
    vehicle_scheduler.modify(session=db_session, params=vs_params)

    # Step 2: Depot Assignment
    all_depots = db_session.query(Depot).filter_by(scenario_id=scenario.id).all()
    all_vehicle_types = db_session.query(VehicleType).filter_by(scenario_id=scenario.id).all()

    depot_config = []
    for depot in all_depots:
        depot_config.append(
            {
                "depot_station": depot.station_id,
                "capacity": 100,
                "vehicle_type": [vt.id for vt in all_vehicle_types],
                "name": depot.name,
            }
        )

    depot_assigner = DepotAssignment()
    da_params = {
        "DepotAssignment.depot_config": depot_config,
        "DepotAssignment.depot_usage": 0.9,
        "DepotAssignment.step_size": 0.2,
        "DepotAssignment.max_iterations": 1,
    }
    depot_assigner.modify(session=db_session, params=da_params)

    # Step 3: Depot Generation
    depot_generator = DepotGenerator()
    depot_generator.modify(session=db_session, params={})

    # Step 4: Simulation
    simulator = Simulation()
    simulator.modify(session=db_session, params={})

    db_session.commit()

    return scenario


class TestDepartureArrivalSocAnalyzer:
    """Test suite for DepartureArrivalSocAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a DepartureArrivalSocAnalyzer instance."""
        return DepartureArrivalSocAnalyzer()

    def test_analyze(
        self, analyzer: DepartureArrivalSocAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test analyzing departure/arrival SoC."""
        result = analyzer.analyze(temp_db, {})

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify DataFrame has expected columns
        expected_columns = [
            "rotation_id",
            "rotation_name",
            "vehicle_type_id",
            "vehicle_type_name",
            "vehicle_id",
            "vehicle_name",
            "time",
            "soc",
            "event_type",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Verify we have data (should have departure and arrival events)
        assert len(result) > 0, "Should have at least one departure/arrival event"

        # Verify event types are correct
        assert set(result["event_type"]).issubset({"Departure", "Arrival"})

    def test_visualize(
        self, analyzer: DepartureArrivalSocAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test visualization method."""
        result = analyzer.analyze(temp_db, {})

        # Test visualization
        fig = DepartureArrivalSocAnalyzer.visualize(result)

        # Verify it returns a plotly figure
        assert isinstance(fig, go.Figure)

    def test_document_params(self, analyzer: DepartureArrivalSocAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)


class TestDepotEventAnalyzer:
    """Test suite for DepotEventAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a DepotEventAnalyzer instance."""
        return DepotEventAnalyzer()

    def test_analyze_all_vehicles(
        self, analyzer: DepotEventAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test analyzing all vehicles without filtering."""
        result = analyzer.analyze(temp_db, {})

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify DataFrame has expected columns
        expected_columns = [
            "time_start",
            "time_end",
            "soc_start",
            "soc_end",
            "vehicle_id",
            "vehicle_type_id",
            "vehicle_type_name",
            "event_type",
            "area_id",
            "trip_id",
            "station_id",
            "location",
            "area_type",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Verify we have data
        assert len(result) > 0, "Should have at least one event"

    def test_analyze_filtered_vehicles(
        self, analyzer: DepotEventAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test analyzing with vehicle_ids filter."""
        # Get first vehicle ID
        first_vehicle = db_session.query(Vehicle).first()
        vehicle_id = first_vehicle.id

        # Count events for this vehicle
        event_count = db_session.query(Event).filter_by(vehicle_id=vehicle_id).count()

        # Analyze with filter
        result = analyzer.analyze(temp_db, {"DepotEventAnalyzer.vehicle_ids": vehicle_id})

        # Verify we only get events from one vehicle
        assert len(result) == event_count
        assert all(result["vehicle_id"] == str(vehicle_id))

    def test_visualize(
        self, analyzer: DepotEventAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test visualization method."""
        result = analyzer.analyze(temp_db, {})

        # Test visualization with different color schemes
        for color_scheme in ["event_type", "location", "area_type"]:
            fig = DepotEventAnalyzer.visualize(result, color_scheme=color_scheme)
            assert isinstance(fig, go.Figure)

    def test_document_params(self, analyzer: DepotEventAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)
        assert "DepotEventAnalyzer.vehicle_ids" in docs


class TestPowerAndOccupancyAnalyzer:
    """Test suite for PowerAndOccupancyAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a PowerAndOccupancyAnalyzer instance."""
        return PowerAndOccupancyAnalyzer()

    def test_analyze_single_area(
        self, analyzer: PowerAndOccupancyAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test analyzing a single area."""
        # Get first area ID
        first_area = db_session.query(Area).first()
        area_id = first_area.id

        # Analyze
        result = analyzer.analyze(temp_db, {"PowerAndOccupancyAnalyzer.area_id": area_id})

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify DataFrame has expected columns
        expected_columns = ["time", "power", "occupancy_charging", "occupancy_total"]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Verify we have data
        assert len(result) > 0, "Should have timeseries data"

    def test_analyze_multiple_areas(
        self, analyzer: PowerAndOccupancyAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test analyzing multiple areas."""
        # Get first two area IDs
        areas = db_session.query(Area).limit(2).all()
        area_ids = [a.id for a in areas]

        # Analyze
        result = analyzer.analyze(temp_db, {"PowerAndOccupancyAnalyzer.area_id": area_ids})

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_analyze_missing_area_id(
        self, analyzer: PowerAndOccupancyAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test that missing area_id raises ValueError."""
        with pytest.raises(ValueError, match="Required parameter.*area_id.*not provided"):
            analyzer.analyze(temp_db, {})

    def test_visualize(
        self, analyzer: PowerAndOccupancyAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test visualization method."""
        first_area = db_session.query(Area).first()
        area_id = first_area.id

        result = analyzer.analyze(temp_db, {"PowerAndOccupancyAnalyzer.area_id": area_id})

        # Test visualization
        fig = PowerAndOccupancyAnalyzer.visualize(result)

        # Verify it returns a plotly figure
        assert isinstance(fig, go.Figure)

    def test_document_params(self, analyzer: PowerAndOccupancyAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)
        assert "PowerAndOccupancyAnalyzer.area_id" in docs
        assert "PowerAndOccupancyAnalyzer.temporal_resolution" in docs


class TestSpecificEnergyConsumptionAnalyzer:
    """Test suite for SpecificEnergyConsumptionAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a SpecificEnergyConsumptionAnalyzer instance."""
        return SpecificEnergyConsumptionAnalyzer()

    def test_analyze(
        self, analyzer: SpecificEnergyConsumptionAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test analyzing specific energy consumption."""
        result = analyzer.analyze(temp_db, {})

        # Verify result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Verify DataFrame has expected columns
        expected_columns = [
            "trip_id",
            "route_id",
            "route_name",
            "distance",
            "energy_consumption",
            "vehicle_type_id",
            "vehicle_type_name",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

        # Verify we have data
        assert len(result) > 0, "Should have trip energy data"

        # Verify energy consumption is positive
        assert all(result["energy_consumption"] >= 0)

    def test_visualize(
        self, analyzer: SpecificEnergyConsumptionAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test visualization method."""
        result = analyzer.analyze(temp_db, {})

        # Test visualization
        fig = SpecificEnergyConsumptionAnalyzer.visualize(result)

        # Verify it returns a plotly figure
        assert isinstance(fig, go.Figure)

    def test_document_params(self, analyzer: SpecificEnergyConsumptionAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)


class TestVehicleSocAnalyzer:
    """Test suite for VehicleSocAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a VehicleSocAnalyzer instance."""
        return VehicleSocAnalyzer()

    def test_analyze(
        self, analyzer: VehicleSocAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test analyzing vehicle SoC."""
        # Get first vehicle ID
        first_vehicle = db_session.query(Vehicle).first()
        vehicle_id = first_vehicle.id

        # Analyze
        result = analyzer.analyze(temp_db, {"VehicleSocAnalyzer.vehicle_id": vehicle_id})

        # Verify result is a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

        # Verify first element is a DataFrame
        df, descriptions = result
        assert isinstance(df, pd.DataFrame)

        # Verify DataFrame has expected columns
        expected_columns = ["time", "soc"]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

        # Verify descriptions is a dict
        assert isinstance(descriptions, dict)
        expected_keys = ["rotation", "charging", "trip"]
        for key in expected_keys:
            assert key in descriptions, f"Missing description key: {key}"

        # Verify we have data
        assert len(df) > 0, "Should have SoC timeseries data"

    def test_analyze_missing_vehicle_id(
        self, analyzer: VehicleSocAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test that missing vehicle_id raises ValueError."""
        with pytest.raises(ValueError, match="Required parameter.*vehicle_id.*not provided"):
            analyzer.analyze(temp_db, {})

    def test_visualize(
        self, analyzer: VehicleSocAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test visualization method."""
        first_vehicle = db_session.query(Vehicle).first()
        vehicle_id = first_vehicle.id

        df, descriptions = analyzer.analyze(temp_db, {"VehicleSocAnalyzer.vehicle_id": vehicle_id})

        # Test visualization
        fig = VehicleSocAnalyzer.visualize(df, descriptions)

        # Verify it returns a plotly figure
        assert isinstance(fig, go.Figure)

    def test_document_params(self, analyzer: VehicleSocAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)
        assert "VehicleSocAnalyzer.vehicle_id" in docs
        assert "VehicleSocAnalyzer.timezone" in docs


class TestDepotLayoutAnalyzer:
    """Test suite for DepotLayoutAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a DepotLayoutAnalyzer instance."""
        return DepotLayoutAnalyzer()

    def test_analyze(
        self, analyzer: DepotLayoutAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test analyzing depot layout."""
        # Get first depot ID
        first_depot = db_session.query(Depot).first()
        depot_id = first_depot.id

        # Analyze
        result = analyzer.analyze(temp_db, {"DepotLayoutAnalyzer.depot_id": depot_id})

        # Verify result is a list of lists
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(block, list) for block in result)

        # Verify each block contains Area objects
        for block in result:
            assert all(isinstance(area, Area) for area in block)

    def test_analyze_missing_depot_id(
        self, analyzer: DepotLayoutAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test that missing depot_id raises ValueError."""
        with pytest.raises(ValueError, match="Required parameter.*depot_id.*not provided"):
            analyzer.analyze(temp_db, {})

    @pytest.mark.xfail(
        reason="eflips-eval depot_layout visualization requires Area objects to be session-bound"
    )
    def test_visualize(
        self, analyzer: DepotLayoutAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test visualization method."""
        first_depot = db_session.query(Depot).first()
        depot_id = first_depot.id

        area_blocks = analyzer.analyze(temp_db, {"DepotLayoutAnalyzer.depot_id": depot_id})

        # Test visualization
        area_dict, fig = DepotLayoutAnalyzer.visualize(area_blocks)

        # Verify it returns a dict and Figure
        assert isinstance(area_dict, dict)
        assert isinstance(fig, Figure)

    def test_document_params(self, analyzer: DepotLayoutAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)
        assert "DepotLayoutAnalyzer.depot_id" in docs


class TestDepotActivityAnalyzer:
    """Test suite for DepotActivityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a DepotActivityAnalyzer instance."""
        return DepotActivityAnalyzer()

    def test_analyze(
        self, analyzer: DepotActivityAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test analyzing depot activity."""
        # Get first depot ID and first rotation times
        first_depot = db_session.query(Depot).first()
        depot_id = first_depot.id

        first_rotation = db_session.query(Rotation).first()
        animation_start = first_rotation.trips[0].departure_time
        animation_end = first_rotation.trips[-1].arrival_time

        # Analyze
        result = analyzer.analyze(
            temp_db,
            {
                "DepotActivityAnalyzer.depot_id": depot_id,
                "DepotActivityAnalyzer.animation_start": animation_start,
                "DepotActivityAnalyzer.animation_end": animation_end,
            },
        )

        # Verify result is a dict
        assert isinstance(result, dict)

        # Verify keys are tuples of (area_id, slot_id)
        for key in result.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], int)  # area_id
            assert isinstance(key[1], int)  # slot_id

        # Verify values are lists of tuples
        for value in result.values():
            assert isinstance(value, list)
            for item in value:
                assert isinstance(item, tuple)
                assert len(item) == 2

    def test_analyze_missing_required_params(
        self, analyzer: DepotActivityAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test that missing required parameters raise ValueError."""
        # Missing depot_id
        with pytest.raises(ValueError, match="Required parameter.*depot_id.*not provided"):
            analyzer.analyze(temp_db, {})

        # Missing animation_start
        with pytest.raises(ValueError, match="Required parameter.*animation_start.*not provided"):
            analyzer.analyze(temp_db, {"DepotActivityAnalyzer.depot_id": 1})

        # Missing animation_end
        with pytest.raises(ValueError, match="Required parameter.*animation_end.*not provided"):
            analyzer.analyze(
                temp_db,
                {
                    "DepotActivityAnalyzer.depot_id": 1,
                    "DepotActivityAnalyzer.animation_start": datetime.now(),
                },
            )

    @pytest.mark.xfail(
        reason="eflips-eval depot_activity_animation visualization requires Area objects to be session-bound"
    )
    def test_visualize(
        self, analyzer: DepotActivityAnalyzer, temp_db: Path, simulated_scenario, db_session: Session
    ):
        """Test visualization method."""
        first_depot = db_session.query(Depot).first()
        depot_id = first_depot.id

        first_rotation = db_session.query(Rotation).first()
        animation_start = first_rotation.trips[0].departure_time
        animation_end = first_rotation.trips[-1].arrival_time

        # Get depot layout
        layout_analyzer = DepotLayoutAnalyzer()
        area_blocks = layout_analyzer.analyze(temp_db, {"DepotLayoutAnalyzer.depot_id": depot_id})

        # Get depot activity
        area_occupancy = analyzer.analyze(
            temp_db,
            {
                "DepotActivityAnalyzer.depot_id": depot_id,
                "DepotActivityAnalyzer.animation_start": animation_start,
                "DepotActivityAnalyzer.animation_end": animation_end,
            },
        )

        # Test visualization
        anim = DepotActivityAnalyzer.visualize(
            area_blocks, area_occupancy, (animation_start, animation_end)
        )

        # Verify it returns an animation
        assert isinstance(anim, animation.FuncAnimation)

    def test_document_params(self, analyzer: DepotActivityAnalyzer):
        """Test parameter documentation."""
        docs = analyzer.document_params()

        assert isinstance(docs, dict)
        assert "DepotActivityAnalyzer.depot_id" in docs
        assert "DepotActivityAnalyzer.animation_start" in docs
        assert "DepotActivityAnalyzer.animation_end" in docs
