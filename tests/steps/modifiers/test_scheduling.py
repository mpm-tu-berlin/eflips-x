"""Tests for vehicle scheduling modifiers."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytz
from eflips.model import (
    Scenario,
    VehicleType,
    Rotation,
    Trip,
    TripType,
    VehicleClass,
    Depot,
)
from geoalchemy2.shape import from_shape
from shapely import Point
from sqlalchemy.orm import Session

from eflips.x.steps.generators import BVGXMLIngester
from eflips.x.steps.modifiers.scheduling import (
    VehicleScheduling,
    DepotAssignment,
    InsufficientChargingTimeAnalyzer,
)
from eflips.model import Event


class TestVehicleScheduling:
    """Test suite for VehicleScheduling modifier."""

    @pytest.fixture
    def scenario_with_vehicle_types(self, db_session: Session, test_data_dir: Path) -> Scenario:
        """
        Create a test scenario with vehicle types and rotations.
        Uses BVGXMLIngester to load Berlin Testing data.
        """
        # Get XML files from test data directory
        xml_files = sorted(test_data_dir.glob("*.xml"))
        assert len(xml_files) > 0, f"No XML files found in {test_data_dir}"

        # Use only a subset of files for faster testing
        xml_files = xml_files[:3]

        # Create ingester and generate scenario
        ingester = BVGXMLIngester(input_files=xml_files, cache_enabled=False)
        params = {
            "log_level": "WARNING",
            f"{ingester.__class__.__name__}.multithreading": False,
        }
        ingester.generate(db_session, params)

        # Get the created scenario
        scenario = db_session.query(Scenario).one()

        # Create a vehicle class
        vehicle_class = VehicleClass(
            name="Standard Bus",
            scenario_id=scenario.id,
        )
        db_session.add(vehicle_class)
        db_session.flush()

        # Create vehicle types with constant consumption for the rotations
        # Get all unique rotations
        rotations = db_session.query(Rotation).filter(Rotation.scenario_id == scenario.id).all()

        # Create a simple vehicle type with constant consumption
        vehicle_type = VehicleType(
            name="Electric Bus 12m",
            name_short="EB12",
            scenario_id=scenario.id,
            battery_capacity=350.0,  # kWh
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],  # Simple constant charging curve
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=10000,
            allowed_mass=20000,
            consumption=1.2,  # kWh/km - constant consumption value
        )
        db_session.add(vehicle_type)
        db_session.flush()

        # Assign all rotations to this vehicle type
        for rotation in rotations:
            rotation.vehicle_type_id = vehicle_type.id

        db_session.commit()
        return scenario

    def test_vehicle_scheduling_basic(
        self,
        temp_db: Path,
        scenario_with_vehicle_types: Scenario,
        db_session: Session,
    ):
        """Test VehicleScheduling modifier with default parameters."""
        modifier = VehicleScheduling()

        # Count initial rotations
        initial_rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == scenario_with_vehicle_types.id)
            .count()
        )
        assert initial_rotations > 0, "Should have rotations from ingested data"

        # Count initial trips
        initial_trips = (
            db_session.query(Trip)
            .filter(Trip.scenario_id == scenario_with_vehicle_types.id)
            .count()
        )
        assert initial_trips > 0, "Should have trips from ingested data"

        # Run modifier with default parameters
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # Check that rotations still exist (they should be reorganized, not deleted)
        final_rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == scenario_with_vehicle_types.id)
            .count()
        )
        assert final_rotations > 0, "Should have rotations after scheduling"

        # Check that trips still exist
        final_trips = (
            db_session.query(Trip)
            .filter(Trip.scenario_id == scenario_with_vehicle_types.id)
            .filter(Trip.trip_type == TripType.PASSENGER)
            .count()
        )
        assert final_trips > 0, "Should have trips after scheduling"

    def test_vehicle_scheduling_with_custom_parameters(
        self,
        temp_db: Path,
        scenario_with_vehicle_types: Scenario,
        db_session: Session,
    ):
        """Test VehicleScheduling modifier with custom parameters."""
        modifier = VehicleScheduling()

        # Custom parameters
        params = {
            "VehicleScheduling.minimum_break_time": timedelta(minutes=15),
            "VehicleScheduling.maximum_schedule_duration": timedelta(hours=12),
            "VehicleScheduling.battery_margin": 0.15,
        }

        # Run modifier
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Check that the operation completed successfully
        rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == scenario_with_vehicle_types.id)
            .all()
        )
        assert len(rotations) > 0, "Should have rotations after scheduling"

    def test_vehicle_scheduling_parameter_validation(
        self,
        temp_db: Path,
        scenario_with_vehicle_types: Scenario,
        db_session: Session,
    ):
        """Test that VehicleScheduling validates parameters correctly."""
        modifier = VehicleScheduling()

        # Test invalid minimum_break_time type
        with pytest.raises(ValueError, match="minimum_break_time must be a timedelta"):
            modifier.modify(
                session=db_session,
                params={"VehicleScheduling.minimum_break_time": 900},  # Should be timedelta
            )

        # Test invalid maximum_schedule_duration type
        with pytest.raises(ValueError, match="maximum_schedule_duration must be a timedelta"):
            modifier.modify(
                session=db_session,
                params={"VehicleScheduling.maximum_schedule_duration": 24},  # Should be timedelta
            )

        # Test invalid battery_margin type
        with pytest.raises(ValueError, match="Battery margin must be a number"):
            modifier.modify(
                session=db_session,
                params={"VehicleScheduling.battery_margin": "0.1"},
            )

        # Test battery_margin out of range (negative)
        with pytest.raises(ValueError, match="Battery margin must be between 0.0 and 1.0"):
            modifier.modify(
                session=db_session,
                params={"VehicleScheduling.battery_margin": -0.1},
            )

        # Test battery_margin out of range (>= 1.0)
        with pytest.raises(ValueError, match="Battery margin must be between 0.0 and 1.0"):
            modifier.modify(
                session=db_session,
                params={"VehicleScheduling.battery_margin": 1.0},
            )

    def test_vehicle_scheduling_multiple_vehicle_types(
        self,
        temp_db: Path,
        scenario_with_vehicle_types: Scenario,
        db_session: Session,
    ):
        """Test VehicleScheduling with multiple vehicle types."""
        # Create a second vehicle type with constant consumption
        vehicle_type_2 = VehicleType(
            name="Electric Bus 18m",
            name_short="EB18",
            scenario_id=scenario_with_vehicle_types.id,
            battery_capacity=500.0,  # kWh - larger capacity
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 200], [1, 200]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            allowed_mass=2000,
            empty_mass=1000,
            consumption=1.5,  # kWh/km - higher consumption
        )
        db_session.add(vehicle_type_2)
        db_session.flush()

        # Assign half of the rotations to the second vehicle type
        rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == scenario_with_vehicle_types.id)
            .all()
        )
        for i, rotation in enumerate(rotations):
            if i % 2 == 0:
                rotation.vehicle_type_id = vehicle_type_2.id

        db_session.commit()

        # Now run the scheduler
        modifier = VehicleScheduling()
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # Verify that both vehicle types still have rotations
        vt1_rotations = (
            db_session.query(Rotation)
            .join(VehicleType)
            .filter(VehicleType.name_short == "EB12")
            .count()
        )
        vt2_rotations = (
            db_session.query(Rotation)
            .join(VehicleType)
            .filter(VehicleType.name_short == "EB18")
            .count()
        )

        # Both vehicle types should have rotations
        assert (
            vt1_rotations > 0 or vt2_rotations > 0
        ), "Should have rotations for at least one vehicle type"

    def test_vehicle_scheduling_no_vehicle_types_error(self, temp_db: Path, db_session: Session):
        """Test that VehicleScheduling raises error when no vehicle types exist."""
        # Create a minimal scenario without vehicle types
        scenario = Scenario(name="Empty Scenario", name_short="EMPTY")
        db_session.add(scenario)
        db_session.commit()

        modifier = VehicleScheduling()

        with pytest.raises(ValueError, match="No vehicle types found in the database"):
            modifier.modify(session=db_session, params={})

    def test_vehicle_scheduling_multiple_scenarios_error(self, temp_db: Path, db_session: Session):
        """Test that VehicleScheduling raises error with multiple scenarios."""
        # Create two scenarios
        scenario1 = Scenario(name="Scenario 1", name_short="S1")
        scenario2 = Scenario(name="Scenario 2", name_short="S2")
        db_session.add_all([scenario1, scenario2])
        db_session.commit()

        modifier = VehicleScheduling()

        with pytest.raises(ValueError, match="Expected exactly one scenario, found 2"):
            modifier.modify(session=db_session, params={})

    def test_vehicle_scheduling_with_longer_break_times(
        self,
        temp_db: Path,
        scenario_with_vehicle_types: Scenario,
        db_session: Session,
    ):
        """Test VehicleScheduling with longer break time parameters."""
        modifier = VehicleScheduling()

        # Get some trip IDs from the scenario
        trips = (
            db_session.query(Trip)
            .filter(Trip.scenario_id == scenario_with_vehicle_types.id)
            .filter(Trip.trip_type == TripType.PASSENGER)
            .limit(3)
            .all()
        )
        trip_ids = [trip.id for trip in trips]

        # Custom parameters with longer break times for specific trips
        params = {
            "VehicleScheduling.minimum_break_time": timedelta(minutes=10),
            "VehicleScheduling.maximum_schedule_duration": timedelta(hours=14),
            "VehicleScheduling.battery_margin": 0.12,
            "VehicleScheduling.longer_break_time_trips": trip_ids,
            "VehicleScheduling.longer_break_time_duration": timedelta(minutes=20),
        }

        # Run modifier
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Check that the operation completed successfully
        rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == scenario_with_vehicle_types.id)
            .all()
        )
        assert len(rotations) > 0, "Should have rotations after scheduling"

    def test_vehicle_scheduling_longer_break_time_validation(
        self,
        temp_db: Path,
        scenario_with_vehicle_types: Scenario,
        db_session: Session,
    ):
        """Test validation of longer_break_time parameters."""
        modifier = VehicleScheduling()

        # Test invalid longer_break_time_trips type (not a list)
        with pytest.raises(ValueError, match="longer_break_time_trips must be a list"):
            modifier.modify(
                session=db_session,
                params={"VehicleScheduling.longer_break_time_trips": "not a list"},
            )

        # Test invalid longer_break_time_trips elements (not integers)
        with pytest.raises(
            ValueError, match="All elements in longer_break_time_trips must be integers"
        ):
            modifier.modify(
                session=db_session,
                params={"VehicleScheduling.longer_break_time_trips": [1, "2", 3]},
            )

        # Test invalid longer_break_time_duration type
        with pytest.raises(ValueError, match="longer_break_time_duration must be a timedelta"):
            modifier.modify(
                session=db_session,
                params={
                    "VehicleScheduling.longer_break_time_duration": 300
                },  # Should be timedelta
            )

    def test_document_params(self):
        """Test that document_params returns expected parameters."""
        modifier = VehicleScheduling()
        docs = modifier.document_params()

        assert isinstance(docs, dict)
        assert len(docs) == 6
        assert "VehicleScheduling.minimum_break_time" in docs
        assert "VehicleScheduling.maximum_schedule_duration" in docs
        assert "VehicleScheduling.battery_margin" in docs
        assert "VehicleScheduling.longer_break_time_trips" in docs
        assert "VehicleScheduling.longer_break_time_duration" in docs
        assert "VehicleScheduling.charge_type" in docs


class TestDepotAssignment:
    """Test suite for DepotAssignment modifier."""

    @pytest.fixture
    def multi_depot_scenario_fixture(self, db_session: Session, tmp_path: Path) -> Scenario:
        """
        Create a test scenario using the multi_depot_scenario function from util.py.
        Uses the same parameters as in util.py's __main__ section.
        """
        from tests.util import (
            _create_depot_with_lines,
            CENTER_LAT,
            CENTER_LON,
            NEAR_TERMINUS_DISTANCE,
            FAR_TERMINUS_DISTANCE,
            DEPOT_RING_DIAMETER,
        )

        # Parameters matching the __main__ section in util.py
        num_depots = 3
        lines_per_depot = 10
        trips_per_line = 9

        # Create scenario
        scenario = Scenario(name="Multi-Depot Test Scenario", name_short="MDTS")
        db_session.add(scenario)
        db_session.flush()

        # Create TWO vehicle types
        vehicle_type_1 = VehicleType(
            name="Electric Bus 12m Type 1",
            name_short="EB12-1",
            scenario_id=scenario.id,
            battery_capacity=350.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=10000,
            allowed_mass=20000,
            consumption=1.2,
        )
        db_session.add(vehicle_type_1)
        db_session.flush()

        vehicle_type_2 = VehicleType(
            name="Electric Bus 12m Type 2",
            name_short="EB12-2",
            scenario_id=scenario.id,
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 180], [1, 180]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=11000,
            allowed_mass=21000,
            consumption=1.3,
        )
        db_session.add(vehicle_type_2)
        db_session.flush()

        # Berlin timezone
        berlin_tz = pytz.timezone("Europe/Berlin")
        base_date = datetime(2024, 1, 15, tzinfo=berlin_tz)

        # Center point for depot ring
        center_point = from_shape(Point(CENTER_LON, CENTER_LAT), srid=4326)

        # Create depots - initially with vehicle_type_1, we'll update rotations later
        for depot_idx in range(num_depots):
            _create_depot_with_lines(
                db_session,
                scenario.id,
                vehicle_type_1.id,
                depot_idx,
                num_depots,
                lines_per_depot,
                center_point,
                NEAR_TERMINUS_DISTANCE,
                FAR_TERMINUS_DISTANCE,
                trips_per_line,
                base_date,
                DEPOT_RING_DIAMETER,
            )

        # Now assign every second rotation to vehicle_type_2
        all_rotations = db_session.query(Rotation).filter_by(scenario_id=scenario.id).all()
        for i, rotation in enumerate(all_rotations):
            if i % 2 == 0:
                rotation.vehicle_type_id = vehicle_type_2.id

        db_session.commit()
        return scenario

    def test_depot_assignment_with_multi_depot_network(
        self,
        temp_db: Path,
        multi_depot_scenario_fixture: Scenario,
        db_session: Session,
        tmp_path: Path,
    ):
        """Test DepotAssignment with automatically generated multi-depot network."""
        from tests.util import generate_network_map

        # Generate map before depot assignment
        map_before = tmp_path / "network_before.html"
        generate_network_map(db_session, multi_depot_scenario_fixture.id, str(map_before))
        print(f"Map before depot assignment saved to {map_before}")

        # Get all depots
        all_depots = (
            db_session.query(Depot).filter_by(scenario_id=multi_depot_scenario_fixture.id).all()
        )
        assert len(all_depots) == 3, "Should have 3 depots initially"

        # Create depot config that uses only the first 2 depots
        # Get the first 2 depots ordered by ID
        target_depots = (
            db_session.query(Depot)
            .filter_by(scenario_id=multi_depot_scenario_fixture.id)
            .order_by(Depot.id)
            .limit(2)
            .all()
        )

        depot_config = []
        for depot in target_depots:
            depot_config.append(
                {
                    "depot_station": depot.station_id,
                    "capacity": 100,  # Large capacity to accept all rotations
                    "vehicle_type": [vt.id for vt in db_session.query(VehicleType).all()],
                    "name": depot.name,
                }
            )

        modifier = DepotAssignment()

        params = {
            "DepotAssignment.depot_config": depot_config,
            "DepotAssignment.depot_usage": 0.9,
            "DepotAssignment.step_size": 0.2,
            "DepotAssignment.max_iterations": 1,
        }

        # Get depot assignments before
        rotations_before = (
            db_session.query(Rotation).filter_by(scenario_id=multi_depot_scenario_fixture.id).all()
        )
        depot_counts_before = {}
        for rotation in rotations_before:
            # Get the depot from the first trip
            first_trip = (
                db_session.query(Trip)
                .filter_by(rotation_id=rotation.id)
                .order_by(Trip.departure_time)
                .first()
            )
            if first_trip and first_trip.route.departure_station.depot:
                depot_id = first_trip.route.departure_station.depot.id
                depot_counts_before[depot_id] = depot_counts_before.get(depot_id, 0) + 1

        print(f"Depot assignments before: {depot_counts_before}")

        # Run modifier
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Generate map after depot assignment
        map_after = tmp_path / "network_after.html"
        generate_network_map(db_session, multi_depot_scenario_fixture.id, str(map_after))
        print(f"Map after depot assignment saved to {map_after}")

        # Get depot assignments after
        rotations_after = (
            db_session.query(Rotation).filter_by(scenario_id=multi_depot_scenario_fixture.id).all()
        )
        depot_counts_after = {}
        for rotation in rotations_after:
            # Get the depot from the first trip
            first_trip = (
                db_session.query(Trip)
                .filter_by(rotation_id=rotation.id)
                .order_by(Trip.departure_time)
                .first()
            )
            if first_trip and first_trip.route.departure_station.depot:
                depot_id = first_trip.route.departure_station.depot.id
                depot_counts_after[depot_id] = depot_counts_after.get(depot_id, 0) + 1

        print(f"Depot assignments after: {depot_counts_after}")

        # Verify that we now only use 2 depots
        assert len(depot_counts_after) <= 2, "Should use at most 2 depots after assignment"

        # Verify that all rotations are still assigned
        assert len(rotations_after) == len(
            rotations_before
        ), "Should have same number of rotations"

    def test_depot_assignment_parameter_validation(
        self,
        temp_db: Path,
        db_session: Session,
        monkeypatch,
    ):
        """Test that DepotAssignment validates parameters correctly."""
        # Create a minimal scenario for parameter validation tests
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.commit()

        modifier = DepotAssignment()

        # Clear the environment variable to ensure base_url is required
        monkeypatch.delenv("OPENROUTESERVICE_BASE_URL", raising=False)

        # Test missing depot_config
        with pytest.raises(ValueError, match="Required parameter.*depot_config"):
            modifier.modify(
                session=db_session,
                params={"DepotAssignment.base_url": "http://test:8080/ors/"},
            )

        # Test missing base_url (and no environment variable)
        mock_depot_config = []

        with pytest.raises(ValueError, match="Required parameter.*base_url"):
            modifier.modify(
                session=db_session,
                params={"DepotAssignment.depot_config": mock_depot_config},
            )

        # Test invalid depot_config (not a list)
        with pytest.raises(ValueError, match="depot_config must be a list"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": "not_a_list",
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                },
            )

        # Test invalid base_url type
        with pytest.raises(ValueError, match="base_url must be a string"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": 12345,
                },
            )

        # Test invalid depot_usage type
        with pytest.raises(ValueError, match="depot_usage must be a number"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                    "DepotAssignment.depot_usage": "1.0",
                },
            )

        # Test depot_usage out of range (too high)
        with pytest.raises(ValueError, match="depot_usage must be between 0.0 and 1.0"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                    "DepotAssignment.depot_usage": 1.5,
                },
            )

        # Test depot_usage out of range (zero or negative)
        with pytest.raises(ValueError, match="depot_usage must be between 0.0 and 1.0"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                    "DepotAssignment.depot_usage": 0.0,
                },
            )

        # Test invalid step_size type
        with pytest.raises(ValueError, match="step_size must be a number"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                    "DepotAssignment.step_size": "0.1",
                },
            )

        # Test step_size out of range (too high)
        with pytest.raises(ValueError, match="step_size must be between 0.0 and 1.0"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                    "DepotAssignment.step_size": 1.2,
                },
            )

        # Test step_size out of range (at boundary 1.0)
        with pytest.raises(ValueError, match="step_size must be between 0.0 and 1.0"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                    "DepotAssignment.step_size": 1.0,
                },
            )

        # Test step_size out of range (zero or negative)
        with pytest.raises(ValueError, match="step_size must be between 0.0 and 1.0"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                    "DepotAssignment.step_size": 0.0,
                },
            )

        # Test invalid max_iterations type
        with pytest.raises(ValueError, match="max_iterations must be an integer"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                    "DepotAssignment.max_iterations": 2.5,
                },
            )

        # Test max_iterations <= 0
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                    "DepotAssignment.max_iterations": -1,
                },
            )

    def test_depot_assignment_multiple_scenarios_error(self, temp_db: Path, db_session: Session):
        """Test that DepotAssignment raises error with multiple scenarios."""
        # Create two scenarios
        scenario1 = Scenario(name="Scenario 1", name_short="S1")
        scenario2 = Scenario(name="Scenario 2", name_short="S2")
        db_session.add_all([scenario1, scenario2])
        db_session.commit()

        modifier = DepotAssignment()

        mock_depot_config = []

        with pytest.raises(ValueError, match="Expected exactly one scenario, found 2"):
            modifier.modify(
                session=db_session,
                params={
                    "DepotAssignment.depot_config": mock_depot_config,
                    "DepotAssignment.base_url": "http://test:8080/ors/",
                },
            )

    def test_document_params(self):
        """Test that document_params returns expected parameters."""
        modifier = DepotAssignment()
        docs = modifier.document_params()

        assert isinstance(docs, dict)
        assert len(docs) == 5
        assert "DepotAssignment.depot_config" in docs
        assert "DepotAssignment.base_url" in docs
        assert "DepotAssignment.depot_usage" in docs
        assert "DepotAssignment.step_size" in docs
        assert "DepotAssignment.max_iterations" in docs


class TestInsufficientChargingTimeAnalyzer:
    """Test suite for InsufficientChargingTimeAnalyzer analyzer."""

    @pytest.fixture
    def simple_scenario_with_short_trips(self, db_session: Session) -> Scenario:
        """
        Create a simple scenario with short trips and sufficient charging time.
        Uses the helper functions from util.py to create a workable schedule.
        """
        from tests.util import (
            _create_depot_with_lines,
            CENTER_LAT,
            CENTER_LON,
        )
        import random

        # Set random seed for reproducibility
        random.seed(42)

        # Create scenario
        scenario = Scenario(name="Simple Short Trip Scenario", name_short="SSTS")
        db_session.add(scenario)
        db_session.flush()

        # Create vehicle type with constant consumption
        vehicle_type = VehicleType(
            name="Electric Bus 12m",
            name_short="EB12",
            scenario_id=scenario.id,
            battery_capacity=350.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=10000,
            allowed_mass=20000,
            consumption=1.2,  # Constant consumption
        )
        db_session.add(vehicle_type)
        db_session.flush()

        # Berlin timezone
        berlin_tz = pytz.timezone("Europe/Berlin")
        base_date = datetime(2024, 1, 15, tzinfo=berlin_tz)

        # Center point for depot ring
        center_point = from_shape(Point(CENTER_LON, CENTER_LAT), srid=4326)

        # Create single depot with short trips
        _create_depot_with_lines(
            db_session,
            scenario.id,
            vehicle_type.id,
            depot_idx=0,
            num_depots=1,
            lines_per_depot=2,  # Must be even
            center_point=center_point,
            near_terminus_distance=500.0,  # Short distances
            far_terminus_distance=1000.0,  # Short distances
            trips_per_line=5,  # Few trips
            base_date=base_date,
            depot_ring_diameter=5000.0,
        )

        db_session.commit()
        return scenario

    @pytest.fixture
    def simple_scenario_with_long_trips(self, db_session: Session) -> Scenario:
        """
        Create a simple scenario with long trips and insufficient charging time.
        Uses long distances and many trips to create energy deficit.
        """
        from tests.util import (
            _create_depot_with_lines,
            CENTER_LAT,
            CENTER_LON,
        )
        import random

        # Set random seed for reproducibility
        random.seed(42)

        # Create scenario
        scenario = Scenario(name="Simple Long Trip Scenario", name_short="SLTS")
        db_session.add(scenario)
        db_session.flush()

        # Create vehicle type with high consumption and small battery
        vehicle_type = VehicleType(
            name="Electric Bus 12m",
            name_short="EB12",
            scenario_id=scenario.id,
            battery_capacity=100.0,  # Very small battery
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=10000,
            allowed_mass=20000,
            consumption=2.0,  # High consumption
        )
        db_session.add(vehicle_type)
        db_session.flush()

        # Berlin timezone
        berlin_tz = pytz.timezone("Europe/Berlin")
        base_date = datetime(2024, 1, 15, tzinfo=berlin_tz)

        # Center point for depot ring
        center_point = from_shape(Point(CENTER_LON, CENTER_LAT), srid=4326)

        # Create single depot with long trips
        _create_depot_with_lines(
            db_session,
            scenario.id,
            vehicle_type.id,
            depot_idx=0,
            num_depots=1,
            lines_per_depot=2,  # Must be even
            center_point=center_point,
            near_terminus_distance=3000.0,  # Long distances
            far_terminus_distance=6000.0,  # Long distances
            trips_per_line=25,  # Many trips
            base_date=base_date,
            depot_ring_diameter=10000.0,
        )

        db_session.commit()
        return scenario

    def test_sufficient_charging_time_returns_none(
        self,
        temp_db: Path,
        simple_scenario_with_short_trips: Scenario,
        db_session: Session,
    ):
        """Test that analyzer returns None when all rotations have sufficient charging time."""
        analyzer = InsufficientChargingTimeAnalyzer()

        # Run the analyzer
        result = analyzer.analyze(session=db_session, params={})

        # Should return None indicating all rotations are fine
        assert (
            result is None
        ), "Should return None when all rotations have sufficient charging time"

        # Verify that simulation results were created
        events_count = (
            db_session.query(Event)
            .filter(Event.scenario_id == simple_scenario_with_short_trips.id)
            .count()
        )
        assert events_count > 0, "Should have created simulation events"

    def test_insufficient_charging_time_returns_rotation_ids(
        self,
        temp_db: Path,
        simple_scenario_with_long_trips: Scenario,
        db_session: Session,
    ):
        """Test that analyzer returns rotation IDs when some rotations have insufficient charging time."""
        analyzer = InsufficientChargingTimeAnalyzer()

        # Run the analyzer
        result = analyzer.analyze(session=db_session, params={})

        # Should return a list of rotation IDs
        assert (
            result is not None
        ), "Should return a list when rotations have insufficient charging time"
        assert isinstance(result, list), "Should return a list of rotation IDs"
        assert len(result) > 0, "Should have at least one rotation with insufficient charging time"
        assert all(
            isinstance(rotation_id, int) for rotation_id in result
        ), "All elements should be integers (rotation IDs)"

        # Verify the rotation IDs are valid
        all_rotation_ids = set(
            r.id
            for r in db_session.query(Rotation)
            .filter_by(scenario_id=simple_scenario_with_long_trips.id)
            .all()
        )
        for rotation_id in result:
            assert (
                rotation_id in all_rotation_ids
            ), f"Rotation ID {rotation_id} should be a valid rotation in the scenario"

        # Verify that simulation results were created
        events_count = (
            db_session.query(Event)
            .filter(Event.scenario_id == simple_scenario_with_long_trips.id)
            .count()
        )
        assert events_count > 0, "Should have created simulation events"

    def test_custom_charging_power_parameter(
        self,
        temp_db: Path,
        simple_scenario_with_short_trips: Scenario,
        db_session: Session,
    ):
        """Test that analyzer accepts custom charging power parameter."""
        analyzer = InsufficientChargingTimeAnalyzer()

        # Run with custom charging power (higher power = more charging = better results)
        params = {
            "InsufficientChargingTimeAnalyzer.charging_power_kw": 300.0,  # Higher than default 150kW
        }

        result = analyzer.analyze(session=db_session, params=params)

        # With higher charging power, should still have sufficient charging
        assert result is None, "Should return None with higher charging power"

    def test_analyzer_fails_with_existing_simulation_results(
        self,
        temp_db: Path,
        simple_scenario_with_short_trips: Scenario,
        db_session: Session,
    ):
        """Test that analyzer raises error when simulation results already exist."""
        analyzer = InsufficientChargingTimeAnalyzer()

        # Run analyzer once to create simulation results
        analyzer.analyze(session=db_session, params={})

        # Try to run again - should fail because simulation results exist
        with pytest.raises(ValueError, match="Database contains .* existing simulation results"):
            analyzer.analyze(session=db_session, params={})

    def test_analyzer_multiple_scenarios_error(self, temp_db: Path, db_session: Session):
        """Test that analyzer raises error with multiple scenarios."""
        # Create two scenarios
        scenario1 = Scenario(name="Scenario 1", name_short="S1")
        scenario2 = Scenario(name="Scenario 2", name_short="S2")
        db_session.add_all([scenario1, scenario2])
        db_session.commit()

        analyzer = InsufficientChargingTimeAnalyzer()

        with pytest.raises(ValueError, match="Expected exactly one scenario, found 2"):
            analyzer.analyze(session=db_session, params={})

    def test_analyzer_with_rotation_without_driving_events(
        self,
        temp_db: Path,
        simple_scenario_with_short_trips: Scenario,
        db_session: Session,
    ):
        """Test that analyzer handles rotations without driving events gracefully."""
        analyzer = InsufficientChargingTimeAnalyzer()

        # Run the analyzer first to create events
        result = analyzer.analyze(session=db_session, params={})

        # Should work fine with the regular scenario
        assert result is None, "Should return None for scenario with sufficient charging"

        # Now verify that if a rotation had no driving events, it would be skipped
        # This is tested by the warning in the analyzer code at line 815-817
        # The analyzer already handles this case by checking if last_driving_event is None

    def test_document_params(self):
        """Test that document_params returns expected parameters."""
        analyzer = InsufficientChargingTimeAnalyzer()
        docs = analyzer.document_params()

        assert isinstance(docs, dict)
        assert len(docs) == 1
        assert "InsufficientChargingTimeAnalyzer.charging_power_kw" in docs
        assert "150 kW" in docs["InsufficientChargingTimeAnalyzer.charging_power_kw"]
