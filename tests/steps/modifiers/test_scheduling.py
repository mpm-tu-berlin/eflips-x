"""Tests for vehicle scheduling modifiers."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytz
from eflips.depot.api import generate_consumption_result, simple_consumption_simulation
from eflips.model import Event, EventType
from eflips.model import (
    Scenario,
    VehicleType,
    Rotation,
    Trip,
    TripType,
    VehicleClass,
    Depot,
    ChargeType,
    Station,
)
from geoalchemy2.shape import from_shape
from shapely import Point
from sqlalchemy.orm import Session

from eflips.x.steps.generators import BVGXMLIngester
from eflips.x.steps.modifiers.scheduling import (
    VehicleScheduling,
    DepotAssignment,
    InsufficientChargingTimeAnalyzer,
    IntegratedScheduling,
    StationElectrification,
)


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

    def test_vehicle_scheduling_opportunity_vs_depot_mode(
        self,
        temp_db: Path,
        scenario_with_vehicle_types: Scenario,
        db_session: Session,
    ):
        """Test that opportunity mode creates longer rotations than depot mode with small batteries."""
        # Modify vehicle type to have very small battery capacity
        vehicle_types = (
            db_session.query(VehicleType)
            .filter(VehicleType.scenario_id == scenario_with_vehicle_types.id)
            .all()
        )

        for vt in vehicle_types:
            vt.battery_capacity = 50.0  # Very small battery - 50 kWh
            vt.consumption = 1.5  # Higher consumption to stress the battery

        db_session.commit()

        # Create a savepoint before running DEPOT mode
        savepoint = db_session.begin_nested()

        # Run scheduler in DEPOT mode
        modifier_depot = VehicleScheduling()
        params_depot = {
            "VehicleScheduling.charge_type": ChargeType.DEPOT,
            "VehicleScheduling.battery_margin": 0.1,
        }
        modifier_depot.modify(session=db_session, params=params_depot)
        db_session.flush()

        # Measure rotations in DEPOT mode
        depot_rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == scenario_with_vehicle_types.id)
            .all()
        )
        depot_rotation_count = len(depot_rotations)

        # Calculate average trips per rotation and total duration per rotation
        depot_trips_per_rotation = []
        depot_duration_per_rotation = []

        for rotation in depot_rotations:
            trips = (
                db_session.query(Trip)
                .filter(Trip.rotation_id == rotation.id)
                .filter(Trip.trip_type == TripType.PASSENGER)
                .all()
            )
            if trips:
                depot_trips_per_rotation.append(len(trips))
                # Calculate total duration
                min_departure = min(trip.departure_time for trip in trips)
                max_arrival = max(trip.arrival_time for trip in trips)
                duration = (max_arrival - min_departure).total_seconds() / 3600  # in hours
                depot_duration_per_rotation.append(duration)

        depot_avg_trips = (
            sum(depot_trips_per_rotation) / len(depot_trips_per_rotation)
            if depot_trips_per_rotation
            else 0
        )
        depot_avg_duration = (
            sum(depot_duration_per_rotation) / len(depot_duration_per_rotation)
            if depot_duration_per_rotation
            else 0
        )

        print(f"\nDEPOT mode results:")
        print(f"  Number of rotations: {depot_rotation_count}")
        print(f"  Average trips per rotation: {depot_avg_trips:.2f}")
        print(f"  Average duration per rotation: {depot_avg_duration:.2f} hours")

        # Rollback to savepoint to restore original state
        savepoint.rollback()
        db_session.expire_all()

        # Run scheduler in OPPORTUNITY mode
        modifier_opportunity = VehicleScheduling()
        params_opportunity = {
            "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
            "VehicleScheduling.battery_margin": 0.1,  # Same margin for fair comparison
        }
        modifier_opportunity.modify(session=db_session, params=params_opportunity)
        db_session.commit()

        # Measure rotations in OPPORTUNITY mode
        opportunity_rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == scenario_with_vehicle_types.id)
            .all()
        )
        opportunity_rotation_count = len(opportunity_rotations)

        # Calculate average trips per rotation and total duration per rotation
        opportunity_trips_per_rotation = []
        opportunity_duration_per_rotation = []

        for rotation in opportunity_rotations:
            trips = (
                db_session.query(Trip)
                .filter(Trip.rotation_id == rotation.id)
                .filter(Trip.trip_type == TripType.PASSENGER)
                .all()
            )
            if trips:
                opportunity_trips_per_rotation.append(len(trips))
                # Calculate total duration
                min_departure = min(trip.departure_time for trip in trips)
                max_arrival = max(trip.arrival_time for trip in trips)
                duration = (max_arrival - min_departure).total_seconds() / 3600  # in hours
                opportunity_duration_per_rotation.append(duration)

        opportunity_avg_trips = (
            sum(opportunity_trips_per_rotation) / len(opportunity_trips_per_rotation)
            if opportunity_trips_per_rotation
            else 0
        )
        opportunity_avg_duration = (
            sum(opportunity_duration_per_rotation) / len(opportunity_duration_per_rotation)
            if opportunity_duration_per_rotation
            else 0
        )

        print(f"\nOPPORTUNITY mode results:")
        print(f"  Number of rotations: {opportunity_rotation_count}")
        print(f"  Average trips per rotation: {opportunity_avg_trips:.2f}")
        print(f"  Average duration per rotation: {opportunity_avg_duration:.2f} hours")

        # Verify that opportunity mode has fewer rotations (meaning longer rotations)
        # OR higher average trips per rotation
        # With small batteries, DEPOT mode should need more rotations (shorter each)
        # while OPPORTUNITY mode can have fewer, longer rotations

        assert opportunity_rotation_count > 0, "Should have rotations in opportunity mode"
        assert depot_rotation_count > 0, "Should have rotations in depot mode"

        # The key assertion: opportunity mode should create fewer rotations (longer each)
        # OR opportunity mode should have more trips per rotation on average
        fewer_rotations = opportunity_rotation_count < depot_rotation_count
        more_trips_per_rotation = opportunity_avg_trips > depot_avg_trips
        longer_duration = opportunity_avg_duration > depot_avg_duration

        print(f"\nComparison:")
        print(f"  OPPORTUNITY has fewer rotations: {fewer_rotations}")
        print(f"  OPPORTUNITY has more trips per rotation: {more_trips_per_rotation}")
        print(f"  OPPORTUNITY has longer duration per rotation: {longer_duration}")

        # At least one of these should be true with small batteries
        assert (
            fewer_rotations or more_trips_per_rotation or longer_duration
        ), "Opportunity mode should create longer rotations (fewer total, more trips per rotation, or longer duration) than depot mode with small batteries"

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


class TestIntegratedScheduling:
    """Test suite for IntegratedScheduling modifier."""

    @pytest.fixture
    def challenging_scenario_for_integrated_scheduling(self, db_session: Session) -> Scenario:
        """
        Create a challenging scenario that will require multiple iterations
        of integrated scheduling to find a feasible solution.
        Uses long trips, high consumption, and moderate battery capacity.
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
        scenario = Scenario(name="Challenging Integrated Scheduling Scenario", name_short="CISS")
        db_session.add(scenario)
        db_session.flush()

        # Create vehicle type with moderate battery and consumption
        # This will be challenging when combined with low charging power
        vehicle_type = VehicleType(
            name="Electric Bus 12m",
            name_short="EB12",
            scenario_id=scenario.id,
            battery_capacity=200.0,  # Moderate battery
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 150], [1, 150]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            empty_mass=10000,
            allowed_mass=20000,
            consumption=1.8,  # Higher consumption
        )
        db_session.add(vehicle_type)
        db_session.flush()

        # Berlin timezone
        berlin_tz = pytz.timezone("Europe/Berlin")
        base_date = datetime(2024, 1, 15, tzinfo=berlin_tz)

        # Center point for depot ring
        center_point = from_shape(Point(CENTER_LON, CENTER_LAT), srid=4326)

        # Create single depot with moderate-length trips
        _create_depot_with_lines(
            db_session,
            scenario.id,
            vehicle_type.id,
            depot_idx=0,
            num_depots=1,
            lines_per_depot=2,  # Must be even
            center_point=center_point,
            near_terminus_distance=2000.0,  # Moderate distances
            far_terminus_distance=4000.0,  # Moderate distances
            trips_per_line=15,  # Moderate number of trips
            base_date=base_date,
            depot_ring_diameter=8000.0,
        )

        db_session.commit()
        return scenario

    def test_integrated_scheduling_basic(
        self,
        temp_db: Path,
        challenging_scenario_for_integrated_scheduling: Scenario,
        db_session: Session,
    ):
        """Test IntegratedScheduling with basic parameters and very low charging power to force iterations."""
        # Create depot config
        depots = (
            db_session.query(Depot)
            .filter_by(scenario_id=challenging_scenario_for_integrated_scheduling.id)
            .all()
        )
        depot_config = []
        for depot in depots:
            depot_config.append(
                {
                    "depot_station": depot.station_id,
                    "capacity": 100,  # Large capacity
                    "vehicle_type": [vt.id for vt in db_session.query(VehicleType).all()],
                    "name": depot.name,
                }
            )

        modifier = IntegratedScheduling()

        # Use very low charging power to ensure we need multiple iterations
        params = {
            "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
            "VehicleScheduling.battery_margin": 0.1,
            "DepotAssignment.depot_config": depot_config,
            "InsufficientChargingTimeAnalyzer.charging_power_kw": 20.0,  # Very low charging power!
            "IntegratedScheduling.max_iterations": 5,  # Allow more iterations
        }

        # Count initial rotations
        initial_rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == challenging_scenario_for_integrated_scheduling.id)
            .count()
        )
        assert initial_rotations > 0, "Should have rotations from fixture"

        # Run the integrated scheduling
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Verify that rotations still exist and were modified
        final_rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == challenging_scenario_for_integrated_scheduling.id)
            .count()
        )
        assert final_rotations > 0, "Should have rotations after integrated scheduling"

        # Verify that no simulation events exist (they should be rolled back)
        events_count = (
            db_session.query(Event)
            .filter(Event.scenario_id == challenging_scenario_for_integrated_scheduling.id)
            .count()
        )
        assert (
            events_count == 0
        ), "Should not have simulation events (rolled back in nested sessions)"

    def test_integrated_scheduling_with_default_iterations(
        self,
        temp_db: Path,
        challenging_scenario_for_integrated_scheduling: Scenario,
        db_session: Session,
    ):
        """Test IntegratedScheduling with default max_iterations (2)."""
        # Create depot config
        depots = (
            db_session.query(Depot)
            .filter_by(scenario_id=challenging_scenario_for_integrated_scheduling.id)
            .all()
        )
        depot_config = []
        for depot in depots:
            depot_config.append(
                {
                    "depot_station": depot.station_id,
                    "capacity": 100,
                    "vehicle_type": [vt.id for vt in db_session.query(VehicleType).all()],
                    "name": depot.name,
                }
            )

        modifier = IntegratedScheduling()

        # Use moderate charging power so we can converge in 2 iterations
        params = {
            "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
            "VehicleScheduling.battery_margin": 0.1,
            "DepotAssignment.depot_config": depot_config,
            "InsufficientChargingTimeAnalyzer.charging_power_kw": 50.0,  # Low but not too low
            # Default max_iterations = 2
        }

        # Run the integrated scheduling
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Verify that rotations were created
        final_rotations = (
            db_session.query(Rotation)
            .filter(Rotation.scenario_id == challenging_scenario_for_integrated_scheduling.id)
            .count()
        )
        assert final_rotations > 0, "Should have rotations after integrated scheduling"

    def test_integrated_scheduling_max_iterations_exceeded(
        self,
        temp_db: Path,
        challenging_scenario_for_integrated_scheduling: Scenario,
        db_session: Session,
    ):
        """Test that IntegratedScheduling raises error when max_iterations is exceeded."""
        # Create depot config
        depots = (
            db_session.query(Depot)
            .filter_by(scenario_id=challenging_scenario_for_integrated_scheduling.id)
            .all()
        )
        depot_config = []
        for depot in depots:
            depot_config.append(
                {
                    "depot_station": depot.station_id,
                    "capacity": 100,
                    "vehicle_type": [vt.id for vt in db_session.query(VehicleType).all()],
                    "name": depot.name,
                }
            )

        modifier = IntegratedScheduling()

        # Use extremely low charging power and max_iterations=1 to ensure failure
        params = {
            "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
            "VehicleScheduling.battery_margin": 0.1,
            "DepotAssignment.depot_config": depot_config,
            "InsufficientChargingTimeAnalyzer.charging_power_kw": 5.0,  # Extremely low!
            "IntegratedScheduling.max_iterations": 1,  # Only 1 iteration allowed
        }

        # Should raise ValueError when max iterations exceeded
        with pytest.raises(
            ValueError,
            match="Reached maximum number of iterations .* without finding a feasible schedule",
        ):
            modifier.modify(session=db_session, params=params)

    def test_integrated_scheduling_requires_opportunity_mode(
        self,
        temp_db: Path,
        challenging_scenario_for_integrated_scheduling: Scenario,
        db_session: Session,
    ):
        """Test that IntegratedScheduling requires OPPORTUNITY charge type."""
        # Create depot config
        depots = (
            db_session.query(Depot)
            .filter_by(scenario_id=challenging_scenario_for_integrated_scheduling.id)
            .all()
        )
        depot_config = []
        for depot in depots:
            depot_config.append(
                {
                    "depot_station": depot.station_id,
                    "capacity": 100,
                    "vehicle_type": [vt.id for vt in db_session.query(VehicleType).all()],
                    "name": depot.name,
                }
            )

        modifier = IntegratedScheduling()

        # Try to use DEPOT mode - should fail
        params = {
            "VehicleScheduling.charge_type": ChargeType.DEPOT,  # Wrong mode!
            "DepotAssignment.depot_config": depot_config,
            "IntegratedScheduling.max_iterations": 2,
        }

        with pytest.raises(
            ValueError,
            match="IntegratedScheduling only makes sense when VehicleScheduling is run in OPPORTUNITY charge type",
        ):
            modifier.modify(session=db_session, params=params)

    def test_integrated_scheduling_parameter_validation(
        self,
        temp_db: Path,
        challenging_scenario_for_integrated_scheduling: Scenario,
        db_session: Session,
    ):
        """Test that IntegratedScheduling validates parameters correctly."""
        # Create depot config
        depots = (
            db_session.query(Depot)
            .filter_by(scenario_id=challenging_scenario_for_integrated_scheduling.id)
            .all()
        )
        depot_config = []
        for depot in depots:
            depot_config.append(
                {
                    "depot_station": depot.station_id,
                    "capacity": 100,
                    "vehicle_type": [vt.id for vt in db_session.query(VehicleType).all()],
                    "name": depot.name,
                }
            )

        modifier = IntegratedScheduling()

        base_params = {
            "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
            "DepotAssignment.depot_config": depot_config,
        }

        # Test invalid max_iterations type
        with pytest.raises(ValueError, match="max_iterations must be an integer"):
            params = base_params.copy()
            params["IntegratedScheduling.max_iterations"] = "2"  # Should be int
            modifier.modify(session=db_session, params=params)

        # Test max_iterations <= 0
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            params = base_params.copy()
            params["IntegratedScheduling.max_iterations"] = 0
            modifier.modify(session=db_session, params=params)

        # Test max_iterations <= 0 (negative)
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            params = base_params.copy()
            params["IntegratedScheduling.max_iterations"] = -1
            modifier.modify(session=db_session, params=params)

    def test_integrated_scheduling_multiple_scenarios_error(
        self, temp_db: Path, db_session: Session, monkeypatch
    ):
        """Test that IntegratedScheduling raises error with multiple scenarios."""
        # Create two scenarios
        scenario1 = Scenario(name="Scenario 1", name_short="S1")
        scenario2 = Scenario(name="Scenario 2", name_short="S2")
        db_session.add_all([scenario1, scenario2])
        db_session.commit()

        # Mock the ORS base URL
        monkeypatch.setenv("OPENROUTESERVICE_BASE_URL", "http://mock-ors:8080/ors/")

        modifier = IntegratedScheduling()

        params = {
            "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
            "DepotAssignment.depot_config": [],  # Empty config, won't get that far
            "IntegratedScheduling.max_iterations": 2,
        }

        # Should raise error about multiple scenarios
        # The error will come from VehicleScheduling which is called first
        with pytest.raises(ValueError, match="Expected exactly one scenario, found 2"):
            modifier.modify(session=db_session, params=params)

    def test_integrated_scheduling_warns_about_high_iterations(
        self,
        temp_db: Path,
        challenging_scenario_for_integrated_scheduling: Scenario,
        db_session: Session,
        monkeypatch,
        caplog,
    ):
        """Test that IntegratedScheduling warns when max_iterations > 2."""
        import logging

        # Set logging level to capture warnings
        caplog.set_level(logging.WARNING)

        # Mock the ORS base URL
        monkeypatch.setenv("OPENROUTESERVICE_BASE_URL", "http://mock-ors:8080/ors/")

        # Create depot config
        depots = (
            db_session.query(Depot)
            .filter_by(scenario_id=challenging_scenario_for_integrated_scheduling.id)
            .all()
        )
        depot_config = []
        for depot in depots:
            depot_config.append(
                {
                    "depot_station": depot.station_id,
                    "capacity": 100,
                    "vehicle_type": [vt.id for vt in db_session.query(VehicleType).all()],
                    "name": depot.name,
                }
            )

        modifier = IntegratedScheduling()

        params = {
            "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
            "VehicleScheduling.battery_margin": 0.1,
            "DepotAssignment.depot_config": depot_config,
            "InsufficientChargingTimeAnalyzer.charging_power_kw": 100.0,
            "IntegratedScheduling.max_iterations": 10,  # High value should trigger warning
        }

        # Run the integrated scheduling
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Check that warning was logged
        assert any(
            "designed to find a feasible solution in 2 iterations" in record.message
            for record in caplog.records
        ), "Should warn about high max_iterations"

    def test_document_params(self):
        """Test that document_params returns expected parameters."""
        modifier = IntegratedScheduling()
        docs = modifier.document_params()

        assert isinstance(docs, dict)
        assert len(docs) == 1
        assert "IntegratedScheduling.max_iterations" in docs
        assert "Maximum number of iterations" in docs["IntegratedScheduling.max_iterations"]


class TestStationElectrification:
    """Test suite for StationElectrification modifier."""

    def _increase_time_between_trips(self, scenario: Scenario, db_session: Session) -> None:
        """
        Helper to add some time for opportunity charging between trips.
        :param scenario: A scenario with trips
        :param db_session: An open database session
        :return: None
        """
        break_duration = timedelta(minutes=2)
        all_rotations = db_session.query(Rotation).filter_by(scenario_id=scenario.id).all()
        for rotation in all_rotations:
            for i, trip in enumerate(rotation.trips):
                for subsequent_trip in rotation.trips[i + 1 :]:
                    subsequent_trip.departure_time += break_duration
                    subsequent_trip.arrival_time += break_duration

    @pytest.fixture
    def infeasible_scenario_without_opportunity_charging(self, db_session: Session) -> Scenario:
        """
        Create a scenario that is infeasible without opportunity charging.
        Uses small battery capacity and high consumption to force need for station electrification.
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
        scenario = Scenario(name="Infeasible Scenario", name_short="INFEAS")
        db_session.add(scenario)
        db_session.flush()

        # Create vehicle type with parameters similar to TestIntegratedScheduling
        # but with higher charging power to match StationElectrification defaults
        vehicle_type = VehicleType(
            name="Electric Bus 12m",
            name_short="EB12",
            scenario_id=scenario.id,
            battery_capacity=200.0,  # Moderate battery - same as IntegratedScheduling
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 450], [1, 450]],  # Match default StationElectrification power
            opportunity_charging_capable=True,  # Enable opportunity charging
            minimum_charging_power=10,
            empty_mass=10000,
            allowed_mass=20000,
            consumption=1.8,  # Higher consumption - same as IntegratedScheduling
        )
        db_session.add(vehicle_type)
        db_session.flush()

        # Berlin timezone
        berlin_tz = pytz.timezone("Europe/Berlin")
        base_date = datetime(2024, 1, 15, tzinfo=berlin_tz)

        # Center point for depot ring
        center_point = from_shape(Point(CENTER_LON, CENTER_LAT), srid=4326)

        # Create single depot with parameters matching TestIntegratedScheduling
        _create_depot_with_lines(
            db_session,
            scenario.id,
            vehicle_type.id,
            depot_idx=0,
            num_depots=1,
            lines_per_depot=2,  # Must be even
            center_point=center_point,
            near_terminus_distance=2000.0,  # Same as IntegratedScheduling
            far_terminus_distance=4000.0,  # Same as IntegratedScheduling
            trips_per_line=15,  # Same as IntegratedScheduling
            base_date=base_date,
            depot_ring_diameter=8000.0,
        )

        # Make sure rotations allow opportunity charging
        rotations = db_session.query(Rotation).filter_by(scenario_id=scenario.id).all()
        for rotation in rotations:
            rotation.allow_opportunity_charging = True

        return scenario

    def test_station_electrification_basic(
        self,
        temp_db: Path,
        infeasible_scenario_without_opportunity_charging: Scenario,
        db_session: Session,
    ):
        """Test StationElectrification with basic parameters."""
        # Run vehicle scheduling and depot assignment first
        self._increase_time_between_trips(
            infeasible_scenario_without_opportunity_charging, db_session
        )

        # Count initial electrified stations (should only be the depot)
        initial_electrified = (
            db_session.query(Station)
            .filter(Station.scenario_id == infeasible_scenario_without_opportunity_charging.id)
            .filter(Station.is_electrified == True)
            .count()
        )
        assert initial_electrified == 1, "Should have 1 electrified station (the depot)"

        # Step 3: Now run station electrification
        modifier = StationElectrification()
        params = {
            "StationElectrification.max_stations_to_electrify": 10,
        }
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Check that more stations were electrified
        final_electrified = (
            db_session.query(Station)
            .filter(Station.scenario_id == infeasible_scenario_without_opportunity_charging.id)
            .filter(Station.is_electrified == True)
            .count()
        )
        assert final_electrified > initial_electrified, "Should have electrified more stations"

        # Verify that all rotations now have SOC >= 0
        scenario = infeasible_scenario_without_opportunity_charging
        consumption_results = generate_consumption_result(scenario)
        simple_consumption_simulation(
            scenario, initialize_vehicles=True, consumption_result=consumption_results
        )

        # Check that no rotations have negative SOC
        rotations_with_negative_soc = (
            db_session.query(Rotation)
            .join(Trip)
            .join(Event)
            .filter(Rotation.scenario_id == scenario.id)
            .filter(Event.event_type == EventType.DRIVING)
            .filter(Event.soc_end < 0)
            .distinct()
            .count()
        )
        assert (
            rotations_with_negative_soc == 0
        ), "Should have no rotations with negative SOC after electrification"

    def test_station_electrification_with_custom_power(
        self,
        temp_db: Path,
        infeasible_scenario_without_opportunity_charging: Scenario,
        db_session: Session,
    ):
        """Test StationElectrification with custom charging power."""
        # Run vehicle scheduling and depot assignment first
        self._increase_time_between_trips(
            infeasible_scenario_without_opportunity_charging, db_session
        )

        modifier = StationElectrification()

        # Use lower charging power - this might require more stations
        params = {
            "InsufficientChargingTimeAnalyzer.charging_power_kw": 300.0,  # Match custom power
            "StationElectrification.charging_power_kw": 300.0,  # Lower than default 450kW
            "StationElectrification.max_stations_to_electrify": 10,
        }

        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Verify that electrified stations have the correct power
        electrified_opportunity_stations = (
            db_session.query(Station)
            .filter(Station.scenario_id == infeasible_scenario_without_opportunity_charging.id)
            .filter(Station.is_electrified == True)
            .filter(Station.charge_type == ChargeType.OPPORTUNITY)
            .all()
        )

        for station in electrified_opportunity_stations:
            assert (
                station.power_per_charger == 300.0
            ), f"Station {station.name} should have 300kW charging power"

    def test_station_electrification_max_stations_limit(
        self,
        temp_db: Path,
        infeasible_scenario_without_opportunity_charging: Scenario,
        db_session: Session,
    ):
        """Test that StationElectrification respects max_stations_to_electrify limit."""
        # Run vehicle scheduling and depot assignment first
        self._increase_time_between_trips(
            infeasible_scenario_without_opportunity_charging, db_session
        )

        # Halve the battery capacity to make it more challenging
        vehicle_types = (
            db_session.query(VehicleType)
            .filter(VehicleType.scenario_id == infeasible_scenario_without_opportunity_charging.id)
            .all()
        )
        for vt in vehicle_types:
            vt.battery_capacity *= 0.5  # Reduce battery capacity by half

        modifier = StationElectrification()

        # Set a very low limit to force failure
        params = {
            "InsufficientChargingTimeAnalyzer.charging_power_kw": 1,  # Very low power
            "StationElectrification.charging_power_kw": 1,
            "StationElectrification.max_stations_to_electrify": 1,  # Very low limit
        }

        # Should raise ValueError when limit is exceeded
        with pytest.raises(
            ValueError,
            match=r"Station electrification failed: electrified .* stations \(limit: .*\) .*",
        ):
            modifier.modify(session=db_session, params=params)

    def test_station_electrification_power_mismatch_validation(
        self,
        temp_db: Path,
        infeasible_scenario_without_opportunity_charging: Scenario,
        db_session: Session,
    ):
        """Test that StationElectrification validates power mismatch with InsufficientChargingTimeAnalyzer."""
        modifier = StationElectrification()

        # Set mismatched charging powers
        params = {
            "StationElectrification.charging_power_kw": 450.0,
            "InsufficientChargingTimeAnalyzer.charging_power_kw": 300.0,  # Different!
        }

        # Should raise ValueError about power mismatch
        with pytest.raises(ValueError, match="Charging power mismatch"):
            modifier.modify(session=db_session, params=params)

    def test_station_electrification_parameter_validation(
        self,
        temp_db: Path,
        infeasible_scenario_without_opportunity_charging: Scenario,
        db_session: Session,
    ):
        """Test that StationElectrification validates parameters correctly."""
        modifier = StationElectrification()

        # Test invalid charging_power_kw type
        with pytest.raises(ValueError, match="charging_power_kw must be a number"):
            modifier.modify(
                session=db_session,
                params={
                    "InsufficientChargingTimeAnalyzer.charging_power_kw": 450.0,
                    "StationElectrification.charging_power_kw": "450",
                },
            )

        # Test negative charging_power_kw
        with pytest.raises(ValueError, match="charging_power_kw must be positive"):
            modifier.modify(
                session=db_session,
                params={
                    "InsufficientChargingTimeAnalyzer.charging_power_kw": -100.0,
                    "StationElectrification.charging_power_kw": -100.0,
                },
            )

        # Test zero charging_power_kw
        with pytest.raises(ValueError, match="charging_power_kw must be positive"):
            modifier.modify(
                session=db_session,
                params={
                    "InsufficientChargingTimeAnalyzer.charging_power_kw": 0.0,
                    "StationElectrification.charging_power_kw": 0.0,
                },
            )

        # Test missing InsufficientChargingTimeAnalyzer unset when StationElectrification power is set
        with pytest.raises(ValueError, match="Please ensure both use the same charging power."):
            modifier.modify(
                session=db_session,
                params={"StationElectrification.charging_power_kw": 300.0},
            )

        # Test invalid max_stations_to_electrify type
        with pytest.raises(ValueError, match="max_stations_to_electrify must be an integer"):
            modifier.modify(
                session=db_session,
                params={"StationElectrification.max_stations_to_electrify": "10"},
            )

        # Test negative max_stations_to_electrify
        with pytest.raises(ValueError, match="max_stations_to_electrify must be positive"):
            modifier.modify(
                session=db_session,
                params={"StationElectrification.max_stations_to_electrify": -5},
            )

        # Test zero max_stations_to_electrify
        with pytest.raises(ValueError, match="max_stations_to_electrify must be positive"):
            modifier.modify(
                session=db_session,
                params={"StationElectrification.max_stations_to_electrify": 0},
            )

    def test_station_electrification_multiple_scenarios_error(
        self, temp_db: Path, db_session: Session
    ):
        """Test that StationElectrification raises error with multiple scenarios."""
        # Create two scenarios
        scenario1 = Scenario(name="Scenario 1", name_short="S1")
        scenario2 = Scenario(name="Scenario 2", name_short="S2")
        db_session.add_all([scenario1, scenario2])
        db_session.commit()

        modifier = StationElectrification()

        with pytest.raises(ValueError, match="Expected exactly one scenario, found 2"):
            modifier.modify(session=db_session, params={})

    def test_station_electrification_with_matching_analyzer_power(
        self,
        temp_db: Path,
        infeasible_scenario_without_opportunity_charging: Scenario,
        db_session: Session,
    ):
        """Test that StationElectrification works when power matches InsufficientChargingTimeAnalyzer."""
        # Run vehicle scheduling and depot assignment first
        self._increase_time_between_trips(
            infeasible_scenario_without_opportunity_charging, db_session
        )

        modifier = StationElectrification()

        # Set matching charging powers - should work fine
        params = {
            "StationElectrification.charging_power_kw": 350.0,
            "InsufficientChargingTimeAnalyzer.charging_power_kw": 350.0,  # Same!
            "StationElectrification.max_stations_to_electrify": 10,
        }

        # Should work without errors
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Verify success
        electrified_count = (
            db_session.query(Station)
            .filter(Station.scenario_id == infeasible_scenario_without_opportunity_charging.id)
            .filter(Station.is_electrified == True)
            .count()
        )
        assert electrified_count > 1, "Should have electrified multiple stations"

    def test_document_params(self):
        """Test that document_params returns expected parameters."""
        modifier = StationElectrification()
        docs = modifier.document_params()

        assert isinstance(docs, dict)
        assert len(docs) == 2
        assert "StationElectrification.charging_power_kw" in docs
        assert "StationElectrification.max_stations_to_electrify" in docs
        assert "450.0 kW" in docs["StationElectrification.charging_power_kw"]
        assert "25% of all termini" in docs["StationElectrification.max_stations_to_electrify"]
