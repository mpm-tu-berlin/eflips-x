"""Tests for vehicle scheduling modifiers."""

from datetime import timedelta
from pathlib import Path

import pytest
from eflips.model import (
    Scenario,
    VehicleType,
    Rotation,
    Trip,
    TripType,
    VehicleClass,
)
from sqlalchemy.orm import Session

from eflips.x.steps.generators import BVGXMLIngester
from eflips.x.steps.modifiers.scheduling import VehicleScheduling


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
        assert len(docs) == 5
        assert "VehicleScheduling.minimum_break_time" in docs
        assert "VehicleScheduling.maximum_schedule_duration" in docs
        assert "VehicleScheduling.battery_margin" in docs
        assert "VehicleScheduling.longer_break_time_trips" in docs
        assert "VehicleScheduling.longer_break_time_duration" in docs
