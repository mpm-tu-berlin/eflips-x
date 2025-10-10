"""Tests for BVG-specific modifier tools."""

import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import eflips.model
import pytest
from eflips.model import (
    Rotation,
    Scenario,
    Trip,
    Route,
    Station,
    VehicleType,
    TripType,
    ConsistencyWarning,
)
from sqlalchemy.orm import Session

from eflips.x.framework import PipelineContext
from eflips.x.steps.modifiers.bvg_tools import RemoveUnusedVehicleTypes


class TestRemoveUnusedVehicleTypes:
    """Test suite for RemoveUnusedVehicleTypes modifier."""

    @pytest.fixture
    def scenario_with_vehicle_types(self, db_session: Session) -> Scenario:
        """Create a test scenario with multiple vehicle types and rotations."""
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.flush()

        # Create stations
        depot_station = Station(
            name="Test Depot",
            name_short="DEPOT",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        end_station = Station(
            name="End Station",
            name_short="END",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        db_session.add_all([depot_station, end_station])
        db_session.flush()

        # Create vehicle types that should be kept
        vt_gn = VehicleType(
            name="Articulated Bus Old",
            scenario_id=scenario.id,
            name_short="GN",
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 300], [1, 300]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
        )
        vt_geg = VehicleType(
            name="Articulated Bus Variant",
            scenario_id=scenario.id,
            name_short="GEG",
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 300], [1, 300]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
        )
        vt_en = VehicleType(
            name="Single Decker Old",
            scenario_id=scenario.id,
            name_short="EN",
            battery_capacity=350.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 250], [1, 250]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            consumption=1,
        )
        vt_d = VehicleType(
            name="Double Decker Old",
            scenario_id=scenario.id,
            name_short="D",
            battery_capacity=450.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 350], [1, 350]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            consumption=1,
        )

        # Create a vehicle type that should be removed
        vt_unused = VehicleType(
            name="Unused Type",
            scenario_id=scenario.id,
            name_short="UNUSED",
            battery_capacity=300.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 200], [1, 200]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
            consumption=1,
        )

        db_session.add_all([vt_gn, vt_geg, vt_en, vt_d, vt_unused])
        db_session.flush()

        # Create routes
        route1 = Route(
            name="Route 1",
            name_short="R1",
            scenario_id=scenario.id,
            departure_station=depot_station,
            arrival_station=end_station,
            distance=5000,
        )
        route2 = Route(
            name="Route 2 Return",
            name_short="R2",
            scenario_id=scenario.id,
            departure_station=end_station,
            arrival_station=depot_station,
            distance=5000,
        )
        db_session.add_all([route1, route2])
        db_session.flush()

        # Create rotations with trips for each vehicle type that should be kept
        for i, vt in enumerate([vt_gn, vt_geg, vt_en, vt_d]):
            rotation = Rotation(
                name=f"Rotation {i}",
                scenario_id=scenario.id,
                vehicle_type=vt,
                allow_opportunity_charging=False,
            )
            db_session.add(rotation)
            db_session.flush()

            trip1 = Trip(
                rotation=rotation,
                route=route1,
                scenario_id=scenario.id,
                trip_type=TripType.PASSENGER,
                departure_time=datetime(2024, 1, 1, 8, 0, tzinfo=ZoneInfo("UTC")),
                arrival_time=datetime(2024, 1, 1, 8, 30, tzinfo=ZoneInfo("UTC")),
            )
            trip2 = Trip(
                rotation=rotation,
                route=route2,
                scenario_id=scenario.id,
                trip_type=TripType.PASSENGER,
                departure_time=datetime(2024, 1, 1, 8, 45, tzinfo=ZoneInfo("UTC")),
                arrival_time=datetime(2024, 1, 1, 9, 15, tzinfo=ZoneInfo("UTC")),
            )
            db_session.add_all([trip1, trip2])

        # Create a rotation with the unused vehicle type
        rotation_unused = Rotation(
            name="Rotation Unused",
            scenario_id=scenario.id,
            vehicle_type=vt_unused,
            allow_opportunity_charging=False,
        )
        db_session.add(rotation_unused)
        db_session.flush()

        trip_unused = Trip(
            rotation=rotation_unused,
            route=route1,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 10, 0, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 10, 30, tzinfo=ZoneInfo("UTC")),
        )
        db_session.add(trip_unused)

        db_session.commit()
        return scenario

    def test_remove_unused_vehicle_types_with_defaults(
        self, temp_db: Path, scenario_with_vehicle_types, db_session: Session
    ):
        """Test RemoveUnusedVehicleTypes modifier using default parameters."""
        modifier = RemoveUnusedVehicleTypes()

        # Create pipeline context
        context = PipelineContext(work_dir=temp_db.parent, current_db=temp_db)

        # Run modifier
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            modifier.modify(
                session=db_session,
                params={},
            )

            # Check that warnings were emitted for using defaults
            non_consistency_warnings = [wa for wa in w if wa.category != ConsistencyWarning]

            assert len(non_consistency_warnings) == 2
            assert "new_vehicle_types" in str(non_consistency_warnings[0].message)
            assert "vehicle_type_conversion" in str(non_consistency_warnings[1].message)

        # Check that exactly 3 new vehicle types exist (EN, GN, DD)
        vehicle_types = db_session.query(VehicleType).all()
        assert len(vehicle_types) == 3

        vt_names = {vt.name_short for vt in vehicle_types}
        assert vt_names == {"EN", "GN", "DD"}

        # Check that rotations were updated to use new vehicle types
        rotations = db_session.query(Rotation).all()
        assert len(rotations) == 4  # Unused rotation should be removed

        # Check that all rotations use one of the new vehicle types
        for rotation in rotations:
            assert rotation.vehicle_type.name_short in {"EN", "GN", "DD"}

        # Verify specific conversions based on defaults
        gn_rotations = (
            db_session.query(Rotation)
            .join(VehicleType)
            .filter(VehicleType.name_short == "GN")
            .count()
        )
        en_rotations = (
            db_session.query(Rotation)
            .join(VehicleType)
            .filter(VehicleType.name_short == "EN")
            .count()
        )
        dd_rotations = (
            db_session.query(Rotation)
            .join(VehicleType)
            .filter(VehicleType.name_short == "DD")
            .count()
        )

        # Based on default mapping: GN->["GN", "GEG"], EN->["EN"], DD->["D"]
        assert gn_rotations == 2  # GN and GEG
        assert en_rotations == 1  # EN
        assert dd_rotations == 1  # D

    def test_remove_unused_vehicle_types_with_custom_params(
        self, temp_db: Path, scenario_with_vehicle_types
    ):
        """Test RemoveUnusedVehicleTypes modifier with custom parameters."""
        modifier = RemoveUnusedVehicleTypes()

        # Create custom vehicle types
        scenario_id = scenario_with_vehicle_types.id
        custom_vt1 = VehicleType(
            name="Custom Bus Type 1",
            scenario_id=scenario_id,
            name_short="CB1",
            battery_capacity=500.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 400], [1, 400]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
        )
        custom_vt2 = VehicleType(
            name="Custom Bus Type 2",
            scenario_id=scenario_id,
            name_short="CB2",
            battery_capacity=600.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 450], [1, 450]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
        )

        # Custom conversion mapping
        custom_conversion = {
            "CB1": ["GN", "GEG", "EN"],  # Map articulated and single to CB1
            "CB2": ["D"],  # Map double decker to CB2
        }

        # Create pipeline context with custom parameters
        params = {
            "RemoveUnusedVehicleTypes.new_vehicle_types": [custom_vt1, custom_vt2],
            "RemoveUnusedVehicleTypes.vehicle_type_conversion": custom_conversion,
        }

        # Run modifier
        db_url = f"sqlite:///{temp_db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            modifier.modify(session=session, params=params)
            session.commit()

            # Verify results
            vehicle_types = session.query(VehicleType).all()
            assert len(vehicle_types) == 2

            vt_names = {vt.name_short for vt in vehicle_types}
            assert vt_names == {"CB1", "CB2"}

            # Check rotation counts
            cb1_rotations = (
                session.query(Rotation)
                .join(VehicleType)
                .filter(VehicleType.name_short == "CB1")
                .count()
            )
            cb2_rotations = (
                session.query(Rotation)
                .join(VehicleType)
                .filter(VehicleType.name_short == "CB2")
                .count()
            )

            assert cb1_rotations == 3  # GN, GEG, EN
            assert cb2_rotations == 1  # D

        finally:
            session.close()
            engine.dispose()

    def test_validation_duplicate_short_names(self, temp_db: Path, scenario_with_vehicle_types):
        """Test that duplicate name_short values in new vehicle types raise an error."""
        modifier = RemoveUnusedVehicleTypes()

        scenario_id = scenario_with_vehicle_types.id
        vt1 = VehicleType(
            name="Type 1",
            scenario_id=scenario_id,
            name_short="SAME",
            battery_capacity=500.0,
        )
        vt2 = VehicleType(
            name="Type 2",
            scenario_id=scenario_id,
            name_short="SAME",  # Duplicate!
            battery_capacity=500.0,
        )

        params = {
            "RemoveUnusedVehicleTypes.new_vehicle_types": [vt1, vt2],
            "RemoveUnusedVehicleTypes.vehicle_type_conversion": {"SAME": ["GN"]},
        }

        db_url = f"sqlite:///{temp_db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            with pytest.raises(ValueError, match="duplicate name_short values"):
                modifier.modify(session=session, params=params)
        finally:
            session.close()
            engine.dispose()

    def test_validation_conversion_keys_mismatch(self, temp_db: Path, scenario_with_vehicle_types):
        """Test that mismatched conversion keys and new vehicle types raise an error."""
        modifier = RemoveUnusedVehicleTypes()

        scenario_id = scenario_with_vehicle_types.id
        vt1 = VehicleType(
            name="Type 1",
            scenario_id=scenario_id,
            name_short="VT1",
            battery_capacity=500.0,
        )

        # Conversion mapping has different key than vehicle type short name
        params = {
            "RemoveUnusedVehicleTypes.new_vehicle_types": [vt1],
            "RemoveUnusedVehicleTypes.vehicle_type_conversion": {"WRONG_KEY": ["GN"]},
        }

        db_url = f"sqlite:///{temp_db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            with pytest.raises(ValueError, match="Mismatch between new vehicle types"):
                modifier.modify(session=session, params=params)
        finally:
            session.close()
            engine.dispose()

    def test_validation_conflicting_names(self, temp_db: Path, scenario_with_vehicle_types):
        """Test that conflicting new vehicle type names raise an error."""
        modifier = RemoveUnusedVehicleTypes()

        # Add a vehicle type to the database that won't be converted
        db_url = f"sqlite:///{temp_db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            scenario_id = scenario_with_vehicle_types.id

            # Add a vehicle type that we're NOT converting
            vt_stays = VehicleType(
                name="Stays",
                scenario_id=scenario_id,
                name_short="STAYS",
                battery_capacity=500.0,
                opportunity_charging_capable=False,
            )
            session.add(vt_stays)
            session.commit()

            # Try to create a new vehicle type with the same name
            vt_conflict = VehicleType(
                name="Conflict",
                scenario_id=scenario_id,
                name_short="STAYS",  # Conflicts!
                battery_capacity=500.0,
                opportunity_charging_capable=False,
            )

            params = {
                "RemoveUnusedVehicleTypes.new_vehicle_types": [vt_conflict],
                "RemoveUnusedVehicleTypes.vehicle_type_conversion": {
                    "STAYS": ["GN"]  # Trying to map GN to STAYS, but STAYS already exists
                },
            }

            with pytest.raises(ValueError, match="conflict with existing types"):
                modifier.modify(session=session, params=params)

        finally:
            session.close()
            engine.dispose()

    def test_document_params(self):
        """Test that document_params returns expected parameter documentation."""
        modifier = RemoveUnusedVehicleTypes()
        docs = modifier.document_params()

        assert "RemoveUnusedVehicleTypes.new_vehicle_types" in docs
        assert "RemoveUnusedVehicleTypes.vehicle_type_conversion" in docs
        assert "VehicleType" in docs["RemoveUnusedVehicleTypes.new_vehicle_types"]
        assert "Dict[str, List[str]]" in docs["RemoveUnusedVehicleTypes.vehicle_type_conversion"]
