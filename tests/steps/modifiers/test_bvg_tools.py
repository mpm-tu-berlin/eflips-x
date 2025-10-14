"""Tests for BVG-specific modifier tools."""

import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import eflips.model
import numpy as np
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
    ConsumptionLut,
    VehicleClass,
    AssocVehicleTypeVehicleClass,
)
from geoalchemy2.elements import WKTElement
from sqlalchemy.orm import Session

from eflips.x.framework import PipelineContext
from eflips.x.steps.modifiers.bvg_tools import (
    RemoveUnusedVehicleTypes,
    RemoveUnusedRotations,
    MergeStations,
    ReduceToNDaysNDepots,
)


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

    def test_remove_unused_vehicle_types_with_multiplier(
        self, temp_db: Path, scenario_with_vehicle_types, db_session: Session
    ):
        """Test RemoveUnusedVehicleTypes modifier using default parameters."""
        modifier = RemoveUnusedVehicleTypes()

        # Create pipeline context
        context = PipelineContext(work_dir=temp_db.parent, current_db=temp_db)

        params = {
            "RemoveUnusedVehicleTypes.override_consumption_lut": {
                "GN": 2.0,
            }
        }

        # Run modifier
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            modifier.modify(
                session=db_session,
                params=params,
            )

            # Check that warnings were emitted for using defaults
            non_consistency_warnings = [wa for wa in w if wa.category != ConsistencyWarning]

            assert len(non_consistency_warnings) == 2
            assert "new_vehicle_types" in str(non_consistency_warnings[0].message)
            assert "vehicle_type_conversion" in str(non_consistency_warnings[1].message)

        # Cehck that the new "GN" short name vehicle type has a consumption lut
        consumption_lut = (
            db_session.query(ConsumptionLut)
            .join(VehicleClass)
            .join(AssocVehicleTypeVehicleClass)
            .join(VehicleType)
            .filter(VehicleType.name_short == "GN")
            .one()
        )

        assert np.isclose(max(consumption_lut.values), 5.332117566234736)

    def test_remove_unused_vehicle_types_with_path(
        self, temp_db: Path, scenario_with_vehicle_types, db_session: Session
    ):
        """Test RemoveUnusedVehicleTypes modifier using default parameters."""
        modifier = RemoveUnusedVehicleTypes()

        # Create pipeline context
        context = PipelineContext(work_dir=temp_db.parent, current_db=temp_db)
        path_to_this_file = Path(__file__).absolute()
        project_root = path_to_this_file.parents[3]
        consumption_lut_path = project_root / "data" / "input" / "consumption_lut_gn.xlsx"

        params = {
            "RemoveUnusedVehicleTypes.override_consumption_lut": {
                "GN": consumption_lut_path.as_posix(),
            }
        }

        # Run modifier
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            modifier.modify(
                session=db_session,
                params=params,
            )

            # Check that warnings were emitted for using defaults
            non_consistency_warnings = [wa for wa in w if wa.category != ConsistencyWarning]

            assert len(non_consistency_warnings) == 2
            assert "new_vehicle_types" in str(non_consistency_warnings[0].message)
            assert "vehicle_type_conversion" in str(non_consistency_warnings[1].message)

        # Cehck that the new "GN" short name vehicle type has a consumption lut
        consumption_lut = (
            db_session.query(ConsumptionLut)
            .join(VehicleClass)
            .join(AssocVehicleTypeVehicleClass)
            .join(VehicleType)
            .filter(VehicleType.name_short == "GN")
            .one()
        )

        assert np.isclose(max(consumption_lut.values), 3.593211767009342)

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
            empty_mass=1000,
            allowed_mass=2000,
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
            empty_mass=2000,
            allowed_mass=4000,
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


class TestRemoveUnusedRotations:
    """Test suite for RemoveUnusedRotations modifier."""

    @pytest.fixture
    def scenario_with_rotations(self, db_session: Session) -> Scenario:
        """Create a test scenario with rotations starting/ending at different stations."""
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.flush()

        # Create stations
        depot1 = Station(
            name="Betriebshof Lichtenberg",
            name_short="BF L",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        depot2 = Station(
            name="Betriebshof Marzahn",
            name_short="BF M",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        other_station = Station(
            name="Other Station",
            name_short="OTHER",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        intermediate_station = Station(
            name="Intermediate Station",
            name_short="INTER",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        db_session.add_all([depot1, depot2, other_station, intermediate_station])
        db_session.flush()

        # Create a vehicle type
        vt = VehicleType(
            name="Test Bus",
            scenario_id=scenario.id,
            name_short="TB",
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 300], [1, 300]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
        )
        db_session.add(vt)
        db_session.flush()

        # Create routes
        route_depot1_inter = Route(
            name="Route Depot1 to Inter",
            name_short="D1-I",
            scenario_id=scenario.id,
            departure_station=depot1,
            arrival_station=intermediate_station,
            distance=5000,
        )
        route_inter_depot1 = Route(
            name="Route Inter to Depot1",
            name_short="I-D1",
            scenario_id=scenario.id,
            departure_station=intermediate_station,
            arrival_station=depot1,
            distance=5000,
        )
        route_depot2_inter = Route(
            name="Route Depot2 to Inter",
            name_short="D2-I",
            scenario_id=scenario.id,
            departure_station=depot2,
            arrival_station=intermediate_station,
            distance=5000,
        )
        route_inter_depot2 = Route(
            name="Route Inter to Depot2",
            name_short="I-D2",
            scenario_id=scenario.id,
            departure_station=intermediate_station,
            arrival_station=depot2,
            distance=5000,
        )
        route_other_inter = Route(
            name="Route Other to Inter",
            name_short="O-I",
            scenario_id=scenario.id,
            departure_station=other_station,
            arrival_station=intermediate_station,
            distance=5000,
        )
        route_inter_other = Route(
            name="Route Inter to Other",
            name_short="I-O",
            scenario_id=scenario.id,
            departure_station=intermediate_station,
            arrival_station=other_station,
            distance=5000,
        )
        db_session.add_all(
            [
                route_depot1_inter,
                route_inter_depot1,
                route_depot2_inter,
                route_inter_depot2,
                route_other_inter,
                route_inter_other,
            ]
        )
        db_session.flush()

        # Rotation 1: Depot1 -> Inter -> Depot1 (should be kept)
        rotation1 = Rotation(
            name="Rotation Valid Depot1",
            scenario_id=scenario.id,
            vehicle_type=vt,
            allow_opportunity_charging=False,
        )
        db_session.add(rotation1)
        db_session.flush()
        trip1_1 = Trip(
            rotation=rotation1,
            route=route_depot1_inter,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 8, 0, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 8, 30, tzinfo=ZoneInfo("UTC")),
        )
        trip1_2 = Trip(
            rotation=rotation1,
            route=route_inter_depot1,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 8, 45, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 9, 15, tzinfo=ZoneInfo("UTC")),
        )
        db_session.add_all([trip1_1, trip1_2])

        # Rotation 2: Depot2 -> Inter -> Depot2 (should be kept)
        rotation2 = Rotation(
            name="Rotation Valid Depot2",
            scenario_id=scenario.id,
            vehicle_type=vt,
            allow_opportunity_charging=False,
        )
        db_session.add(rotation2)
        db_session.flush()
        trip2_1 = Trip(
            rotation=rotation2,
            route=route_depot2_inter,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 9, 0, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 9, 30, tzinfo=ZoneInfo("UTC")),
        )
        trip2_2 = Trip(
            rotation=rotation2,
            route=route_inter_depot2,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 9, 45, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 10, 15, tzinfo=ZoneInfo("UTC")),
        )
        db_session.add_all([trip2_1, trip2_2])

        # Rotation 3: Other -> Inter -> Other (should be removed - not a depot)
        rotation3 = Rotation(
            name="Rotation Invalid Other",
            scenario_id=scenario.id,
            vehicle_type=vt,
            allow_opportunity_charging=False,
        )
        db_session.add(rotation3)
        db_session.flush()
        trip3_1 = Trip(
            rotation=rotation3,
            route=route_other_inter,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 10, 0, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 10, 30, tzinfo=ZoneInfo("UTC")),
        )
        trip3_2 = Trip(
            rotation=rotation3,
            route=route_inter_other,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 10, 45, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 11, 15, tzinfo=ZoneInfo("UTC")),
        )
        db_session.add_all([trip3_1, trip3_2])

        # Rotation 4: Depot1 -> Inter -> Depot2 (should be removed - starts and ends at different depots)
        rotation4 = Rotation(
            name="Rotation Invalid Different Depots",
            scenario_id=scenario.id,
            vehicle_type=vt,
            allow_opportunity_charging=False,
        )
        db_session.add(rotation4)
        db_session.flush()
        trip4_1 = Trip(
            rotation=rotation4,
            route=route_depot1_inter,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 11, 0, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 11, 30, tzinfo=ZoneInfo("UTC")),
        )
        trip4_2 = Trip(
            rotation=rotation4,
            route=route_inter_depot2,
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 11, 45, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 12, 15, tzinfo=ZoneInfo("UTC")),
        )
        db_session.add_all([trip4_1, trip4_2])

        db_session.commit()
        return scenario

    def test_remove_unused_rotations_with_defaults(
        self, temp_db: Path, scenario_with_rotations, db_session: Session
    ):
        """Test RemoveUnusedRotations modifier using default parameters."""
        modifier = RemoveUnusedRotations()

        # Run modifier with default BVG depot names (includes "BF L" and "BF M")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            modifier.modify(session=db_session, params={})

            # Check that warning was emitted for using defaults
            non_consistency_warnings = [wa for wa in w if wa.category != ConsistencyWarning]
            assert len(non_consistency_warnings) == 1
            assert "depot_station_short_names" in str(non_consistency_warnings[0].message)

        # Default depot names include "BF L" and "BF M", so 2 valid rotations should remain
        rotations = db_session.query(Rotation).all()
        assert len(rotations) == 2

        # Verify the kept rotations are the valid ones
        rotation_names = {rot.name for rot in rotations}
        assert rotation_names == {"Rotation Valid Depot1", "Rotation Valid Depot2"}

    def test_remove_unused_rotations_with_custom_params(
        self, temp_db: Path, scenario_with_rotations, db_session: Session
    ):
        """Test RemoveUnusedRotations modifier with custom depot station names."""
        modifier = RemoveUnusedRotations()

        # Use custom depot names that match our test depots
        params = {"RemoveUnusedRotations.depot_station_short_names": ["BF L", "BF M"]}

        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Check that only valid rotations remain (2 rotations starting/ending at the same depot)
        rotations = db_session.query(Rotation).all()
        assert len(rotations) == 2

        # Verify the kept rotations are the valid ones
        rotation_names = {rot.name for rot in rotations}
        assert rotation_names == {"Rotation Valid Depot1", "Rotation Valid Depot2"}

        # Verify trips were properly deleted for removed rotations
        all_trips = db_session.query(Trip).all()
        assert len(all_trips) == 4  # 2 trips per valid rotation

    def test_remove_unused_rotations_keeps_single_depot(
        self, temp_db: Path, scenario_with_rotations, db_session: Session
    ):
        """Test that only rotations from a single depot can be kept."""
        modifier = RemoveUnusedRotations()

        # Only keep depot1
        params = {"RemoveUnusedRotations.depot_station_short_names": ["BF L"]}

        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Check that only depot1 rotation remains
        rotations = db_session.query(Rotation).all()
        assert len(rotations) == 1
        assert rotations[0].name == "Rotation Valid Depot1"

    def test_remove_unused_rotations_empty_depot_list_error(
        self, temp_db: Path, scenario_with_rotations, db_session: Session
    ):
        """Test that empty depot list raises ValueError."""
        modifier = RemoveUnusedRotations()

        params = {"RemoveUnusedRotations.depot_station_short_names": []}

        with pytest.raises(ValueError, match="cannot be empty"):
            modifier.modify(session=db_session, params=params)

    def test_remove_unused_rotations_nonexistent_depot(
        self, temp_db: Path, scenario_with_rotations, db_session: Session
    ):
        """Test behavior when depot station names don't exist in database."""
        modifier = RemoveUnusedRotations()

        params = {"RemoveUnusedRotations.depot_station_short_names": ["NONEXISTENT"]}

        # Should not raise error, but should remove all rotations
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        rotations = db_session.query(Rotation).all()
        assert len(rotations) == 0

    def test_document_params(self):
        """Test that document_params returns expected parameter documentation."""
        modifier = RemoveUnusedRotations()
        docs = modifier.document_params()

        assert "RemoveUnusedRotations.depot_station_short_names" in docs
        assert "List[str]" in docs["RemoveUnusedRotations.depot_station_short_names"]
        assert "BVG" in docs["RemoveUnusedRotations.depot_station_short_names"]


class TestMergeStations:
    """Test suite for MergeStations modifier."""

    @pytest.fixture
    def scenario_with_nearby_stations(self, db_session: Session) -> Scenario:
        """Create a test scenario covering all three merging test cases."""
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.flush()

        # Test Case 1: Nearby stations with similar names (SHOULD merge)
        station_nearby_similar_1 = Station(
            name="Berlin Hauptbahnhof",
            name_short="HBF1",
            scenario_id=scenario.id,
            geom=WKTElement("POINT(0 0)", srid=4326),
            is_electrified=False,
        )
        station_nearby_similar_2 = Station(
            name="S+U Berlin Hauptbahnhof",
            name_short="HBF2",
            scenario_id=scenario.id,
            geom=WKTElement("POINT(0.0005 0)", srid=4326),  # ~50m away
            is_electrified=False,
        )

        # Test Case 2: Far away stations with similar names (should NOT merge)
        station_far_similar = Station(
            name="Berlin Hbf Platform 2",
            name_short="HBF3",
            scenario_id=scenario.id,
            geom=WKTElement("POINT(1 1)", srid=4326),  # ~157km away
            is_electrified=False,
        )

        # Test Case 3: Nearby stations with different names (should NOT merge)
        station_nearby_different = Station(
            name="Zoologischer Garten",
            name_short="ZOO",
            scenario_id=scenario.id,
            geom=WKTElement("POINT(0.0005 0.0005)", srid=4326),  # ~70m away, different name
            is_electrified=False,
        )

        # Destination station for routes
        station_destination = Station(
            name="Alexanderplatz",
            name_short="ALEX",
            scenario_id=scenario.id,
            geom=WKTElement("POINT(0.01 0.01)", srid=4326),
            is_electrified=False,
        )

        db_session.add_all(
            [
                station_nearby_similar_1,
                station_nearby_similar_2,
                station_far_similar,
                station_nearby_different,
                station_destination,
            ]
        )
        db_session.flush()

        # Create a vehicle type
        vt = VehicleType(
            name="Test Bus",
            scenario_id=scenario.id,
            name_short="TB",
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 300], [1, 300]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
        )
        db_session.add(vt)
        db_session.flush()

        # Create routes using all stations so they're considered for merging
        routes = [
            Route(
                name=f"Route from {station.name_short}",
                name_short=f"R{i}",
                scenario_id=scenario.id,
                departure_station=station,
                arrival_station=station_destination,
                distance=5000,
            )
            for i, station in enumerate(
                [
                    station_nearby_similar_1,
                    station_nearby_similar_2,
                    station_far_similar,
                    station_nearby_different,
                ]
            )
        ]
        db_session.add_all(routes)
        db_session.flush()

        # Create a rotation with a trip
        rotation = Rotation(
            name="Test Rotation",
            scenario_id=scenario.id,
            vehicle_type=vt,
            allow_opportunity_charging=False,
        )
        db_session.add(rotation)
        db_session.flush()

        trip = Trip(
            rotation=rotation,
            route=routes[0],
            scenario_id=scenario.id,
            trip_type=TripType.PASSENGER,
            departure_time=datetime(2024, 1, 1, 8, 0, tzinfo=ZoneInfo("UTC")),
            arrival_time=datetime(2024, 1, 1, 8, 30, tzinfo=ZoneInfo("UTC")),
        )
        db_session.add(trip)

        db_session.commit()
        return scenario

    def test_merge_stations_with_defaults(
        self, temp_db: Path, scenario_with_nearby_stations, db_session: Session
    ):
        """Test MergeStations modifier covering all three test cases."""
        modifier = MergeStations()

        # Count initial stations
        initial_station_count = db_session.query(Station).count()
        assert initial_station_count == 5  # 4 test stations + 1 destination

        # Run modifier with defaults (100m distance, 80% match)
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # After merging: 4 stations should remain
        # 1. Berlin Hauptbahnhof (merged with S+U Berlin Hauptbahnhof)
        # 2. Berlin Hbf Platform 2 (NOT merged - far away despite similar name)
        # 3. Zoologischer Garten (NOT merged - nearby but different name)
        # 4. Alexanderplatz (destination)
        stations = db_session.query(Station).all()
        assert len(stations) == 4, f"Expected 4 stations after merging, got {len(stations)}"

        station_names = {s.name for s in stations}

        # Case 1: Nearby + Similar name -> SHOULD merge (keep shorter name)
        assert "Berlin Hauptbahnhof" in station_names
        assert "S+U Berlin Hauptbahnhof" not in station_names

        # Case 2: Far + Similar name -> should NOT merge
        assert "Berlin Hbf Platform 2" in station_names

        # Case 3: Nearby + Different name -> should NOT merge
        assert "Zoologischer Garten" in station_names

        # Destination should remain
        assert "Alexanderplatz" in station_names

    def test_merge_stations_with_stricter_params(
        self, temp_db: Path, scenario_with_nearby_stations, db_session: Session
    ):
        """Test MergeStations with stricter matching criteria."""
        modifier = MergeStations()

        # Use stricter criteria (smaller distance, higher percentage)
        params = {
            "MergeStations.max_distance_meters": 30.0,  # Only 30 meters
            "MergeStations.match_percentage": 90.0,  # 90% match required
        }

        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # With stricter criteria, no stations should be merged
        stations = db_session.query(Station).all()
        assert len(stations) == 5  # All original stations remain

    def test_merge_stations_updates_routes(
        self, temp_db: Path, scenario_with_nearby_stations, db_session: Session
    ):
        """Test that routes are properly updated to reference merged stations."""
        modifier = MergeStations()

        # Get initial route references
        route1 = db_session.query(Route).filter(Route.name_short == "R0").one()
        route2 = db_session.query(Route).filter(Route.name_short == "R1").one()
        initial_departure_1 = route1.departure_station_id
        initial_departure_2 = route2.departure_station_id

        # These should be different initially
        assert initial_departure_1 != initial_departure_2

        # Run merger
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # Refresh routes
        db_session.expire_all()
        route1 = db_session.query(Route).filter(Route.name_short == "R0").one()
        route2 = db_session.query(Route).filter(Route.name_short == "R1").one()

        # After merging, both routes should point to the same station
        assert route1.departure_station_id == route2.departure_station_id

    def test_merge_stations_validation_negative_distance(
        self, temp_db: Path, scenario_with_nearby_stations, db_session: Session
    ):
        """Test that negative distance raises ValueError."""
        modifier = MergeStations()

        params = {"MergeStations.max_distance_meters": -10.0}

        with pytest.raises(ValueError, match="must be positive"):
            modifier.modify(session=db_session, params=params)

    def test_merge_stations_validation_invalid_percentage(
        self, temp_db: Path, scenario_with_nearby_stations, db_session: Session
    ):
        """Test that invalid match percentage raises ValueError."""
        modifier = MergeStations()

        # Test percentage > 100
        params = {"MergeStations.match_percentage": 150.0}

        with pytest.raises(ValueError, match="between 0 and 100"):
            modifier.modify(session=db_session, params=params)

        # Test negative percentage
        params = {"MergeStations.match_percentage": -10.0}

        with pytest.raises(ValueError, match="between 0 and 100"):
            modifier.modify(session=db_session, params=params)

    def test_merge_stations_keeps_shortest_name(
        self, temp_db: Path, scenario_with_nearby_stations, db_session: Session
    ):
        """Test that the station with the shortest name is kept."""
        modifier = MergeStations()

        # Run merger
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # Get remaining stations
        stations = db_session.query(Station).all()
        station_names = {s.name for s in stations}

        # "Berlin Hauptbahnhof" (21 chars) should be kept over "S+U Berlin Hauptbahnhof" (24 chars)
        assert "Berlin Hauptbahnhof" in station_names
        assert "S+U Berlin Hauptbahnhof" not in station_names

    def test_document_params(self):
        """Test that document_params returns expected parameter documentation."""
        modifier = MergeStations()
        docs = modifier.document_params()

        assert "MergeStations.max_distance_meters" in docs
        assert "MergeStations.match_percentage" in docs
        assert "float" in docs["MergeStations.max_distance_meters"]
        assert "100.0" in docs["MergeStations.max_distance_meters"]
        assert "80.0" in docs["MergeStations.match_percentage"]


class TestReduceToNDaysNDepots:
    """Test suite for ReduceToNDaysNDepots modifier."""

    @pytest.fixture
    def scenario_with_multiple_days_depots(self, db_session: Session) -> Scenario:
        """Create a test scenario with rotations across multiple days and depots."""
        scenario = Scenario(name="Test Scenario", name_short="TEST")
        db_session.add(scenario)
        db_session.flush()

        # Create 3 depot stations
        depot1 = Station(
            name="Depot 1",
            name_short="D1",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        depot2 = Station(
            name="Depot 2",
            name_short="D2",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        depot3 = Station(
            name="Depot 3",
            name_short="D3",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        intermediate = Station(
            name="Intermediate",
            name_short="INT",
            scenario_id=scenario.id,
            geom=None,
            is_electrified=False,
        )
        db_session.add_all([depot1, depot2, depot3, intermediate])
        db_session.flush()

        # Create a vehicle type
        vt = VehicleType(
            name="Test Bus",
            scenario_id=scenario.id,
            name_short="TB",
            battery_capacity=400.0,
            battery_capacity_reserve=0.0,
            charging_curve=[[0, 300], [1, 300]],
            opportunity_charging_capable=True,
            minimum_charging_power=10,
        )
        db_session.add(vt)
        db_session.flush()

        # Create routes for each depot
        routes = {}
        for depot in [depot1, depot2, depot3]:
            routes[f"{depot.name_short}_out"] = Route(
                name=f"Route from {depot.name_short}",
                name_short=f"{depot.name_short}O",
                scenario_id=scenario.id,
                departure_station=depot,
                arrival_station=intermediate,
                distance=5000,
            )
            routes[f"{depot.name_short}_back"] = Route(
                name=f"Route to {depot.name_short}",
                name_short=f"{depot.name_short}B",
                scenario_id=scenario.id,
                departure_station=intermediate,
                arrival_station=depot,
                distance=5000,
            )
        db_session.add_all(routes.values())
        db_session.flush()

        # Create rotations across 3 days with different trip counts per day
        # Day 1 (2024-01-01): 10 trips (most trips - should be kept by default)
        # Day 2 (2024-01-02): 7 trips
        # Day 3 (2024-01-03): 3 trips (fewest trips)

        # Depot 1: 5 rotations (most rotations - should NOT be kept by default)
        # Depot 2: 2 rotations (fewest - should be kept by default)
        # Depot 3: 3 rotations (middle - should be kept by default)

        rotation_configs = [
            # (depot, day, num_rotations)
            (depot1, datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC")), 3),  # 6 trips
            (depot1, datetime(2024, 1, 2, tzinfo=ZoneInfo("UTC")), 2),  # 4 trips
            (depot2, datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC")), 2),  # 4 trips
            (depot2, datetime(2024, 1, 3, tzinfo=ZoneInfo("UTC")), 1),  # 2 trips
            (depot3, datetime(2024, 1, 2, tzinfo=ZoneInfo("UTC")), 1),  # 2 trips
            (depot3, datetime(2024, 1, 3, tzinfo=ZoneInfo("UTC")), 1),  # 2 trips
        ]

        rotation_counter = 0
        for depot, base_date, num_trips in rotation_configs:
            rotation = Rotation(
                name=f"Rotation {rotation_counter}",
                scenario_id=scenario.id,
                vehicle_type=vt,
                allow_opportunity_charging=False,
            )
            db_session.add(rotation)
            db_session.flush()

            # Add trips for this rotation
            for trip_idx in range(num_trips):
                trip_out = Trip(
                    rotation=rotation,
                    route=routes[f"{depot.name_short}_out"],
                    scenario_id=scenario.id,
                    trip_type=TripType.PASSENGER,
                    departure_time=base_date.replace(hour=8 + trip_idx * 2),
                    arrival_time=base_date.replace(hour=8 + trip_idx * 2, minute=30),
                )
                trip_back = Trip(
                    rotation=rotation,
                    route=routes[f"{depot.name_short}_back"],
                    scenario_id=scenario.id,
                    trip_type=TripType.PASSENGER,
                    departure_time=base_date.replace(hour=8 + trip_idx * 2, minute=45),
                    arrival_time=base_date.replace(hour=9 + trip_idx * 2),
                )
                db_session.add_all([trip_out, trip_back])

            rotation_counter += 1

        db_session.commit()
        return scenario

    def test_reduce_to_one_day_two_depots_defaults(
        self, temp_db: Path, scenario_with_multiple_days_depots, db_session: Session
    ):
        """Test ReduceToNDaysNDepots modifier with default parameters (1 day, 2 depots)."""
        modifier = ReduceToNDaysNDepots()

        # Count initial rotations
        initial_rotation_count = db_session.query(Rotation).count()
        assert initial_rotation_count == 6

        # Run modifier with defaults
        modifier.modify(session=db_session, params={})
        db_session.commit()

        rotations = db_session.query(Rotation).all()
        assert len(rotations) == 2

        # Verify it's from the correct day and depot
        kept_rotation = rotations[0]
        first_trip = kept_rotation.trips[0]
        assert first_trip.departure_time.date() == datetime(2024, 1, 1).date()
        assert first_trip.route.departure_station.name_short in ["D1", "D2"]

    def test_reduce_to_custom_days_depots(
        self, temp_db: Path, scenario_with_multiple_days_depots, db_session: Session
    ):
        """Test ReduceToNDaysNDepots modifier with custom parameters."""
        modifier = ReduceToNDaysNDepots()

        # Keep 2 days and 3 depots (all depots)
        params = {
            "ReduceToNDaysNDepots.num_days": 2,
            "ReduceToNDaysNDepots.num_depots": 3,
        }

        modifier.modify(session=db_session, params=params)
        db_session.commit()

        rotations = db_session.query(Rotation).all()
        assert len(rotations) == 4

    def test_reduce_to_single_depot(
        self, temp_db: Path, scenario_with_multiple_days_depots, db_session: Session
    ):
        """Test reducing to a single depot."""
        modifier = ReduceToNDaysNDepots()

        params = {
            "ReduceToNDaysNDepots.num_days": 3,  # Keep all days
            "ReduceToNDaysNDepots.num_depots": 1,  # Keep only 1 depot (fewest rotations)
        }

        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # Should keep only Depot 2 (2 rotations total) across all days
        rotations = db_session.query(Rotation).all()
        assert len(rotations) == 2

        # Verify all rotations are from Depot 2
        for rotation in rotations:
            if rotation.trips:
                assert rotation.trips[0].route.departure_station.name_short == "D1"

    def test_validation_negative_days(
        self, temp_db: Path, scenario_with_multiple_days_depots, db_session: Session
    ):
        """Test that negative num_days raises ValueError."""
        modifier = ReduceToNDaysNDepots()

        params = {"ReduceToNDaysNDepots.num_days": -1}

        with pytest.raises(ValueError, match="num_days must be positive"):
            modifier.modify(session=db_session, params=params)

    def test_validation_negative_depots(
        self, temp_db: Path, scenario_with_multiple_days_depots, db_session: Session
    ):
        """Test that negative num_depots raises ValueError."""
        modifier = ReduceToNDaysNDepots()

        params = {"ReduceToNDaysNDepots.num_depots": 0}

        with pytest.raises(ValueError, match="num_depots must be positive"):
            modifier.modify(session=db_session, params=params)

    def test_reduce_more_depots_than_available(
        self, temp_db: Path, scenario_with_multiple_days_depots, db_session: Session
    ):
        """Test requesting more depots than available."""
        modifier = ReduceToNDaysNDepots()

        params = {
            "ReduceToNDaysNDepots.num_days": 3,
            "ReduceToNDaysNDepots.num_depots": 10,  # More than the 3 available
        }

        # Should not raise error, just keep all depots
        modifier.modify(session=db_session, params=params)
        db_session.commit()

        # All 6 rotations should be kept
        rotations = db_session.query(Rotation).all()
        assert len(rotations) == 6

    def test_empty_scenario(self, db_session: Session):
        """Test behavior with a scenario that has no trips."""
        scenario = Scenario(name="Empty Scenario", name_short="EMPTY")
        db_session.add(scenario)
        db_session.commit()

        modifier = ReduceToNDaysNDepots()

        # Should not raise error, just log warning
        modifier.modify(session=db_session, params={})
        db_session.commit()

        # No rotations should exist
        rotations = db_session.query(Rotation).count()
        assert rotations == 0

    def test_document_params(self):
        """Test that document_params returns expected parameter documentation."""
        modifier = ReduceToNDaysNDepots()
        docs = modifier.document_params()

        assert "ReduceToNDaysNDepots.num_days" in docs
        assert "ReduceToNDaysNDepots.num_depots" in docs
        assert "int" in docs["ReduceToNDaysNDepots.num_days"]
        assert "int" in docs["ReduceToNDaysNDepots.num_depots"]
        assert "1" in docs["ReduceToNDaysNDepots.num_days"]
        assert "2" in docs["ReduceToNDaysNDepots.num_depots"]
