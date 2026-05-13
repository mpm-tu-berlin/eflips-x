"""Tests for GTFS utility modifiers."""

from pathlib import Path

import eflips.model
import pytest
from eflips.model import (
    ConsumptionLut as DbConsumptionLut,
    Rotation,
    Trip,
    VehicleClass,
    VehicleType,
)
from sqlalchemy.orm import Session

from eflips.x.steps.modifiers.consumption_luts import (
    CONSUMPTION_LUT_DIR,
    ConsumptionLut,
)
from eflips.x.steps.modifiers.gtfs_utilities import ConfigureVehicleTypes


@pytest.fixture
def db_session(writable_bvg_scenario_db: Path):
    db_url = f"sqlite:///{writable_bvg_scenario_db.absolute().as_posix()}"
    engine = eflips.model.create_engine(db_url)
    session = Session(engine)
    yield session
    session.close()
    engine.dispose()


class TestConfigureVehicleTypes:

    @pytest.fixture
    def modifier(self) -> ConfigureVehicleTypes:
        return ConfigureVehicleTypes()

    def test_single_vt_scalar_params(self, modifier: ConfigureVehicleTypes, db_session: Session):
        modifier.modify(
            db_session,
            {
                "ConfigureVehicleTypes.vehicle_type_names": ["default"],
                "ConfigureVehicleTypes.battery_capacity": 500.0,
                "ConfigureVehicleTypes.consumption": 2.0,
            },
        )
        vts = db_session.query(VehicleType).all()
        assert len(vts) == 1
        assert vts[0].name == "default"
        assert vts[0].battery_capacity == 500.0
        assert vts[0].consumption == 2.0
        for rot in db_session.query(Rotation).all():
            assert rot.vehicle_type_id == vts[0].id
            assert rot.allow_opportunity_charging is True

    def test_multi_vt_dict_params_default_assignment(
        self, modifier: ConfigureVehicleTypes, db_session: Session
    ):
        modifier.modify(
            db_session,
            {
                "ConfigureVehicleTypes.vehicle_type_names": ["A", "B"],
                "ConfigureVehicleTypes.battery_capacity": {"A": 240.0, "B": 360.0},
                "ConfigureVehicleTypes.consumption": {"A": 1.3, "B": 1.7},
                # charging_curve broadcast to both
            },
        )
        by_name = {vt.name: vt for vt in db_session.query(VehicleType).all()}
        assert set(by_name) == {"A", "B"}
        assert by_name["A"].battery_capacity == 240.0
        assert by_name["B"].battery_capacity == 360.0
        # Without trip overrides, every rotation goes to first VT ("A").
        for rot in db_session.query(Rotation).all():
            assert rot.vehicle_type_id == by_name["A"].id

    def test_dict_with_wrong_keys_raises(
        self, modifier: ConfigureVehicleTypes, db_session: Session
    ):
        with pytest.raises(ValueError, match="must equal vehicle_type_names"):
            modifier.modify(
                db_session,
                {
                    "ConfigureVehicleTypes.vehicle_type_names": ["A", "B"],
                    "ConfigureVehicleTypes.battery_capacity": {"A": 240.0},
                },
            )

    def test_missing_vehicle_type_names_raises(
        self, modifier: ConfigureVehicleTypes, db_session: Session
    ):
        with pytest.raises(ValueError, match="vehicle_type_names is required"):
            modifier.modify(db_session, {})

    def test_trip_to_vehicle_type_with_lut(
        self, modifier: ConfigureVehicleTypes, db_session: Session
    ):
        # Pick a trip from a multi-trip rotation so we exercise the
        # rotation-split path.
        target_trip = None
        for t in db_session.query(Trip).all():
            if len(t.rotation.trips) > 1:
                target_trip = t
                break
        assert target_trip is not None, "Need a multi-trip rotation for this test"
        target_trip_id = target_trip.id
        original_rotation_id = target_trip.rotation_id

        modifier.modify(
            db_session,
            {
                "ConfigureVehicleTypes.vehicle_type_names": ["small", "big"],
                "ConfigureVehicleTypes.battery_capacity": {"small": 200.0, "big": 400.0},
                "ConfigureVehicleTypes.consumption": {
                    "small": 1.2,
                    "big": ConsumptionLut.NOR_BUS_12M,
                },
                "ConfigureVehicleTypes.empty_mass": {"small": 8000.0, "big": 13000.0},
                "ConfigureVehicleTypes.allowed_mass": {
                    "small": 4000.0,
                    "big": 6000.0,
                },
                "ConfigureVehicleTypes.trip_to_vehicle_type": {target_trip_id: "big"},
            },
        )

        by_name = {vt.name: vt for vt in db_session.query(VehicleType).all()}
        assert set(by_name) == {"small", "big"}
        # LUT-backed VT must have NULL consumption.
        assert by_name["big"].consumption is None
        # And a linked VehicleClass + ConsumptionLut.
        assert len(by_name["big"].vehicle_classes) == 1
        vc = by_name["big"].vehicle_classes[0]
        assert vc.consumption_lut is not None
        assert isinstance(vc.consumption_lut, DbConsumptionLut)
        assert len(vc.consumption_lut.values) > 0
        # Scalar consumption VT keeps its float.
        assert by_name["small"].consumption == 1.2

        # Targeted trip moved to a fresh single-trip rotation pinned to "big".
        moved = db_session.get(Trip, target_trip_id)
        assert moved.rotation_id != original_rotation_id
        assert len(moved.rotation.trips) == 1
        assert moved.rotation.vehicle_type_id == by_name["big"].id

        # All other rotations point to the default VT ("small").
        for rot in db_session.query(Rotation).all():
            if rot.id == moved.rotation_id:
                continue
            assert rot.vehicle_type_id == by_name["small"].id

    def test_document_params_keys(self, modifier: ConfigureVehicleTypes):
        docs = modifier.document_params()
        assert "ConfigureVehicleTypes.vehicle_type_names" in docs
        assert "ConfigureVehicleTypes.trip_to_vehicle_type" in docs
        assert "ConfigureVehicleTypes.battery_capacity" in docs
        assert "ConfigureVehicleTypes.consumption" in docs
        assert "ConfigureVehicleTypes.charging_curve" in docs
        assert "ConfigureVehicleTypes.empty_mass" in docs


class TestConsumptionLutCsvs:

    @pytest.mark.parametrize("member", list(ConsumptionLut))
    def test_csv_present(self, member: ConsumptionLut):
        assert member.path.exists(), f"Missing LUT CSV: {member.path}"

    def test_lut_dir_contains_only_known_csvs(self):
        on_disk = {p.name for p in CONSUMPTION_LUT_DIR.glob("*.csv")}
        expected = {m.value for m in ConsumptionLut}
        assert on_disk == expected
