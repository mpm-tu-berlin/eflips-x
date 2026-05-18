"""Tests for GTFS utility modifiers."""

from pathlib import Path

import eflips.model
import pytest
from eflips.model import Rotation, VehicleType
from sqlalchemy.orm import Session

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

    def test_default_params_applied_to_all_vehicle_types(
        self, modifier: ConfigureVehicleTypes, db_session: Session
    ):
        modifier.modify(db_session, {})
        for vt in db_session.query(VehicleType).all():
            assert vt.battery_capacity == 360.0
            assert vt.consumption == 1.5
            assert vt.opportunity_charging_capable is True

    def test_custom_battery_capacity(self, modifier: ConfigureVehicleTypes, db_session: Session):
        modifier.modify(db_session, {"ConfigureVehicleTypes.battery_capacity": 500.0})
        for vt in db_session.query(VehicleType).all():
            assert vt.battery_capacity == 500.0

    def test_custom_consumption(self, modifier: ConfigureVehicleTypes, db_session: Session):
        modifier.modify(db_session, {"ConfigureVehicleTypes.consumption": 2.0})
        for vt in db_session.query(VehicleType).all():
            assert vt.consumption == 2.0

    def test_custom_charging_curve(self, modifier: ConfigureVehicleTypes, db_session: Session):
        curve = [[0.0, 200.0], [0.8, 200.0], [1.0, 50.0]]
        modifier.modify(db_session, {"ConfigureVehicleTypes.charging_curve": curve})
        for vt in db_session.query(VehicleType).all():
            assert vt.charging_curve == curve

    def test_all_rotations_get_opportunity_charging(
        self, modifier: ConfigureVehicleTypes, db_session: Session
    ):
        modifier.modify(db_session, {})
        for rotation in db_session.query(Rotation).all():
            assert rotation.allow_opportunity_charging is True

    def test_logs_warning_when_no_vehicle_types(
        self, modifier: ConfigureVehicleTypes, db_session: Session, caplog
    ):
        import logging

        db_session.query(Rotation).delete()
        db_session.query(VehicleType).delete()
        db_session.flush()
        with caplog.at_level(logging.WARNING):
            modifier.modify(db_session, {})
        assert any("No vehicle types" in r.message for r in caplog.records)

    def test_document_params_keys(self, modifier: ConfigureVehicleTypes):
        docs = modifier.document_params()
        assert isinstance(docs, dict)
        assert "ConfigureVehicleTypes.battery_capacity" in docs
        assert "ConfigureVehicleTypes.consumption" in docs
        assert "ConfigureVehicleTypes.charging_curve" in docs
