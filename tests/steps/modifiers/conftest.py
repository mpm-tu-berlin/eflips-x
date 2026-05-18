"""Shared fixtures for modifier tests."""

import shutil
from pathlib import Path
from typing import Generator

import eflips.model
import pytest
from eflips.model import Base, Rotation, Scenario, VehicleClass, VehicleType
from sqlalchemy.orm import Session

from eflips.x.steps.generators import BVGXMLIngester


@pytest.fixture(scope="module")
def bvg_xml_scenario_db(tmp_path_factory, test_data_dir: Path) -> Generator[Path, None, None]:
    """BVG XML ingested scenario with vehicle types — created once per test module.

    Replicates the setup previously performed by the per-test
    TestVehicleScheduling.scenario_with_vehicle_types fixture.
    Treat as read-only; use writable_bvg_scenario_db for tests that modify data.
    """
    xml_files = sorted(test_data_dir.glob("*.xml"))[:3]
    assert len(xml_files) > 0, f"No XML files found in {test_data_dir}"

    db_path = tmp_path_factory.mktemp("bvg_xml") / "bvg_scenario.db"
    db_url = f"sqlite:///{db_path.absolute().as_posix()}"
    engine = eflips.model.create_engine(db_url)
    Base.metadata.create_all(engine)
    session = Session(engine)

    BVGXMLIngester(input_files=xml_files, cache_enabled=False).generate(
        session,
        {"log_level": "WARNING", "BVGXMLIngester.multithreading": False},
    )

    scenario = session.query(Scenario).one()

    vehicle_class = VehicleClass(name="Standard Bus", scenario_id=scenario.id)
    session.add(vehicle_class)
    session.flush()

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
        consumption=1.2,
    )
    session.add(vehicle_type)
    session.flush()

    for rotation in session.query(Rotation).filter_by(scenario_id=scenario.id):
        rotation.vehicle_type_id = vehicle_type.id

    session.commit()
    session.close()
    engine.dispose()
    yield db_path


@pytest.fixture
def writable_bvg_scenario_db(bvg_xml_scenario_db: Path, tmp_path: Path) -> Path:
    """Per-test writable copy of the BVG XML scenario DB."""
    db_copy = tmp_path / "bvg_writable.db"
    shutil.copy2(bvg_xml_scenario_db, db_copy)
    return db_copy
