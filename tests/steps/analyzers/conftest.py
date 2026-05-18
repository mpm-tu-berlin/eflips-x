"""Shared fixtures for analyzer tests."""

from pathlib import Path
from typing import Generator

import eflips.model
import pytest
from eflips.model import Base, ChargeType, Depot, VehicleType
from sqlalchemy.orm import Session

from eflips.x.steps.modifiers.scheduling import DepotAssignment, VehicleScheduling
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation
from tests.util import multi_depot_scenario


@pytest.fixture(scope="module")
def simulated_db(ors_cache, tmp_path_factory) -> Generator[tuple[Path, int], None, None]:
    """Full pipeline (scenario → scheduling → assignment → depot gen → simulation).

    Created once per test module and shared by all tests in that module via
    simulated_session (which opens a fresh rollback session for each test).

    Moved from the per-file _simulated_db fixture in test_output_analyzers.py so that
    test_bvg_tools.py can reuse the same expensive simulation result.
    """
    db_path = tmp_path_factory.mktemp("simulated") / "simulated.db"
    db_url = f"sqlite:///{db_path.absolute().as_posix()}"
    engine = eflips.model.create_engine(db_url)
    Base.metadata.create_all(engine)
    session = Session(engine)

    scenario = multi_depot_scenario(session, num_depots=2, lines_per_depot=4, trips_per_line=30)

    VehicleScheduling().modify(
        session,
        {
            "VehicleScheduling.charge_type": ChargeType.DEPOT,
            "VehicleScheduling.battery_margin": 0.1,
        },
    )

    all_depots = session.query(Depot).filter_by(scenario_id=scenario.id).all()
    all_vts = session.query(VehicleType).filter_by(scenario_id=scenario.id).all()
    depot_config = [
        {
            "depot_station": d.station_id,
            "capacity": 100,
            "vehicle_type": [vt.id for vt in all_vts],
            "name": d.name,
        }
        for d in all_depots
    ]
    DepotAssignment().modify(
        session,
        {
            "DepotAssignment.depot_config": depot_config,
            "DepotAssignment.depot_usage": 0.9,
            "DepotAssignment.step_size": 0.2,
            "DepotAssignment.max_iterations": 1,
        },
    )

    DepotGenerator().modify(session, {})
    Simulation().modify(session, {})

    session.commit()
    scenario_id = scenario.id
    session.close()
    engine.dispose()

    yield db_path, scenario_id

    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def simulated_session(simulated_db: tuple[Path, int]) -> Generator[Session, None, None]:
    """Per-test rollback session into the shared simulated DB."""
    db_path, _ = simulated_db
    db_url = f"sqlite:///{db_path.absolute().as_posix()}"
    engine = eflips.model.create_engine(db_url)
    session = Session(engine)
    yield session
    session.rollback()
    session.close()
    engine.dispose()


@pytest.fixture
def simulated_db_path(simulated_db: tuple[Path, int]) -> Path:
    """Path to the shared simulated DB file."""
    db_path, _ = simulated_db
    return db_path


@pytest.fixture
def simulated_scenario_id(simulated_db: tuple[Path, int]) -> int:
    """Scenario ID from the shared simulated DB."""
    _, scenario_id = simulated_db
    return scenario_id
