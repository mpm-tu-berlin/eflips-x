"""Shared fixtures for all steps tests (modifiers + analyzers)."""

import shutil
from pathlib import Path
from typing import Generator

import eflips.model
import pytest
from eflips.model import Base
from sqlalchemy.orm import Session

from tests.util import multi_depot_scenario


@pytest.fixture(scope="module")
def small_multi_depot_db(ors_cache, tmp_path_factory) -> Generator[Path, None, None]:
    """2 depots, 4 lines/depot, 10 trips/line — created once per test module.

    Treat as read-only.  Use writable_scenario_db for tests that need to modify data.
    """
    db_path = tmp_path_factory.mktemp("mds") / "multi_depot.db"
    db_url = f"sqlite:///{db_path.absolute().as_posix()}"
    engine = eflips.model.create_engine(db_url)
    Base.metadata.create_all(engine)
    session = Session(engine)
    multi_depot_scenario(session, num_depots=2, lines_per_depot=4, trips_per_line=10)
    session.close()
    engine.dispose()
    yield db_path


@pytest.fixture
def small_scenario_session(small_multi_depot_db: Path) -> Generator[Session, None, None]:
    """Per-test read-only session into the shared multi-depot DB, rolled back after each test."""
    db_url = f"sqlite:///{small_multi_depot_db.absolute().as_posix()}"
    engine = eflips.model.create_engine(db_url)
    session = Session(engine)
    yield session
    session.rollback()
    session.close()
    engine.dispose()


@pytest.fixture
def writable_scenario_db(small_multi_depot_db: Path, tmp_path: Path) -> Path:
    """Per-test writable copy of the shared multi-depot DB (cheap file copy)."""
    db_copy = tmp_path / "writable.db"
    shutil.copy2(small_multi_depot_db, db_copy)
    return db_copy
