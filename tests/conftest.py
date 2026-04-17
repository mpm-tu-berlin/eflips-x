"""Pytest configuration and fixtures for eflips-x tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator
from zipfile import ZipFile

import eflips.model
import pytest
from eflips.model import Base
from sqlalchemy.orm import Session


@pytest.fixture
def temp_db() -> Generator[Path, None, None]:
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Initialize the database
    db_url = f"sqlite:///{db_path.absolute().as_posix()}"
    engine = eflips.model.create_engine(db_url)
    Base.metadata.create_all(engine)
    engine.dispose()

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def db_session(temp_db: Path) -> Generator[Session, None, None]:
    """Create a database session for testing."""
    db_url = f"sqlite:///{temp_db.absolute().as_posix()}"
    engine = eflips.model.create_engine(db_url)
    session = Session(engine)

    yield session

    session.close()
    engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def ors_cache() -> None:
    """Extract ORS route-cache ZIP once per worker session and set DEPOT_ROTATION_MATCHING_ORS_CACHE.

    This replaces the per-class set_cache_directory autouse fixtures that previously lived in
    individual test files and were re-running the expensive ZIP extraction for every test.
    """
    if os.environ.get("DEPOT_ROTATION_MATCHING_ORS_CACHE") is not None:
        return
    path_to_cache_zip = (
        Path(__file__).resolve().parent / "steps" / "modifiers" / "depot_rotation_match_cache.zip"
    )
    if not path_to_cache_zip.exists():
        return
    # Per-worker extraction dir so pytest-xdist workers don't race on the same path.
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    worker_root = Path(tempfile.gettempdir()) / f"eflips_ors_cache_{worker}"
    target = worker_root / "DEPOT_ROTATION_MATCHING_ORS_CACHE"
    marker = worker_root / ".extracted"
    if not marker.exists():
        worker_root.mkdir(parents=True, exist_ok=True)
        with ZipFile(path_to_cache_zip, "r") as zip_ref:
            zip_ref.extractall(worker_root)
        marker.touch()
    os.environ["DEPOT_ROTATION_MATCHING_ORS_CACHE"] = str(target)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "data" / "input" / "Berlin Testing"


@pytest.fixture(scope="session")
def gtfs_test_data_dir() -> Path:
    """Return path to GTFS test data directory."""
    return Path(__file__).parent.parent / "data" / "input" / "GTFS"
