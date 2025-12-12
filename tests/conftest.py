"""Pytest configuration and fixtures for eflips-x tests."""

import tempfile
from pathlib import Path
from typing import Generator

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


@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "data" / "input" / "Berlin Testing"


@pytest.fixture
def gtfs_test_data_dir() -> Path:
    """Return path to GTFS test data directory."""
    return Path(__file__).parent.parent / "data" / "input" / "GTFS"
