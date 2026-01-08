"""Tests for CopyCreator generator step."""

from pathlib import Path

import eflips.model
import pytest
from eflips.model import Scenario, Base
from sqlalchemy.orm import Session

from eflips.x.framework import PipelineContext
from eflips.x.steps.generators import CopyCreator


class TestCopyCreator:
    """Test suite for CopyCreator."""

    @pytest.fixture
    def source_db(self, tmp_path: Path) -> Path:
        """Create a simple source database for testing."""
        db_path = tmp_path / "source.db"
        db_url = f"sqlite:////{db_path.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        Base.metadata.create_all(engine)

        # Add a simple scenario to the database
        session = Session(engine)
        scenario = Scenario(name="Test Scenario")
        session.add(scenario)
        session.commit()
        session.close()
        engine.dispose()

        return db_path

    @pytest.fixture
    def copy_creator(self, source_db: Path) -> CopyCreator:
        """Create a CopyCreator instance with test database."""
        return CopyCreator(input_files=[source_db], cache_enabled=False)

    def test_init(self, source_db: Path):
        """Test that CopyCreator initializes correctly."""
        creator = CopyCreator(input_files=[source_db])
        assert len(creator.input_files) == 1
        assert isinstance(creator.input_files[0], Path)
        assert creator.input_files[0] == source_db

    def test_init_requires_single_file(self, source_db: Path, tmp_path: Path):
        """Test that CopyCreator requires exactly one database file."""
        # Create a second database file
        second_db = tmp_path / "second.db"
        second_db.touch()

        # Should fail with multiple files
        with pytest.raises(ValueError, match="exactly one database file"):
            CopyCreator(input_files=[source_db, second_db])

        # Should fail with no files
        with pytest.raises(ValueError, match="exactly one database file"):
            CopyCreator(input_files=[])

    def test_init_requires_existing_file(self, tmp_path: Path):
        """Test that CopyCreator requires existing files."""
        nonexistent_file = tmp_path / "nonexistent.db"
        with pytest.raises(ValueError, match="do not exist"):
            CopyCreator(input_files=[nonexistent_file])

    def test_document_params(self):
        """Test that document_params returns empty dict."""
        params = CopyCreator.document_params()
        assert isinstance(params, dict)
        assert len(params) == 0

    def test_execute_copies_database(self, source_db: Path, tmp_path: Path):
        """Test that execute copies the database correctly."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        context = PipelineContext(work_dir=work_dir, params={})
        creator = CopyCreator(input_files=[source_db], cache_enabled=False)

        creator.execute(context=context)

        # Check that the database was copied
        assert context.current_db is not None
        assert context.current_db.exists()
        assert context.current_db.name == "step_001_CopyCreator.db"

        # Verify the database has the same content
        db_url = f"sqlite:////{context.current_db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        scenarios = session.query(Scenario).all()
        assert len(scenarios) == 1
        assert scenarios[0].name == "Test Scenario"

        session.close()
        engine.dispose()

    def test_generate_raises_not_implemented(self, copy_creator: CopyCreator):
        """Test that generate() raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="should never be called"):
            copy_creator.generate(session=None, params={})  # type: ignore
