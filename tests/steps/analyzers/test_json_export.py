"""Tests for ScenarioJsonExporter."""

import json
from pathlib import Path

import pytest
from eflips.model import Scenario
from sqlalchemy.orm import Session

from eflips.x.steps.analyzers.json_export import ScenarioJsonExporter


@pytest.fixture
def db_session(small_scenario_session: Session) -> Session:
    return small_scenario_session


class TestScenarioJsonExporter:

    @pytest.fixture
    def analyzer(self) -> ScenarioJsonExporter:
        return ScenarioJsonExporter()

    def test_analyze_returns_dict(
        self, analyzer: ScenarioJsonExporter, db_session: Session, tmp_path: Path
    ):
        output_path = tmp_path / "export.json"
        result = analyzer.analyze(
            db_session, {"ScenarioJsonExporter.output_path": str(output_path)}
        )
        assert isinstance(result, dict)

    def test_output_file_written(
        self, analyzer: ScenarioJsonExporter, db_session: Session, tmp_path: Path
    ):
        output_path = tmp_path / "export.json"
        analyzer.analyze(db_session, {"ScenarioJsonExporter.output_path": str(output_path)})
        assert output_path.exists()
        with open(output_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_output_path_created_recursively(
        self, analyzer: ScenarioJsonExporter, db_session: Session, tmp_path: Path
    ):
        output_path = tmp_path / "deep" / "nested" / "dir" / "export.json"
        analyzer.analyze(db_session, {"ScenarioJsonExporter.output_path": str(output_path)})
        assert output_path.exists()

    def test_raises_when_no_scenario(
        self, analyzer: ScenarioJsonExporter, temp_db: Path, tmp_path: Path
    ):
        import eflips.model
        from eflips.model import Base
        from sqlalchemy.orm import Session as _Session

        db_url = f"sqlite:///{temp_db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        Base.metadata.create_all(engine)
        with _Session(engine) as empty_session:
            with pytest.raises(ValueError, match="No scenario found"):
                analyzer.analyze(
                    empty_session,
                    {"ScenarioJsonExporter.output_path": str(tmp_path / "out.json")},
                )
        engine.dispose()

    def test_document_params_has_output_path_key(self, analyzer: ScenarioJsonExporter):
        docs = analyzer.document_params()
        assert isinstance(docs, dict)
        assert "ScenarioJsonExporter.output_path" in docs

    def test_visualize_returns_none(self, analyzer: ScenarioJsonExporter):
        assert analyzer.visualize({}) is None
