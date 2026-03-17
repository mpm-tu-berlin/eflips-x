"""
JSON export analyzer for eflips-x pipeline.

Exports a scenario to a JSON file using eflips-model's export utility.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import sqlalchemy.orm
from eflips.model import Scenario
from eflips.model.util.export_json import export_scenario_to_json

from eflips.x.framework import Analyzer

logger = logging.getLogger(__name__)


class ScenarioJsonExporter(Analyzer):
    """Analyzer that exports the first scenario in the database to a JSON file.

    This is read-only — the Analyzer framework creates a temporary DB copy,
    so inserting this step mid-pipeline does not affect ``context.current_db``.
    """

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            "ScenarioJsonExporter.output_path": (
                "Path where the exported JSON file will be written."
            ),
        }

    def analyze(self, session: sqlalchemy.orm.Session, params: Dict[str, Any]) -> Any:
        output_path = Path(params[f"{self.__class__.__name__}.output_path"])
        scenario = session.query(Scenario).first()
        if scenario is None:
            raise ValueError("No scenario found in database")

        json_data = export_scenario_to_json(scenario_id=scenario.id, session=session)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported scenario to {output_path}")
        return json_data

    def visualize(self, result: Any) -> None:
        pass
