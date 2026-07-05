#!/usr/bin/env python3

"""
Impact analysis flow: run the eflips-impact analyzers against an existing database.

Given a :class:`PipelineContext` that already points at a simulated database and
carries the parameters every analyzer needs, this flow:

1. Runs :class:`TCOAnalyzer`, saving its result table (``.xlsx``) and stacked-bar
   plot (``.html``).
2. Runs :class:`LCAAnalyzer`, saving its result table (``.xlsx``) and stacked-bar
   plot (``.html``).

The :class:`PipelineContext` is the single input of the flow; it is constructed
in :func:`main`, where the database path, working directory and all JSON
parameter paths are initialized explicitly. Unlike ``example.py`` (which builds a
database from scratch) this flow only *reads* an existing database, mirroring how
``analysis_flow.py`` runs analyzers against a finished simulation.

Set the ``DATABASE_FILE`` / ``SCENARIO_NAME`` constants below, then run it::

    python -m eflips.x.flows.impact_flow
"""
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from prefect import flow

from eflips.x.flows.analysis_flow import save_visualization
from eflips.x.framework import Analyzer, PipelineContext
from eflips.x.steps.modifiers.general_utilities import (
    CompleteFleet,
    LCAConfigurator,
    TCOConfigurator,
)
from eflips.x.steps.analyzers import LCAAnalyzer, TCOAnalyzer

logger = logging.getLogger(__name__)

# Repository root: eflips/x/flows/impact_flow.py -> four levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_IMPACT_DIR = PROJECT_ROOT / "data" / "input" / "impact"

# Database file to analyze and the scenario display name. Edit these to point at
# the simulated database you want to run the impact analyzers against.
DATABASE_FILE = Path("/home/shuyao/remote_results/Simulation_ou.db")
SCENARIO_NAME = "OU"


def run_and_save_analyzer(
    analyzer: Analyzer,
    context: PipelineContext,
    output_dir: Path,
    name: str,
) -> Any:
    """
    Execute a single impact analyzer and persist its outputs.

    The analyzer is run against ``context`` (a copy of its database is made
    internally, so the source database is never modified). A tabular result is
    written to ``<name>.xlsx``. Plots are then saved depending on what the
    analyzer exposes:

    - if it defines a ``VISUALIZERS`` map (suffix -> visualize method name),
      each figure is written to ``<name>_<suffix>.html`` (e.g. the LCAAnalyzer
      produces ``<name>_by_type.html`` and ``<name>_by_scope.html``);
    - otherwise, if it provides a single ``visualize`` method, the figure is
      written to ``<name>.html``.

    Args:
        analyzer: The analyzer instance to execute.
        context: The pipeline context pointing at the database to analyze.
        output_dir: Directory where result table and plot are saved.
        name: Base file name (without extension) for the saved outputs.

    Returns:
        The raw result returned by ``analyzer.execute()``.
    """
    result = analyzer.execute(context=context)

    if isinstance(result, pd.DataFrame):
        table_file = output_dir / f"{name}.xlsx"
        result.to_excel(table_file, index=False)
        logger.info("Wrote %s result table to: %s", name, table_file)

    visualizers = getattr(analyzer, "VISUALIZERS", None)
    if visualizers:
        for suffix, method_name in visualizers.items():
            figure = getattr(analyzer, method_name)(result)
            save_visualization(figure, output_dir / f"{name}_{suffix}.html", analyzer)
    elif hasattr(analyzer, "visualize"):
        vis = analyzer.visualize(result)
        save_visualization(vis, output_dir / f"{name}.html", analyzer)

    return result


@flow(name="impact-flow")
def run_impact_analyzers(context: PipelineContext, output_dir: Path) -> Dict[str, Any]:
    """
    Run the eflips-impact analyzers against the database in *context*.

    All inputs (database path, working directory, JSON parameter paths and
    scenario name) must already be set on *context* by the caller -- see
    :func:`main` for how the context is constructed.

    Args:
        context: Pipeline context pointing at the database to analyze, with all
            analyzer parameters initialized.
        output_dir: Directory for the result tables and plots.

    Returns:
        Dict mapping analyzer key (``"tco"``, ``"lca"``) to its raw result.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # Write the fleet topology then the tco_parameters / lca_parameters into the
    # database (Modifiers). Each advances context.current_db to the configured
    # database that the analyzer immediately after it reads from.

    logger.info("Running CompleteFleet ...")
    CompleteFleet().execute(context=context)

    logger.info("Running TCOConfigurator ...")
    TCOConfigurator().execute(context=context)

    logger.info("Running TCOAnalyzer ...")
    results["tco"] = run_and_save_analyzer(TCOAnalyzer(), context, output_dir, "tco")

    logger.info("Running LCAConfigurator ...")
    LCAConfigurator().execute(context=context)

    logger.info("Running LCAAnalyzer ...")
    results["lca"] = run_and_save_analyzer(LCAAnalyzer(), context, output_dir, "lca")

    logger.info("Impact analysis complete. Outputs saved to: %s", output_dir)
    return results


def main() -> None:
    """
    Build the PipelineContext and run the impact flow.

    The database file and scenario name are taken from the ``DATABASE_FILE`` and
    ``SCENARIO_NAME`` module constants defined at the top of this file; edit them
    to analyze a different database.
    """
    logging.basicConfig(level=logging.INFO)

    database_file = DATABASE_FILE.resolve()
    if not database_file.exists():
        raise FileNotFoundError(f"Database file does not exist: {database_file}")

    output_dir = (
        PROJECT_ROOT / "data" / "output" / "transition_plan" / f"{database_file.stem}_impact"
    )

    # Initialize all paths explicitly when building the pipeline context.
    fleet_json = DEFAULT_IMPACT_DIR / "fleet.json"
    tco_json = DEFAULT_IMPACT_DIR / "tco.json"
    lca_json = DEFAULT_IMPACT_DIR / "lca.json"
    lca_overrides_json = DEFAULT_IMPACT_DIR / "lca_overrides.json"

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "CompleteFleet.fleet_json": str(fleet_json),
        "TCOConfigurator.tco_json": str(tco_json),
        "TCOAnalyzer.scenario_name": SCENARIO_NAME,
        "LCAConfigurator.lca_json": str(lca_json),
        "LCAConfigurator.lca_overrides_json": str(lca_overrides_json),
        "LCAAnalyzer.scenario_name": SCENARIO_NAME,
    }
    context = PipelineContext(
        work_dir=output_dir,
        params=params,
        current_db=database_file,
    )

    run_impact_analyzers(context=context, output_dir=output_dir)


if __name__ == "__main__":
    main()
