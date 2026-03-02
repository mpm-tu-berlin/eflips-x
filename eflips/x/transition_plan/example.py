import importlib.util
import json
import os
from datetime import datetime
from typing import Dict, Any

import matplotlib

from eflips.x.framework import PipelineContext

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from eflips.x.transition_plan.multi_stage_simulation import simulate_multi_stage_electrification
from eflips.x.transition_plan.transition_plan import (
    run_transition_planner,
    run_tco_calculation,
    generate_transition_plan_plots,
)
from eflips.x.flows import run_steps

from eflips.x.transition_plan.transition_plan import PlaygroundAnalyzer


if __name__ == "__main__":

    data_dir = Path(__file__).parent.parent.parent.parent.parent / "eflips-data"
    db_name = "Simulation_ou_mini.db"
    input_db = data_dir / db_name

    FORCE_RERUN = True
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if FORCE_RERUN:
        work_dir = Path(__file__).parent.parent.parent / "transition_plan" / f"{run_id + db_name}"
    else:
        work_dir = Path(__file__).parent.parent.parent / "transition_plan" / db_name
    os.makedirs(work_dir, exist_ok=True)

    tco_params_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "TCO" / "berlin_literature.py"
    )

    transition_plan_config_path = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "transition_plan"
        / "transition_plan_config.py"
    )

    print(f"Working directory: {work_dir}")
    print(f"Input database: {input_db}")

    from eflips.x.flows import generate_all_plots

    spec = importlib.util.spec_from_file_location(
        "transition_plan_config", transition_plan_config_path
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    context_params: Dict[str, Any] = {
        "log_level": "INFO",
        "TCOParameterConfigurator.tco_params_path": tco_params_path,
        "TCOParameterConfigurator.scenario_name": "berlin_literature",
        "TransitionPlanner.name": config.name,
        "TransitionPlanner.sets": config.sets,
        "TransitionPlanner.variables": config.variables,
        "TransitionPlanner.constraints": config.constraints_long_term,
        "TransitionPlanner.expressions": config.expressions_long_term,
        "TransitionPlanner.objective_components": config.objective_components,
    }

    transition_planner_params = {
        k: v for k, v in context_params.items() if k.startswith("TransitionPlanner.")
    }
    with open(work_dir / "transition_planner_params.json", "w") as f:
        json.dump(transition_planner_params, f, indent=2, default=str)

    context = PipelineContext(
        work_dir=work_dir,
        current_db=input_db,
        params=context_params,
    )
    output_dir = context.work_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    #

    current_db_path, results = run_transition_planner(context)

    if results is not None:
        generate_transition_plan_plots(results, output_dir)

    simulate_multi_stage_electrification(
        unelectrified_blocks=results["unelectrified_blocks"],
        workdir=work_dir,
        input_db=current_db_path,
        force_rerun=FORCE_RERUN,
    )

    print(f"Transition planner results saved to: {current_db_path}")

    if False:
        playground_analyzer = PlaygroundAnalyzer()
        playground_flow = [playground_analyzer]

        for step in playground_flow:
            result = step.execute(context=context)
            # result.savefig(output_dir / f"{db_name}.png")

        generate_all_plots(
            context=context,
            output_dir=output_dir,
            pre_simulation_only=False,
        )

        run_tco_calculation(
            workdir=work_dir,
            input_db=input_db,
            tco_params_path=tco_params_path,
            scenario_name="berlin_literature",
        )

        simulate_multi_stage_electrification(
            unelectrified_blocks=unelectrified_blocks,
            workdir=work_dir,
            input_db=current_db,
            log_level="INFO",
            # force_rerun=True,
        )
