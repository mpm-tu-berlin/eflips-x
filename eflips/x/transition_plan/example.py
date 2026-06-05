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

from eflips.x.transition_plan.multi_stage_simulation import (
    plot_multi_stage_vehicle_type_statistics,
    simulate_multi_stage_electrification,
)
from eflips.x.transition_plan.transition_plan import (
    run_transition_planner,
    run_tco_calculation,
)
from eflips.x.flows import run_steps

from eflips.x.transition_plan.transition_plan import PlaygroundAnalyzer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run transition planner pipeline.")
    parser.add_argument(
        "--tco_params",
        type=str,
        default=None,
        help="Path to TCO parameters JSON file.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent.parent.parent.parent / "eflips-data"
    db_name = "Simulation_ou.db"
    input_db = data_dir / db_name

    work_subdir = Path(args.tco_params).stem if args.tco_params is not None else db_name
    work_dir = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "output"
        / "transition_plan"
        / work_subdir
    )
    os.makedirs(work_dir, exist_ok=True)

    print(f"Working directory: {work_dir}")
    print(f"Input database: {input_db}")

    from eflips.x.flows import generate_all_plots

    # Single scenario that holds both electric and diesel VehicleTypes. Diesel
    # VTs are created on the fly by CreateHybridFleet from the electric ones.
    SCENARIO_ID = 1

    tco_params_path = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "TCO"
        / "sensitivity_analysis"
        / args.tco_params
        if args.tco_params is not None
        else Path(__file__).parent.parent.parent.parent / "data" / "TCO" / "tco.json"
    )

    tco_params_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "TCO" / "sensitivity_analysis" / args.tco_params
        if args.tco_params is not None
        else Path(__file__).parent.parent.parent.parent / "data" / "TCO" / "tco.json"
    )

    fleet_info_path = Path(__file__).parent.parent.parent.parent / "data" / "TCO" / "fleet.json"

    lca_params_path = Path(__file__).parent.parent.parent.parent / "data" / "TCO" / "lca.json"
    lca_overrides_path = (
        Path(__file__).parent.parent.parent.parent / "data" / "TCO" / "lca_overrides.json"
    )

    transition_plan_config_path = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "transition_plan"
        / "transition_plan_config.py"
    )

    spec = importlib.util.spec_from_file_location(
        "transition_plan_config", transition_plan_config_path
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    multi_stage_output_dir = work_dir / "plots" / "multi_stage"

    context_params: Dict[str, Any] = {
        "log_level": "INFO",
        "scenario_id": SCENARIO_ID,
        "TCOParameterConfigurator.tco_params_path": tco_params_path,
        "TCOParameterConfigurator.fleet_info_path": fleet_info_path,
        # TransitionPlanner params are preserved for when the planner step is
        # wired back into run_transition_planner.
        "TransitionPlanner.constraint_params": config.constraints,
        "TransitionPlanner.name": config.name,
        "TransitionPlanner.sets": config.sets,
        "TransitionPlanner.variables": config.variables,
        "TransitionPlanner.constraints": config.constraints_long_term,
        "TransitionPlanner.expressions": config.expressions_long_term,
        "TransitionPlanner.objective_components": config.objective_components,
        "TransitionPlanner.plot_save_path": str(
            multi_stage_output_dir / "transition_planner_results.png"
        ),
        "TransitionPlanner.csv_save_dir": work_dir / "csv",
        "LCACalculator.lca_json_path": lca_params_path,
        "LCACalculator.overrides_json_path": lca_overrides_path,
        "LCACalculator.plot_by_type_save_path": str(multi_stage_output_dir / "lca_by_type.png"),
        "LCACalculator.plot_by_scope_save_path": str(multi_stage_output_dir / "lca_by_scope.png"),
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
    pd.DataFrame(
        [
            {"year": year, "rotation_id": rotation_id}
            for year, rotation_ids in results["unelectrified_blocks"].items()
            for rotation_id in rotation_ids
        ]
    ).to_csv(output_dir / "unelectrified_blocks.csv", index=False)

    # per_stage_vt_stats = simulate_multi_stage_electrification(
    #     unelectrified_blocks=results["unelectrified_blocks"],
    #     workdir=work_dir,
    #     input_db=current_db_path,
    # )
    #
    # plot_multi_stage_vehicle_type_statistics(
    #     per_stage_results=per_stage_vt_stats,
    #     output_dir=multi_stage_output_dir,
    # )

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
