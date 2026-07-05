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
    plot_slot_distributions,
    simulate_multi_stage_electrification,
)
from eflips.x.transition_plan.transition_plan import (
    run_transition_planner,
    run_tco_calculation,
    run_diesel_simulation,
    TransitionPlanner,
)
from eflips.x.flows import run_steps

from eflips.x.transition_plan.transition_plan import PlaygroundAnalyzer

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent.parent.parent.parent / "eflips-data"

    diesel_db_name = "diesel_mini.db"
    diesel_input_db = data_dir / diesel_db_name
    electric_db_name = "electric_mini.db"
    electric_input_db = data_dir / electric_db_name

    work_dir_name = "test"

    work_dir = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "output"
        / "transition_plan"
        / work_dir_name
    )
    os.makedirs(work_dir, exist_ok=True)

    print(f"Working directory: {work_dir}")

    from eflips.x.flows import generate_all_plots

    # Single scenario that holds both electric and diesel VehicleTypes. Diesel
    # VTs are created on the fly by CreateDieselVehicleTypes from the electric ones.
    SCENARIO_ID = 1

    tco_params_path = Path(__file__).parent.parent.parent.parent / "data" / "TCO" / "tco.json"

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

    # Regenerate the depot slot-distribution plots from CSVs already saved under
    # ``work_dir/csv`` (transition-planner CSVs + per-stage occupancy), without
    # re-running the diesel simulation, transition planner or co-simulation. Set
    # to True to plot from existing data and exit.
    SLOT_PLOTS_ONLY = False
    if SLOT_PLOTS_ONLY:
        plot_slot_distributions(
            csv_dir=work_dir / "csv",
            output_dir=multi_stage_output_dir,
        )
        raise SystemExit(0)

    # ---------------------------------------------------------------------- #
    # 1. Diesel fleet baseline -- its own context and database. Set
    #    ``current_db`` directly to the diesel-flow source database; it is only
    #    copied into the diesel work_dir, never modified.
    # ---------------------------------------------------------------------- #

    SKIP_DIESEL_FLEET_PARAMS = (
        False  # Set to True to skip diesel-flow simulation and use hard-coded params.
    )

    if not SKIP_DIESEL_FLEET_PARAMS:
        diesel_context = PipelineContext(
            work_dir=work_dir / "diesel",
            current_db=diesel_input_db,  # TODO: set to the diesel-flow source database path.
            params={
                "log_level": "INFO",
                "DepotGenerator.charging_power_kw": 90.0,
                "Simulation.ignore_unstable_simulation": True,
                "Simulation.ignore_delayed_trips": True,
            },
        )
        os.makedirs(diesel_context.work_dir, exist_ok=True)
        diesel_fleet_params = run_diesel_simulation(diesel_context)

    else:

        from eflips.transition.parameter_registry import DieselFleetParams

        diesel_fleet_params = DieselFleetParams(
            bus_count_by_type={12: 67, 13: 112},
            initial_depot_capacities={160522: 202, 103159411: 33},
            slot_to_bus_ratio=1.0,
        )

    print("Diesel fleet params:")

    print(diesel_fleet_params)

    # ---------------------------------------------------------------------- #
    # 2. Electric transition-planning pipeline -- its own context and database,
    #    consuming the diesel fleet result from step 1.
    # ---------------------------------------------------------------------- #
    electric_context = PipelineContext(
        work_dir=work_dir,
        current_db=electric_input_db,  # TODO: set to the electric scenario database path.
        params={
            "log_level": "INFO",
            "scenario_id": SCENARIO_ID,
            TransitionPlanner.DIESEL_FLEET_PARAMS_PARAM: diesel_fleet_params,
            # Electric pipeline: fleet topology + TCO params written into the DB.
            "CompleteFleet.fleet_json": fleet_info_path,
            "TCOConfigurator.tco_json": tco_params_path,
            # Transition planner model configuration.
            "TransitionPlanner.constraint_params": config.constraints,
            "TransitionPlanner.name": config.name,
            "TransitionPlanner.sets": config.sets,
            "TransitionPlanner.variables": config.variables,
            "TransitionPlanner.constraints": config.constraints_long_term,
            "TransitionPlanner.expressions": config.expressions_long_term,
            "TransitionPlanner.objective_components": config.objective_components,
            "TransitionPlanner.procurement_components": config.procurement_components,
            "TransitionPlanner.shifted_procurement_components": config.shifted_procurement_components,
            "TransitionPlanner.plot_save_path": str(
                multi_stage_output_dir / "transition_planner_results.png"
            ),
            "TransitionPlanner.csv_save_dir": work_dir / "csv",
            "LCACalculator.lca_json_path": lca_params_path,
            "LCACalculator.overrides_json_path": lca_overrides_path,
            "LCACalculator.plot_by_type_save_path": str(
                multi_stage_output_dir / "lca_by_type.png"
            ),
            "LCACalculator.plot_by_scope_save_path": str(
                multi_stage_output_dir / "lca_by_scope.png"
            ),
        },
    )

    transition_planner_params = {
        k: v for k, v in electric_context.params.items() if k.startswith("TransitionPlanner.")
    }
    with open(work_dir / "transition_planner_params.json", "w") as f:
        json.dump(transition_planner_params, f, indent=2, default=str)

    output_dir = electric_context.work_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    current_db_path, results = run_transition_planner(electric_context)
    pd.DataFrame(
        [
            {"year": year, "rotation_id": rotation_id}
            for year, rotation_ids in results["unelectrified_blocks"].items()
            for rotation_id in rotation_ids
        ]
    ).to_csv(output_dir / "unelectrified_blocks.csv", index=False)

    # Cumulative electric depot-charger slots per depot per operational year
    # (size-weighted, length/12): the electrified + under-construction footprint
    # the diesel depot-config builder subtracts from initial_depot_capacities.
    results["depot_electric_slots_by_year"].to_csv(
        output_dir / "depot_electric_slots_by_year.csv", index=False
    )

    # Multi-stage co-simulation. ``input_db`` must be the DIESEL database produced
    # by run_diesel_simulation (every rotation on a diesel vehicle type, plus the
    # electric vehicle types), branched from the same electric scenario so its
    # rotation and station ids match ``results`` / ``diesel_fleet_params``. Uncomment
    # once the diesel flow (diesel_context) is run to produce that database.

    if SKIP_DIESEL_FLEET_PARAMS:

        input_db = Path(
            "/home/shuyao/PycharmProjects/eflips-x/data/output/transition_plan/test/diesel/step_005_Simulation.db"
        )
    else:
        input_db = diesel_context.current_db
    per_stage_vt_stats = simulate_multi_stage_electrification(
        unelectrified_blocks=results["unelectrified_blocks"],
        electric_slots_by_year=results["depot_electric_slots_by_year_map"],
        initial_depot_capacities=diesel_fleet_params.initial_depot_capacities,
        workdir=work_dir,
        input_db=input_db,  # diesel-sim output db
        electric_db=current_db_path,  # run_transition_planner output db
        csv_dir=work_dir / "csv",  # per-stage stats saved alongside planner CSVs
        # Drop each stage's step_001..step_004 scratch databases on success,
        # keeping only step_005_Simulation.db (~1.3 GB reclaimed across stages).
        cleanup_intermediate_dbs=True,
    )

    plot_multi_stage_vehicle_type_statistics(
        per_stage_results=per_stage_vt_stats,
        output_dir=multi_stage_output_dir,
    )

    # Planned (transition plan) vs. simulated (per-stage peak occupancy) depot
    # slot distribution: in operation electric / under construction / diesel.
    plot_slot_distributions(
        csv_dir=work_dir / "csv",
        output_dir=multi_stage_output_dir,
        initial_depot_capacities=diesel_fleet_params.initial_depot_capacities,
    )

    print(f"Transition planner results saved to: {current_db_path}")
