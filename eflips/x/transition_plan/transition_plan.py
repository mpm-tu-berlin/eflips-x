import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir

import sqlalchemy.orm.session
from eflips.depot.api import SmartChargingStrategy
from eflips.model import Rotation, VehicleType, Scenario
from sqlalchemy import func

from eflips.x.framework import PipelineStep, PipelineContext, Modifier
from eflips.x.flows import run_steps, generate_all_plots
from eflips.x.steps.modifiers.simulation import Simulation, DepotGenerator

from prefect import flow, task

from prefect.task_runners import ProcessPoolTaskRunner
from typing import List, Dict, Any


# rewrite Simulation to use a hybrid fleet configuration, and run several flow in parallel with different parameters
class CreateHybridFleet(Modifier):

    VEHICLE_TYPE_MAPPING: Dict[str, tuple] = {
        "EN": ("Diesel EN", "D_EN"),
        "GN": ("Diesel GN", "D_GN"),
        "DD": ("Diesel DD", "D_DD"),
    }

    DIESEL_CONSUMPTION = 0.0001  # Near-zero consumption for diesel simulation

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
            f"{cls.__name__}.unelectrified_block_ids": """
    The list of integers representing electrified blocks in this depot simulation. All unelectrified blocks are regarded as 
    diesel blocks and will be assigned to vehicles with very low energy consumption representing the diesel buses and only 
    use constant consumption estimation.

    Default: None
                """.strip(),
        }

    def _create_diesel_vehicle_type(
        self,
        session: sqlalchemy.orm.session.Session,
        electric_type: VehicleType,
        scenario: Scenario,
    ) -> VehicleType:
        """Create a diesel version of an electric vehicle type."""
        short_name = electric_type.name_short
        if short_name not in self.VEHICLE_TYPE_MAPPING:
            raise ValueError(
                f"No diesel type mapping for vehicle type '{short_name}'. "
                f"Available mappings: {list(self.VEHICLE_TYPE_MAPPING.keys())}"
            )

        diesel_name, diesel_short = self.VEHICLE_TYPE_MAPPING[short_name]

        diesel_type = VehicleType(
            scenario=scenario,
            name=diesel_name,
            name_short=diesel_short,
            battery_capacity=electric_type.battery_capacity,
            charging_curve=electric_type.charging_curve,
            opportunity_charging_capable=electric_type.opportunity_charging_capable,
            consumption=self.DIESEL_CONSUMPTION,
            battery_capacity_reserve=electric_type.battery_capacity_reserve,
            minimum_charging_power=electric_type.minimum_charging_power,
            charging_efficiency=electric_type.charging_efficiency,
        )
        session.add(diesel_type)
        return diesel_type

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """Apply hybrid fleet simulation configuration to the database."""
        unelectrified_block_ids = params.get(
            f"{self.__class__.__name__}.unelectrified_block_ids", None
        )

        if unelectrified_block_ids is not None or unelectrified_block_ids != []:
            # Set up hybrid fleet configuration based on electrified blocks
            # This is a placeholder for the actual implementation
            print(f"Configuring hybrid fleet with electrified blocks: {unelectrified_block_ids}")

            unelectrified_blocks = (
                session.query(Rotation).filter(Rotation.id.in_(unelectrified_block_ids)).all()
            )

            scenario = unelectrified_blocks[0].scenario
            # Add diesel vehicle types

            elec_en = (
                session.query(VehicleType)
                .filter(VehicleType.name_short == "EN", VehicleType.scenario_id == scenario.id)
                .one()
            )

            diesel_en = VehicleType(
                scenario=scenario,
                name="Diesel EN",
                name_short="D_EN",
                battery_capacity=elec_en.battery_capacity,
                charging_curve=elec_en.charging_curve,
                opportunity_charging_capable=elec_en.opportunity_charging_capable,
                consumption=0.0001,
                battery_capacity_reserve=elec_en.battery_capacity_reserve,
                minimum_charging_power=elec_en.minimum_charging_power,
                charging_efficiency=elec_en.charging_efficiency,
            )

            session.add(diesel_en)

            elec_gn = (
                session.query(VehicleType)
                .filter(VehicleType.name_short == "GN", VehicleType.scenario_id == scenario.id)
                .one()
            )

            diesel_gn = VehicleType(
                scenario=scenario,
                name="Diesel GN",
                name_short="D_GN",
                battery_capacity=elec_gn.battery_capacity,
                charging_curve=elec_gn.charging_curve,
                opportunity_charging_capable=elec_gn.opportunity_charging_capable,
                consumption=0.0001,
                battery_capacity_reserve=elec_gn.battery_capacity_reserve,
                minimum_charging_power=elec_gn.minimum_charging_power,
                charging_efficiency=elec_gn.charging_efficiency,
            )

            session.add(diesel_gn)

            elec_dd = (
                session.query(VehicleType)
                .filter(VehicleType.name_short == "DD", VehicleType.scenario_id == scenario.id)
                .one()
            )
            diesel_dd = VehicleType(
                scenario=scenario,
                name="Diesel DD",
                name_short="D_DD",
                battery_capacity=elec_dd.battery_capacity,
                charging_curve=elec_dd.charging_curve,
                opportunity_charging_capable=elec_dd.opportunity_charging_capable,
                consumption=0.0001,
                battery_capacity_reserve=elec_dd.battery_capacity_reserve,
                minimum_charging_power=elec_dd.minimum_charging_power,
                charging_efficiency=elec_dd.charging_efficiency,
            )
            session.add(diesel_dd)

            for d_block in unelectrified_blocks:
                if d_block.vehicle_type.name_short == "EN":
                    d_block.vehicle_type = diesel_en
                elif d_block.vehicle_type.name_short == "GN":
                    d_block.vehicle_type = diesel_gn
                elif d_block.vehicle_type.name_short == "DD":
                    d_block.vehicle_type = diesel_dd
                else:
                    raise ValueError(
                        f"Unknown vehicle type {d_block.vehicle_type.name_short} for diesel conversion."
                    )

            session.flush()


@task
@flow
def run_steps_local(context: PipelineContext, steps: List[PipelineStep]) -> None:
    """Run a sequence of pipeline steps."""
    for step in steps:
        step.execute(context=context)


@task
def generate_plots_as_task(
    context: PipelineContext,
    output_dir: Path,
    include_videos: bool = False,
    pre_simulation_only: bool = False,
) -> None:
    generate_all_plots(
        context=context,
        output_dir=output_dir,
        include_videos=include_videos,
        pre_simulation_only=pre_simulation_only,
    )


@flow(
    task_runner=ProcessPoolTaskRunner(max_workers=4),
)
def simulate_depot_multi_stage(
    unelectrified_blocks: List[List], workdir, pipeline_context: PipelineContext
) -> None:
    """Run multiple simulations with different depot configurations in parallel. The flow is designed as follows:
    1. For each stage defined by a list of unelectrified blocks:
        - Create a sub-pipeline context for each flow with a unique working directory.
        - Configure the CreateHybridFleet step with the current unelectrified blocks.
        - Add the DepotGenerator and Simulation steps to the pipeline.
        - Execute the pipeline steps in parallel using Prefect's task runner.
    2. After all simulations are complete, generate plots for each simulation run.
    Args:
        unelectrified_blocks (List[List]): A list where each element is a list of unelectrified block IDs for a stage.
        workdir (Path): The base working directory for the simulations.
        pipeline_context (PipelineContext): The base pipeline context to be used for each simulation.


    """

    parallel_flows = []
    sub_flow_work_dirs = []

    for unelectrified_blocks_per_stage in unelectrified_blocks:
        sub_pipeline_context = PipelineContext(
            work_dir=pipeline_context.work_dir,
            current_db=pipeline_context.current_db,
            params=pipeline_context.params.copy(),
        )
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # specify another output dir / work dir for each flow
        sub_pipeline_context.work_dir = workdir / (
            run_id + "stage" + str(unelectrified_blocks.index(unelectrified_blocks_per_stage))
        )
        sub_flow_work_dirs.append(sub_pipeline_context.work_dir)

        os.makedirs(sub_pipeline_context.work_dir, exist_ok=True)
        steps = []
        sub_pipeline_context.params["CreateHybridFleet.unelectrified_block_ids"] = (
            unelectrified_blocks_per_stage
        )
        create_hybrid_fleet = CreateHybridFleet()
        steps.append(create_hybrid_fleet)
        sub_pipeline_context.params["DepotGenerator.generate_optimal_depot"] = False
        depot_generator = DepotGenerator()
        steps.append(depot_generator)

        sub_pipeline_context.params["Simulation.smart_charging"] = SmartChargingStrategy.NONE
        simulation = Simulation()
        steps.append(simulation)

        parallel_flows.append(run_steps_local.submit(context=sub_pipeline_context, steps=steps))

        print("Waiting for simulations to complete...")

    for pf in parallel_flows:
        pf.result()

    plot_flows = []

    for run_dir in sub_flow_work_dirs:
        plot_pipeline_context = PipelineContext(
            work_dir=run_dir,
            current_db=run_dir / "step_003_Simulation.db",
            # TODO it is a hardcoded step name here, need to improve. And still need to delete all the existing dirs.
            params=pipeline_context.params,
        )
        plot_flows.append(
            generate_plots_as_task.submit(
                context=plot_pipeline_context,
                output_dir=run_dir / "plots",
                include_videos=False,
                pre_simulation_only=False,
            )
        )

    print("Waiting for plot generation to complete...")


if __name__ == "__main__":
    unelectrified_blocks = [[694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704], [694]]

    # setting curent db path
    # pass PipelineConstext into flow

    work_dir = Path(__file__).parent.parent.parent / "transition_plan"
    print(f"Working directory: {work_dir}")

    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    current_db = data_dir / "eflips_demo.db"

    params: Dict[str, Any] = {
        "log_level": "INFO",
    }

    pipeline_context = PipelineContext(
        work_dir=work_dir,
        current_db=current_db,
        params=params,
    )
    simulate_depot_multi_stage(
        unelectrified_blocks=unelectrified_blocks,
        workdir=work_dir,
        pipeline_context=pipeline_context,
    )
    print(pipeline_context.current_db)

    # get all dir under work_dir
