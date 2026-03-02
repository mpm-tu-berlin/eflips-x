import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import logging

import sqlalchemy.orm.session
from eflips.depot.api import SmartChargingStrategy
from eflips.model import Rotation, Scenario, VehicleType, DrivetrainType, Station
from prefect import flow, task
from prefect.futures import wait
from prefect.task_runners import ProcessPoolTaskRunner

from eflips.x.flows.analysis_flow import execute_simple_analyzer
from eflips.x.framework import Modifier, PipelineContext, PipelineStep
from eflips.x.steps.analyzers import (
    RotationInfoAnalyzer,
    GeographicTripPlotAnalyzer,
    DepartureArrivalSocAnalyzer,
    DepotEventAnalyzer,
    SpecificEnergyConsumptionAnalyzer,
)
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation


logger = logging.getLogger(__name__)


class CreateHybridFleet(Modifier):
    """
    Creates a hybrid fleet by adding diesel vehicle types for unelectrified blocks.

    This modifier creates "diesel" versions of existing electric vehicle types
    with near-zero consumption to represent diesel buses that don't require
    charging infrastructure.
    """

    # Mapping: electric_short_name -> (diesel_name, diesel_short_name)
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
            drivetrain_type=DrivetrainType.ICE,
        )
        session.add(diesel_type)
        return diesel_type

    def _create_all_diesel_types(
        self,
        session: sqlalchemy.orm.session.Session,
        scenario: Scenario,
    ) -> Dict[str, VehicleType]:
        """Create diesel versions of all mapped electric vehicle types."""
        diesel_types: Dict[str, VehicleType] = {}

        for short_name in self.VEHICLE_TYPE_MAPPING.keys():
            electric_type = (
                session.query(VehicleType)
                .filter(
                    VehicleType.name_short == short_name,
                    VehicleType.scenario_id == scenario.id,
                )
                .one_or_none()
            )

            if electric_type is not None:
                diesel_types[short_name] = self._create_diesel_vehicle_type(
                    session, electric_type, scenario
                )

        return diesel_types

    def _assign_diesel_types(
        self,
        blocks: List[Rotation],
        diesel_types: Dict[str, VehicleType],
    ) -> None:
        """Assign diesel vehicle types to unelectrified blocks."""
        for block in blocks:
            original_short = block.vehicle_type.name_short
            if original_short not in diesel_types:
                raise ValueError(
                    f"Unknown vehicle type '{original_short}' for diesel conversion. "
                    f"Available types: {list(diesel_types.keys())}"
                )
            block.vehicle_type = diesel_types[original_short]

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """Apply hybrid fleet simulation configuration to the database."""
        unelectrified_block_ids = params.get(
            f"{self.__class__.__name__}.unelectrified_block_ids", None
        )

        current_vehicle_types = session.query(VehicleType).all()
        for vt in current_vehicle_types:
            if vt.drivetrain_type != DrivetrainType.BEV:
                vt.drivetrain_type = DrivetrainType.BEV
            session.add(vt)

        if not unelectrified_block_ids:
            return

        print(f"Configuring hybrid fleet with unelectrified blocks: {unelectrified_block_ids}")

        unelectrified_blocks = (
            session.query(Rotation).filter(Rotation.id.in_(unelectrified_block_ids)).all()
        )

        if not unelectrified_blocks:
            return

        scenario = unelectrified_blocks[0].scenario

        # Create diesel vehicle types
        diesel_types = self._create_all_diesel_types(session, scenario)

        # Assign diesel types to unelectrified blocks
        self._assign_diesel_types(unelectrified_blocks, diesel_types)

        session.flush()


@task
def run_hybrid_fleet_simulation(
    context: PipelineContext,
    steps: List[PipelineStep],
    plot_output_dir: Path,
) -> None:
    """Run simulation steps serially, then generate plots."""
    for step in steps:
        step.execute(context=context)

    generate_simple_plots(
        context=context,
        output_dir=plot_output_dir,
    )


def generate_simple_plots(
    context: PipelineContext,
    output_dir: Path,
) -> None:
    """Generate plots for the transition plan results."""
    # Placeholder for actual plot generation logic based on results
    # For example, you could generate a plot showing the unelectrified blocks and depot locations

    logger.info(f"Starting analysis flow, outputs will be saved to: {output_dir}")

    # Define output directories
    simple_dir = output_dir / "simple"

    simple_dir.mkdir(parents=True, exist_ok=True)

    # Define simple analyzers by category
    pre_simulation_analyzers = [
        RotationInfoAnalyzer,
        GeographicTripPlotAnalyzer,
    ]

    post_simulation_analyzers = [
        DepartureArrivalSocAnalyzer,
        DepotEventAnalyzer,
        SpecificEnergyConsumptionAnalyzer,
    ]

    analyzers_to_run = pre_simulation_analyzers + post_simulation_analyzers

    all_futures = []
    for analyzer_class in analyzers_to_run:
        output_file = simple_dir / f"{analyzer_class.__name__}.html"
        future = execute_simple_analyzer.submit(analyzer_class, context, output_file)  # type: ignore[type-abstract]
        all_futures.append(future)

    # Execute InteractiveMapAnalyzer with depot and station plot directories
    logger.info("Submitting InteractiveMapAnalyzer...")
    map_output_file = output_dir / "InteractiveMapAnalyzer.html"

    logger.info("All analysis tasks submitted. Waiting for completion...")
    wait(all_futures)
    logger.info("Analysis flow complete. All outputs saved.")


@flow(
    task_runner=ProcessPoolTaskRunner(max_workers=4),
)
def simulate_multi_stage_electrification(
    unelectrified_blocks: Dict[int, List[int]],
    workdir: Path,
    input_db: Path,
    log_level: str = "INFO",
    force_rerun: bool = False,
) -> None:
    """Run multiple simulations with different depot configurations in parallel.

    Each stage runs serially: CreateHybridFleet -> DepotGenerator -> Simulation -> Plots.
    Stages run in parallel using ProcessPoolTaskRunner.

    Args:
        unelectrified_blocks: A list where each element is a list of unelectrified block IDs for a stage.
        workdir: The base working directory for the simulations.
        input_db: Path to the input database.
        log_level: Logging level for the pipeline steps.
    """
    parallel_flows = []

    for i, stage_blocks in unelectrified_blocks.items():
        if force_rerun:
            run_id = f"stage_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            run_id = f"stage_{i}"
        stage_workdir = workdir / run_id
        os.makedirs(stage_workdir, exist_ok=True)

        # Create fresh params for this stage
        params: Dict[str, Any] = {
            "log_level": log_level,
            "CreateHybridFleet.unelectrified_block_ids": stage_blocks,
            "DepotGenerator.generate_optimal_depot": False,
            "Simulation.smart_charging": SmartChargingStrategy.NONE,
        }

        # Create context for this stage
        sub_context = PipelineContext(
            work_dir=stage_workdir,
            current_db=input_db,
            params=params,
        )

        from eflips.x.flows.analysis_flow import query_all_ids

        all_rotation_ids = query_all_ids(sub_context, Rotation)

        all_electrified_ids = [b_id for b_id in all_rotation_ids if b_id not in stage_blocks]
        params["GeographicTripPlotAnalyzer.rotation_ids"] = all_electrified_ids
        params["GeographicTripPlotAnalyzer.plot_charging_station"] = True
        params["GeographicTripPlotAnalyzer.plot_depot_charger_count"] = True

        # Build pipeline steps
        steps: List[PipelineStep] = [
            CreateHybridFleet(),
            DepotGenerator(),
            Simulation(),
        ]

        # Submit stage for parallel execution
        parallel_flows.append(
            run_hybrid_fleet_simulation.submit(
                context=sub_context,
                steps=steps,
                plot_output_dir=stage_workdir / "plots",
            )
        )

    print("Waiting for simulations to complete...")
    for pf in parallel_flows:
        pf.result()
    print("All stages completed.")
