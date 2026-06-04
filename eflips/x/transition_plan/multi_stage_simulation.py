import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy.orm.session
from eflips.depot.api import SmartChargingStrategy
from eflips.eval.output import prepare as eval_output_prepare
from collections import defaultdict

from eflips.model import (
    Area,
    Depot,
    EnergySource,
    Event,
    EventType,
    Rotation,
    Scenario,
    Station,
    Vehicle,
    VehicleType,
)
from prefect import flow, task
from prefect.futures import wait
from prefect.task_runners import ProcessPoolTaskRunner
from sqlalchemy import func

from eflips.x.flows.analysis_flow import execute_simple_analyzer
from eflips.x.framework import Modifier, PipelineContext, PipelineStep, Analyzer
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
    Create a diesel counterpart for every electric vehicle type in the scenario
    and optionally reassign a set of blocks to the diesel types.

    A diesel VT is created for every ``EnergySource.BATTERY_ELECTRIC`` VehicleType
    using the ``"Diesel {name_short}"`` naming convention expected by
    :class:`eflips.transition.parameter_registry.ParameterRegistry`. Diesel VTs
    get near-zero consumption and ``energy_source=EnergySource.DIESEL``.

    When ``unelectrified_block_ids`` is given, the matching rotations are
    reassigned to their diesel counterpart. Otherwise only the VT records are
    created — safe to reuse upstream of TCO/transition-planner steps.

    Idempotent: a diesel VT whose ``name_short`` already exists in the scenario
    is reused, never duplicated.
    """

    DIESEL_CONSUMPTION = 0.0001  # Near-zero consumption for diesel simulation
    DIESEL_PREFIX = "Diesel "

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
            f"{cls.__name__}.unelectrified_block_ids": """
    Optional list of Rotation IDs to reassign to diesel vehicle types. When set,
    those rotations are treated as diesel blocks and attached to the matching
    diesel VehicleType (near-zero consumption, constant-consumption estimation).

    Default: None
                """.strip(),
            f"{cls.__name__}.scenario_id": """
    Scenario to create diesel vehicle types in. Falls back to the global
    ``scenario_id`` parameter, and finally to the scenario of the first
    unelectrified block when only ``unelectrified_block_ids`` is given.

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
        diesel_type = VehicleType(
            scenario=scenario,
            name=f"{self.DIESEL_PREFIX}{electric_type.name}",
            name_short=f"{self.DIESEL_PREFIX}{electric_type.name_short}",
            battery_capacity=electric_type.battery_capacity,
            charging_curve=electric_type.charging_curve,
            opportunity_charging_capable=electric_type.opportunity_charging_capable,
            consumption=self.DIESEL_CONSUMPTION,
            battery_capacity_reserve=electric_type.battery_capacity_reserve,
            minimum_charging_power=electric_type.minimum_charging_power,
            charging_efficiency=electric_type.charging_efficiency,
            energy_source=EnergySource.DIESEL,
        )
        session.add(diesel_type)
        return diesel_type

    def _create_all_diesel_types(
        self,
        session: sqlalchemy.orm.session.Session,
        scenario: Scenario,
    ) -> Dict[str, VehicleType]:
        """Create a diesel counterpart for every electric VehicleType in the scenario.

        Returns a mapping keyed by the electric VT's ``name_short``. Existing
        diesel counterparts (matched by ``"Diesel {name_short}"``) are reused.
        """
        electric_types = (
            session.query(VehicleType)
            .filter(
                VehicleType.scenario_id == scenario.id,
                VehicleType.energy_source == EnergySource.BATTERY_ELECTRIC,
            )
            .all()
        )

        if len(electric_types) == 0:
            logger.warning(
                f"No electric vehicle types found in scenario {scenario.id}. "
                f"CreateHybridFleet convert them to electric for now"
            )

            all_types = (
                session.query(VehicleType).filter(VehicleType.energy_source.is_(None)).all()
            )
            for vt in all_types:
                vt.energy_source = EnergySource.BATTERY_ELECTRIC
        session.flush()
        session.expire_all()

        electric_types = (
            session.query(VehicleType)
            .filter(
                VehicleType.scenario_id == scenario.id,
                VehicleType.energy_source == EnergySource.BATTERY_ELECTRIC,
            )
            .all()
        )

        diesel_types: Dict[str, VehicleType] = {}
        for electric_type in electric_types:
            diesel_short = f"{self.DIESEL_PREFIX}{electric_type.name_short}"
            existing = (
                session.query(VehicleType)
                .filter(
                    VehicleType.scenario_id == scenario.id,
                    VehicleType.name_short == diesel_short,
                )
                .one_or_none()
            )
            if existing is not None:
                diesel_types[electric_type.name_short] = existing
            else:
                diesel_types[electric_type.name_short] = self._create_diesel_vehicle_type(
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
        """Create diesel VTs and optionally reassign unelectrified blocks to them."""
        unelectrified_block_ids = params.get(
            f"{self.__class__.__name__}.unelectrified_block_ids", None
        )

        scenario = session.query(Scenario).one()

        diesel_types = self._create_all_diesel_types(session, scenario)

        if unelectrified_block_ids:
            print(
                f"Configuring hybrid fleet with unelectrified blocks: "
                f"{unelectrified_block_ids}"
            )
            unelectrified_blocks = (
                session.query(Rotation).filter(Rotation.id.in_(unelectrified_block_ids)).all()
            )
            if unelectrified_blocks:
                self._assign_diesel_types(unelectrified_blocks, diesel_types)

        session.flush()


class DeleteDepotEvents(Modifier):
    """Delete all depot events to simulate a greenfield scenario for each stage."""

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {**super().document_params()}

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """Delete all depot events."""

        session.query(Event).delete()
        session.query(Rotation).update({Rotation.vehicle_id: None})
        session.query(Vehicle).delete()
        logger.info(f"Deleted depot events and vehicles.")

        session.flush()


class VehicleTypeStatistics(Analyzer):
    """Per-vehicle-type fleet size and peak depot occupancy (by depot).

    Returns ``{"counts": DataFrame, "occupancy": DataFrame}``:

    - ``counts``: one row per :class:`VehicleType` with ``vehicle_type_id``,
      ``vehicle_type_name``, ``energy_source``, ``vehicle_count``.
    - ``occupancy``: one row per (VehicleType, Depot) with
      ``vehicle_type_id``, ``vehicle_type_name``, ``energy_source``,
      ``depot_id``, ``depot_name``, ``peak_occupancy``. ``peak_occupancy``
      is ``max(occupancy_total)`` from
      :func:`eflips.eval.output.prepare.power_and_occupancy` over all areas
      that target this VT in that depot and have at least one
      ``EventType.STANDBY_DEPARTURE`` event. VTs with no matching areas
      (e.g. diesel) are absent from ``occupancy``.
    """

    def __init__(self, code_version: str = "1", cache_enabled: bool = True) -> None:
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.temporal_resolution": """
    Temporal resolution in seconds passed to power_and_occupancy when computing
    peak depot occupancy. Default: 60.
                """.strip(),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        temporal_resolution = params.get(f"{self.__class__.__name__}.temporal_resolution", 60)

        vehicle_counts: Dict[int, int] = {
            vt_id: count
            for vt_id, count in session.query(Vehicle.vehicle_type_id, func.count(Vehicle.id))
            .group_by(Vehicle.vehicle_type_id)
            .all()
        }

        vehicle_types = session.query(VehicleType).all()
        depot_names: Dict[int, str] = {
            d_id: name for d_id, name in session.query(Depot.id, Depot.name).all()
        }

        counts_rows: List[Dict[str, Any]] = []
        occupancy_rows: List[Dict[str, Any]] = []

        for vt in vehicle_types:
            counts_rows.append(
                {
                    "vehicle_type_id": vt.id,
                    "vehicle_type_name": vt.name,
                    "energy_source": vt.energy_source,
                    "vehicle_count": int(vehicle_counts.get(vt.id, 0)),
                }
            )

            areas_by_depot: Dict[int, List[int]] = defaultdict(list)
            for area_id, depot_id in (
                session.query(Area.id, Area.depot_id)
                .join(Event, Event.area_id == Area.id)
                .filter(
                    Area.vehicle_type_id == vt.id,
                    Event.event_type == EventType.STANDBY_DEPARTURE,
                )
                .distinct()
                .all()
            ):
                areas_by_depot[depot_id].append(area_id)

            for depot_id, area_ids in areas_by_depot.items():
                occupancy_df = eval_output_prepare.power_and_occupancy(
                    area_ids, session, temporal_resolution
                )
                occupancy_rows.append(
                    {
                        "vehicle_type_id": vt.id,
                        "vehicle_type_name": vt.name,
                        "energy_source": vt.energy_source,
                        "depot_id": depot_id,
                        "depot_name": depot_names.get(depot_id),
                        "peak_occupancy": float(occupancy_df["occupancy_total"].max()),
                    }
                )

        return {
            "counts": pd.DataFrame(counts_rows),
            "occupancy": pd.DataFrame(occupancy_rows),
        }


@task
def run_hybrid_fleet_simulation(
    context: PipelineContext,
    steps: List[PipelineStep],
    plot_output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Run simulation steps serially, collect VehicleTypeStatistics, then generate plots.

    The final step must be :class:`VehicleTypeStatistics`; its result
    (``{"counts": DataFrame, "occupancy": DataFrame}``) is returned so the
    caller can aggregate per-stage statistics for cross-stage plotting.
    """
    vt_stats_result: Dict[str, pd.DataFrame] = {}
    for step in steps:
        if isinstance(step, VehicleTypeStatistics):
            vt_stats_result = step.execute(context=context)
        else:
            step.execute(context=context)

    generate_simple_plots(
        context=context,
        output_dir=plot_output_dir,
    )

    return vt_stats_result


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
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Run multiple simulations with different depot configurations in parallel.

    Each stage runs serially: CreateHybridFleet -> DepotGenerator -> Simulation
    -> VehicleTypeStatistics -> Plots. Stages run in parallel using
    ProcessPoolTaskRunner.

    Args:
        unelectrified_blocks: A list where each element is a list of unelectrified block IDs for a stage.
        workdir: The base working directory for the simulations.
        input_db: Path to the input database.
        log_level: Logging level for the pipeline steps.

    Returns:
        Dict keyed by stage id, each value being the
        :class:`VehicleTypeStatistics` result
        (``{"counts": DataFrame, "occupancy": DataFrame}``) for that stage.
    """
    parallel_flows: List[Any] = []
    stage_ids: List[int] = []

    for i, stage_blocks in unelectrified_blocks.items():
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
            DeleteDepotEvents(),
            CreateHybridFleet(),
            DepotGenerator(),
            Simulation(),
            VehicleTypeStatistics(),
        ]

        # Submit stage for parallel execution
        parallel_flows.append(
            run_hybrid_fleet_simulation.submit(
                context=sub_context,
                steps=steps,
                plot_output_dir=stage_workdir / "plots",
            )
        )
        stage_ids.append(i)

    print("Waiting for simulations to complete...")
    per_stage_results: Dict[int, Dict[str, pd.DataFrame]] = {}
    for stage_id, pf in zip(stage_ids, parallel_flows):
        per_stage_results[stage_id] = pf.result()
    print("All stages completed.")
    return per_stage_results


DIESEL_HATCH = "///"


def _vehicle_type_color_map(vt_names: List[str]) -> Dict[str, Any]:
    """Stable color per vehicle type, shared across both plots."""
    cmap = plt.get_cmap("tab20")
    return {name: cmap(i % cmap.N) for i, name in enumerate(vt_names)}


def _hatch_legend_handles() -> List[mpatches.Patch]:
    return [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Electric"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=DIESEL_HATCH, label="Diesel"),
    ]


def plot_multi_stage_vehicle_type_statistics(
    per_stage_results: Dict[int, Dict[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Render cross-stage VehicleTypeStatistics plots.

    Writes two PNGs into ``output_dir``:

    - ``vehicle_count_per_stage.png`` — stacked bars of vehicle count per
      vehicle type, one bar per stage. Diesel types are hatched.
    - ``peak_occupancy_per_stage_depot.png`` — grouped (by depot) + stacked
      (by vehicle type) bars of peak depot occupancy, one group per stage.
      Diesel types are hatched (typically absent — diesel has no depot
      events).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_ids = sorted(per_stage_results.keys())
    counts = pd.concat(
        [per_stage_results[s]["counts"].assign(stage=s) for s in stage_ids],
        ignore_index=True,
    )
    occupancy = pd.concat(
        [per_stage_results[s]["occupancy"].assign(stage=s) for s in stage_ids],
        ignore_index=True,
    )

    # Shared across both plots: stable color + diesel lookup per VT name.
    all_vt_names = sorted(set(counts["vehicle_type_name"]).union(occupancy["vehicle_type_name"]))
    vt_colors = _vehicle_type_color_map(all_vt_names)
    diesel_by_vt = {
        name: energy == EnergySource.DIESEL
        for name, energy in pd.concat(
            [
                counts[["vehicle_type_name", "energy_source"]],
                occupancy[["vehicle_type_name", "energy_source"]],
            ]
        )
        .drop_duplicates()
        .itertuples(index=False)
    }

    _plot_vehicle_count_per_stage(counts, stage_ids, vt_colors, diesel_by_vt, output_dir)
    _plot_peak_occupancy_per_stage_depot(occupancy, stage_ids, vt_colors, diesel_by_vt, output_dir)


def _plot_vehicle_count_per_stage(
    counts: pd.DataFrame,
    stage_ids: List[int],
    vt_colors: Dict[str, Any],
    diesel_by_vt: Dict[str, bool],
    output_dir: Path,
) -> None:
    pivot = (
        counts.pivot_table(
            index="stage",
            columns="vehicle_type_name",
            values="vehicle_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(stage_ids)
        .fillna(0)
    )

    fig, ax = plt.subplots(layout="constrained")
    x = np.arange(len(stage_ids))
    bottom = np.zeros(len(stage_ids))
    for vt_name in pivot.columns:
        heights = pivot[vt_name].to_numpy()
        ax.bar(
            x,
            heights,
            bottom=bottom,
            color=vt_colors[vt_name],
            edgecolor="black",
            hatch=DIESEL_HATCH if diesel_by_vt.get(vt_name) else None,
            label=vt_name,
        )
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in stage_ids])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Vehicle count")
    ax.set_title("Vehicle count per type per stage")

    type_legend = ax.legend(title="Vehicle type", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(type_legend)
    ax.legend(
        handles=_hatch_legend_handles(),
        title="Energy source",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
    )

    fig.savefig(output_dir / "vehicle_count_per_stage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_peak_occupancy_per_stage_depot(
    occupancy: pd.DataFrame,
    stage_ids: List[int],
    vt_colors: Dict[str, Any],
    diesel_by_vt: Dict[str, bool],
    output_dir: Path,
) -> None:
    if occupancy.empty:
        logger.warning("No peak occupancy data; skipping peak occupancy plot.")
        return

    depots = sorted(occupancy["depot_name"].dropna().unique().tolist())
    pivot = occupancy.pivot_table(
        index=["stage", "depot_name"],
        columns="vehicle_type_name",
        values="peak_occupancy",
        aggfunc="max",
        fill_value=0,
    )

    x = np.arange(len(stage_ids))

    for depot in depots:
        fig, ax = plt.subplots(layout="constrained")
        bottom = np.zeros(len(stage_ids))
        for vt_name in pivot.columns:
            heights = np.array(
                [
                    (pivot.loc[(s, depot), vt_name] if (s, depot) in pivot.index else 0.0)
                    for s in stage_ids
                ]
            )
            if not heights.any():
                continue
            ax.bar(
                x,
                heights,
                bottom=bottom,
                width=0.6,
                color=vt_colors[vt_name],
                edgecolor="black",
                hatch=DIESEL_HATCH if diesel_by_vt.get(vt_name) else None,
                label=vt_name,
            )
            bottom += heights

        ax.set_xticks(x)
        ax.set_xticklabels([f"Stage {s}" for s in stage_ids])
        ax.set_xlabel("Stage")
        ax.set_ylabel("Peak occupancy")
        ax.set_title(f"Peak depot occupancy — {depot}")

        type_handles = [
            mpatches.Patch(
                facecolor=vt_colors[vt],
                edgecolor="black",
                hatch=DIESEL_HATCH if diesel_by_vt.get(vt) else None,
                label=vt,
            )
            for vt in pivot.columns
        ]
        type_legend = ax.legend(
            handles=type_handles,
            title="Vehicle type",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        ax.add_artist(type_legend)
        ax.legend(
            handles=_hatch_legend_handles(),
            title="Energy source",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
        )

        safe_name = depot.replace(" ", "_").replace("/", "-")
        fig.savefig(
            output_dir / f"peak_occupancy_per_stage_{safe_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
