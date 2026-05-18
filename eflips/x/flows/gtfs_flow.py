#!/usr/bin/env python3

"""
Generalized GTFS Simulation Flow

Reads agency/depot configuration from an Excel file (depot_locations.xlsx) and runs
both DEPOT and OPPORTUNITY charging variants for each configured agency. Exports a
JSON scenario after depot assignment (before terminus charger placement).

Usage:
    python -m eflips.x.flows.gtfs_flow [--plots] [--agency FILTER] [--gtfs-dir DIR] [--parallel]
"""

import argparse
import logging
import math
import multiprocessing
import re
from concurrent.futures import as_completed, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from eflips.depot.api import SmartChargingStrategy  # type: ignore[import-untyped]
from eflips.model import ChargeType
from prefect import flow

from eflips.x.flows import generate_all_plots, run_steps
from eflips.x.framework import PipelineContext, PipelineStep
from eflips.x.steps.analyzers.json_export import ScenarioJsonExporter
from eflips.x.steps.generators import GTFSIngester, CopyCreator
from eflips.x.steps.modifiers.bvg_tools import MergeStations
from eflips.x.steps.modifiers.general_utilities import RemoveUnusedData
from eflips.x.steps.modifiers.gtfs_utilities import ConfigureVehicleTypes
from eflips.x.steps.modifiers.scheduling import (
    VehicleScheduling,
    DepotAssignment,
    IntegratedScheduling,
    InsufficientChargingTimeAnalyzer,
    StationElectrification,
)
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DepotConfig:
    name: str
    coords: Tuple[float, float]  # (lon, lat)
    capacity: int  # NaN → 9999


@dataclass
class AgencyConfig:
    """One logical simulation: one GTFS file, one or more agencies, one or more depots.

    Rows in ``depot_locations.xlsx`` are grouped by ``simulation_id`` to form
    a single AgencyConfig. Within a group, each row may contribute an agency
    (``agency_id`` / ``agency_name``), a depot (``name`` / ``coords`` / ...),
    or both.
    """

    simulation_id: str
    gtfs_file: Path
    agency_ids: List[str]
    agency_names: List[str]
    depots: List[DepotConfig]
    battery_capacity: float = 360.0
    consumption: float = 1.5
    charging_curve: List[List[float]] = field(default_factory=lambda: [[0.0, 450.0], [1.0, 450.0]])

    @property
    def agency_name(self) -> str:
        """Human-readable label combining all agencies in this simulation."""
        return " / ".join(self.agency_names) if self.agency_names else self.simulation_id

    @property
    def slug(self) -> str:
        """Sanitized simulation id for directory naming."""
        s = re.sub(r"[^a-z0-9]+", "_", self.simulation_id.lower()).strip("_")
        return s[:60]


# ---------------------------------------------------------------------------
# Excel parsing
# ---------------------------------------------------------------------------


def _parse_depot_row(row: "pd.Series[Any]") -> DepotConfig | None:
    """Extract a DepotConfig from an Excel row, or None if the row has no depot."""
    if not pd.notna(row.get("name")):
        return None
    lat_str, lon_str = str(row["coords"]).split(",", 1)
    capacity = 9999 if math.isnan(float(row["capacity"])) else int(row["capacity"])
    return DepotConfig(
        name=row["name"],
        coords=(float(lon_str.strip()), float(lat_str.strip())),
        capacity=capacity,
    )


def _norm_id(v: Any) -> str:
    """Normalize an Excel cell value to a string id.

    Why: pandas reads int-like columns as float64 when NaNs are present, so
    ``str(123)`` silently becomes ``'123.0'`` and stops matching GTFS ids.
    """
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


def _single_file_name(group: "pd.DataFrame", sim_id: str) -> str:
    """Return the single distinct non-null ``file_name`` in a simulation group."""
    file_names = {str(v) for v in group["file_name"].dropna()}
    if not file_names:
        raise ValueError(f"Simulation '{sim_id}' has no file_name on any row")
    if len(file_names) > 1:
        raise ValueError(f"Simulation '{sim_id}' has conflicting file_name values: {file_names}")
    return next(iter(file_names))


def parse_depot_locations(excel_path: Path) -> List[AgencyConfig]:
    """Parse the depot_locations Excel file into a list of AgencyConfig.

    Rows are grouped by ``simulation_id``. Within a simulation group:

    - ``file_name`` is taken from the (only) non-null value (rows may leave
      it blank; all non-blank values in the group must agree).
    - Each row with a non-null ``agency_id`` contributes one agency.
    - Each row with a non-null depot ``name`` contributes one depot.

    Only simulations with at least one depot are emitted. If the Excel file
    doesn't have a ``simulation_id`` column (legacy format) or a row leaves it
    blank, we fall back to ``agency_name`` for grouping — so single-agency
    rows continue to work without edits.
    """
    df = pd.read_excel(excel_path)

    if "simulation_id" not in df.columns:
        df["simulation_id"] = df["agency_name"]
    else:
        df["simulation_id"] = df["simulation_id"].fillna(df["agency_name"])

    df = df[df["simulation_id"].notna()].copy()
    if df.empty:
        raise ValueError(f"No entries with simulation_id found in {excel_path}")

    configs: List[AgencyConfig] = []
    for sim_id, group in df.groupby("simulation_id", sort=False):
        sim_id = str(sim_id)
        file_name = _single_file_name(group, sim_id)

        agencies_mask = group["agency_id"].notna()
        agency_ids = [_norm_id(v) for v in group.loc[agencies_mask, "agency_id"]]
        agency_names = [_norm_id(v) for v in group.loc[agencies_mask, "agency_name"].dropna()]
        depots = [d for d in (_parse_depot_row(r) for _, r in group.iterrows()) if d]

        if not depots:
            continue  # simulations with no configured depot are skipped

        configs.append(
            AgencyConfig(
                simulation_id=sim_id,
                gtfs_file=excel_path.parent / file_name,
                agency_ids=agency_ids,
                agency_names=agency_names,
                depots=depots,
            )
        )
    return configs


def build_depot_config(depots: List[DepotConfig]) -> List[Dict[str, Any]]:
    """Convert DepotConfig list to the dict format expected by DepotAssignment."""
    return [
        {
            "depot_station": depot.coords,
            "name": depot.name,
            "vehicle_type": ["default_bus"],
            "capacity": depot.capacity,
        }
        for depot in depots
    ]


# ---------------------------------------------------------------------------
# Pipeline phases (subflows)
# ---------------------------------------------------------------------------


@flow(
    name="GTFS Common Phase",
    flow_run_name="{agency_name} - common",
)
def run_common_phase(
    agency: AgencyConfig,
    cache_base: Path,
    agency_name: str,  # used for flow_run_name only
) -> Path:
    """Run the common pipeline phase (ingest → merge → cleanup → configure).

    Returns the path to the common DB for branching.
    """
    work_dir = cache_base / "common"
    work_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "GTFSIngester.bus_only": True,
        "GTFSIngester.duration": "WEEK",
        "GTFSIngester.agency_ids": agency.agency_ids,
        "ConfigureVehicleTypes.battery_capacity": agency.battery_capacity,
        "ConfigureVehicleTypes.consumption": agency.consumption,
        "ConfigureVehicleTypes.charging_curve": agency.charging_curve,
    }

    steps: List[PipelineStep] = [
        GTFSIngester(input_files=[agency.gtfs_file]),
        MergeStations(),
        RemoveUnusedData(),
        ConfigureVehicleTypes(),
    ]

    context = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps=steps, context=context)

    assert context.current_db is not None
    return context.current_db


@flow(
    name="GTFS Depot Variant",
    flow_run_name="{agency_name} - depot",
)
def run_depot_variant(
    agency: AgencyConfig,
    common_db: Path,
    cache_base: Path,
    output_base: Path,
    agency_name: str,  # used for flow_run_name only
    enable_plots: bool = False,
) -> None:
    """Run the DEPOT charging variant."""
    work_dir = cache_base / "depot"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_base / "depot"
    output_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "VehicleScheduling.charge_type": ChargeType.DEPOT,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=0),
        "VehicleScheduling.maximum_schedule_duration": timedelta(hours=24),
        "DepotAssignment.depot_config": build_depot_config(agency.depots),
        "Simulation.repetition_period": timedelta(weeks=1),
        "Simulation.smart_charging": SmartChargingStrategy.EVEN,
        "Simulation.ignore_unstable_simulation": True,
        "ScenarioJsonExporter.output_path": str(output_dir / "scenario.json"),
    }

    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[common_db]).execute(context=context)

    steps: List[PipelineStep] = [
        VehicleScheduling(),
        DepotAssignment(),
    ]
    run_steps(steps=steps, context=context)

    # Export JSON scenario (read-only, does not advance current_db)
    ScenarioJsonExporter().execute(context=context)

    post_steps: List[PipelineStep] = [
        DepotGenerator(),
        Simulation(),
    ]
    run_steps(steps=post_steps, context=context)

    if enable_plots:
        plots_dir = output_dir / "visualizations"
        plots_dir.mkdir(parents=True, exist_ok=True)
        generate_all_plots(
            context=context, output_dir=plots_dir, include_videos=False, pre_simulation_only=False
        )


@flow(
    name="GTFS Opportunity Variant",
    flow_run_name="{agency_name} - opportunity",
)
def run_opportunity_variant(
    agency: AgencyConfig,
    common_db: Path,
    cache_base: Path,
    output_base: Path,
    agency_name: str,  # used for flow_run_name only
    enable_plots: bool = False,
) -> None:
    """Run the OPPORTUNITY charging variant."""
    work_dir = cache_base / "opportunity"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_base / "opportunity"
    output_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=0),
        "VehicleScheduling.maximum_schedule_duration": timedelta(hours=24),
        "IntegratedScheduling.max_iterations": 3,
        "DepotAssignment.depot_config": build_depot_config(agency.depots),
        # InsufficientChargingTimeAnalyzer must use the same charging power as
        # StationElectrification — the latter validates this and refuses to run otherwise.
        "InsufficientChargingTimeAnalyzer.charging_power_kw": 450.0,
        "StationElectrification.charging_power_kw": 450.0,
        "StationElectrification.max_stations_to_electrify": 9999,
        "Simulation.repetition_period": timedelta(weeks=1),
        "Simulation.smart_charging": SmartChargingStrategy.EVEN,
        "Simulation.ignore_unstable_simulation": True,
        "ScenarioJsonExporter.output_path": str(output_dir / "scenario.json"),
    }

    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[common_db]).execute(context=context)

    # NOTE: IntegratedScheduling returns the database in the "just after vehicle scheduling"
    # state — it rolls back its nested DepotAssignment calls, so we must re-run DepotAssignment.
    steps: List[PipelineStep] = [
        IntegratedScheduling(),
        DepotAssignment(),
    ]
    run_steps(steps=steps, context=context)

    # Export JSON scenario (read-only, does not advance current_db)
    ScenarioJsonExporter().execute(context=context)

    # Diagnostic: identify rotations that are infeasible even with all termini electrified.
    # If this returns a non-empty result, StationElectrification cannot rescue them — the
    # schedule itself needs more break time (e.g. via IntegratedScheduling or by raising
    # VehicleScheduling.minimum_break_time above terminus_deadtime_s).
    insufficient_time_analyzer = InsufficientChargingTimeAnalyzer()
    insufficient_result = insufficient_time_analyzer.execute(context=context)

    if insufficient_result is not None:
        critical_ids = insufficient_result["rotation_ids"]
        soc_data = insufficient_result.get("soc_data", {})
        charging_power_kw = params.get("InsufficientChargingTimeAnalyzer.charging_power_kw", 450.0)

        lines = []
        for rot_id in critical_ids:
            if rot_id in soc_data:
                soc_df, _event_spans, rot_start, rot_end = soc_data[rot_id]
                min_soc = float(soc_df["soc"].min())
                deficit_kwh = abs(min_soc) * agency.battery_capacity
                lines.append(
                    f"  Rotation {rot_id}: min SoC={min_soc:.3f} "
                    f"(deficit ≈{deficit_kwh:.0f} kWh), "
                    f"window {rot_start} – {rot_end}"
                )
            else:
                lines.append(f"  Rotation {rot_id}: no SoC data available")

        raise RuntimeError(
            f"[{agency.agency_name}] Opportunity charging is not feasible: "
            f"{len(critical_ids)} rotation(s) cannot accumulate enough charge even with all "
            f"termini electrified at {charging_power_kw:.0f} kW "
            f"and a {agency.battery_capacity:.0f} kWh battery.\n"
            f"These routes are structurally too long or have too few break opportunities.\n"
            f"Infeasible rotations:\n"
            + "\n".join(lines)
            + "\nPossible remedies: increase battery_capacity, add a minimum_break_time, "
            "or accept DEPOT-only charging for this agency."
        )

    post_steps: List[PipelineStep] = [
        StationElectrification(),
        DepotGenerator(),
        Simulation(),
    ]
    run_steps(steps=post_steps, context=context)

    if enable_plots:
        plots_dir = output_dir / "visualizations"
        plots_dir.mkdir(parents=True, exist_ok=True)
        generate_all_plots(
            context=context, output_dir=plots_dir, include_videos=False, pre_simulation_only=False
        )


# ---------------------------------------------------------------------------
# Plain wrapper functions for ProcessPoolExecutor (must be module-level to be picklable)
# ---------------------------------------------------------------------------


def _run_agency_flow_worker(**kwargs: Any) -> None:
    run_agency_flow(**kwargs)


@flow(
    name="GTFS Agency",
    flow_run_name="{agency_name}",
)
def run_agency_flow(
    agency: AgencyConfig,
    cache_base_root: Path,
    output_base_root: Path,
    agency_name: str,  # used for flow_run_name only
    enable_plots: bool = False,
    tolerate_failures: bool = True,
) -> None:
    """Run the full pipeline (common → depot + opportunity) for a single agency."""
    gtfs_stem = agency.gtfs_file.stem
    slug = agency.slug

    cache_base = cache_base_root / gtfs_stem / slug
    output_base = output_base_root / gtfs_stem / slug

    logger.info(f"Processing agency: {agency.agency_name} ({gtfs_stem}/{slug})")

    common_db = run_common_phase(
        agency=agency,
        cache_base=cache_base,
        agency_name=agency.agency_name,
    )

    variant_kwargs = dict(
        agency=agency,
        common_db=common_db,
        cache_base=cache_base,
        output_base=output_base,
        agency_name=agency.agency_name,
        enable_plots=enable_plots,
    )
    run_depot_variant(**variant_kwargs)  # type: ignore[call-overload]
    run_opportunity_variant(**variant_kwargs)  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(name="Generalized GTFS Flow")
def gtfs_flow(
    enable_plots: bool = False,
    agency_filter: str | None = None,
    gtfs_dir: Path | None = None,
    parallel: bool = False,
    tolerate_failures: bool = True,
) -> None:
    """Run DEPOT and OPPORTUNITY charging variants for all configured GTFS agencies."""
    if gtfs_dir is None:
        gtfs_dir = PROJECT_ROOT / "data" / "input" / "GTFS"

    excel_path = gtfs_dir / "depot_locations.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"Depot locations Excel not found: {excel_path}")

    agencies = parse_depot_locations(excel_path)

    if agency_filter:
        filter_lower = agency_filter.lower()

        def matches(a: AgencyConfig) -> bool:
            if filter_lower in a.simulation_id.lower():
                return True
            return any(filter_lower in n.lower() for n in a.agency_names)

        agencies = [a for a in agencies if matches(a)]
        if not agencies:
            raise ValueError(f"No agencies matched filter '{agency_filter}'")

    cache_base_root = PROJECT_ROOT / "data" / "cache" / "gtfs"
    output_base_root = PROJECT_ROOT / "data" / "output" / "gtfs"

    failed_agencies: List[Tuple[AgencyConfig, Exception]] = []

    if parallel:
        max_workers = min(len(agencies), multiprocessing.cpu_count())
        logger.info(f"Running {len(agencies)} agencies in parallel (max_workers={max_workers})")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_agency_flow_worker,
                    agency=agency,
                    cache_base_root=cache_base_root,
                    output_base_root=output_base_root,
                    agency_name=agency.agency_name,
                    enable_plots=enable_plots,
                    tolerate_failures=tolerate_failures,
                ): agency
                for agency in agencies
            }
            for future in as_completed(futures):
                agency = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Agency '{agency.agency_name}' failed: {exc}")
                    if tolerate_failures:
                        failed_agencies.append((agency, exc))
                    else:
                        raise
    else:
        for agency in agencies:
            if tolerate_failures:
                try:
                    run_agency_flow(
                        agency=agency,
                        cache_base_root=cache_base_root,
                        output_base_root=output_base_root,
                        agency_name=agency.agency_name,
                        enable_plots=enable_plots,
                        tolerate_failures=tolerate_failures,
                    )
                except Exception as exc:
                    logger.error(f"Agency '{agency.agency_name}' failed: {exc}")
                    failed_agencies.append((agency, exc))
            else:
                run_agency_flow(
                    agency=agency,
                    cache_base_root=cache_base_root,
                    output_base_root=output_base_root,
                    agency_name=agency.agency_name,
                    enable_plots=enable_plots,
                    tolerate_failures=tolerate_failures,
                )

    if failed_agencies:
        summary_lines = [
            f"  {a.agency_name}: {type(exc).__name__}: {str(exc).splitlines()[0]}"
            for a, exc in failed_agencies
        ]
        raise RuntimeError(
            f"{len(failed_agencies)} of {len(agencies)} agency run(s) failed:\n"
            + "\n".join(summary_lines)
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generalized GTFS Simulation Flow")
    parser.add_argument("--plots", action="store_true", help="Enable plot generation")
    parser.add_argument(
        "--agency",
        type=str,
        default=None,
        help="Case-insensitive substring filter on agency name",
    )
    parser.add_argument(
        "--gtfs-dir",
        type=Path,
        default=None,
        help="Override input directory (must contain depot_locations.xlsx and GTFS zips)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run agencies in parallel using processes (one process per agency)",
    )
    parser.add_argument(
        "--no-tolerate-failures",
        action="store_true",
        help="Abort immediately when any agency fails instead of collecting all failures",
    )
    args = parser.parse_args()

    gtfs_flow(
        enable_plots=args.plots,
        agency_filter=args.agency,
        gtfs_dir=args.gtfs_dir,
        parallel=args.parallel,
        tolerate_failures=not args.no_tolerate_failures,
    )
