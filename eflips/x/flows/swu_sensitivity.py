#!/usr/bin/env python3

"""
SWU Sensitivity Analysis

One-factor-at-a-time sweep of four input parameters (battery capacity, outside
temperature, depot charging power, terminus charging power) against fleet
sizing and infrastructure outputs, for both depot-only (DEP) and opportunity
(TERM) charging variants. Produces one combined table plus print-sized per-
factor plots and a screen-sized seaborn factor grid.

Usage:
    python -m eflips.x.flows.swu_sensitivity run     # run the sweep, write table
    python -m eflips.x.flows.swu_sensitivity plot    # read table, write plots
    python -m eflips.x.flows.swu_sensitivity all     # run + plot
"""

import argparse
import logging
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast


# ---------------------------------------------------------------------------
# kv_cache concurrency patch — must be applied before any module that calls
# ``eflips.model.util.geometry.get_altitude`` runs against the cache, but the
# class-level monkey-patch is also safe to apply after the module-level store
# was constructed (method lookup goes through the class).
# ---------------------------------------------------------------------------


def _patch_kv_cache_for_parallel_access() -> None:
    """Make ``kv_cache.KVStore`` safe to share between many parallel workers.

    The eflips altitude cache at ``~/.cache/eflips/.../eflips_ingest_altitude_cache.db``
    is read on every depot-rotation matching iteration via ``get_altitude``.
    Two design choices in ``kv_cache`` cause it to deadlock under high
    parallelism:

    1. ``KVStore.get()`` issues a ``DELETE`` for expired entries on **every**
       read. The altitude cache never sets a TTL, so the delete touches
       nothing — but it still acquires the SQLite write lock.
    2. Connections are opened with the default 5-second ``busy_timeout``.
       With dozens of Dask workers contending for that write lock, the
       timeout fires before the lock is released.

    We monkey-patch the class to skip the cleanup and to install a 60-second
    busy timeout on every new connection. The patch is idempotent — calling
    it from each worker process is harmless.
    """
    from kv_cache import main as kv_main  # type: ignore[import-untyped]

    if getattr(kv_main.KVStore, "_eflips_x_patched", False):
        return

    def _noop_cleanup_expired(_self: Any) -> None:
        return

    def _get_connection_with_long_timeout(self: Any) -> sqlite3.Connection:
        if not hasattr(self._conn, "connection"):
            conn = sqlite3.connect(self.db_path, timeout=60.0)
            conn.execute("PRAGMA busy_timeout = 60000")
            self._conn.connection = conn
        return self._conn.connection  # type: ignore[no-any-return]

    kv_main.KVStore._cleanup_expired = _noop_cleanup_expired
    kv_main.KVStore._get_connection = _get_connection_with_long_timeout
    kv_main.KVStore._eflips_x_patched = True


_patch_kv_cache_for_parallel_access()


import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # type: ignore[import-untyped]  # noqa: E402
from eflips.model import Depot, Event, EventType, Rotation  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from prefect import flow, task  # noqa: E402
from prefect_dask.task_runners import DaskTaskRunner  # noqa: E402
from sqlalchemy.orm import joinedload  # noqa: E402

from eflips.x.flows.swu_flow import (  # noqa: E402
    CACHE_BASE,
    OUTPUT_BASE,
    run_common_phase,
    run_depot_variant,
    run_opportunity_variant,
    run_pre_common_phase,
)
from eflips.x.framework import PipelineContext  # noqa: E402
from eflips.x.steps.analyzers.bvg_tools import (  # noqa: E402
    PLOT_HEIGHT_INCH,
    PLOT_WIDTH_INCH,
    ScenarioComparisonAnalyzer,
    configure_latex_plotting,
)
from eflips.x.steps.analyzers.output_analyzers import (  # noqa: E402
    EnergyConsumptionByVehicleTypeAnalyzer,
    PowerAndOccupancyAnalyzer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

SWEEPS: Dict[str, List[float]] = {
    "battery_capacity_kwh": [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0],
    "temperature_celsius": [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    "depot_charging_power_kw": [50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
    "terminus_charging_power_kw": [
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
        350.0,
        400.0,
        450.0,
        500.0,
        550.0,
        600.0,
    ],
}

CHARGE_TYPES: List[str] = ["DEP", "TERM"]

# When a factor is not the swept one, it is held at these defaults. The defaults
# match the constants used by the unswept swu_flow() so the "default" row of one
# sweep reproduces today's published results.
DEFAULTS: Dict[str, float] = {
    "battery_capacity_kwh": 600.0,
    "temperature_celsius": 10.0,
    "depot_charging_power_kw": 75.0,
    "terminus_charging_power_kw": 300.0,
}

METRIC_COLUMNS: List[str] = [
    "mean_energy_consumption_kwh_per_km",
    "vehicle_count",
    "electrified_termini",
    "terminus_chargers_utilized",
    "depot_chargers",
    "peak_depot_power_kw",
    "mean_rotation_duration_h",
    "mean_depot_arrival_soc",
]

FACTOR_LABEL: Dict[str, str] = {
    "battery_capacity_kwh": "Battery capacity [kWh]",
    "temperature_celsius": "Outside temperature [°C]",
    "depot_charging_power_kw": "Depot charging power [kW]",
    "terminus_charging_power_kw": "Terminus charging power [kW]",
}

OUTPUT_LABEL: Dict[str, str] = {
    "mean_energy_consumption_kwh_per_km": "Mean energy consumption [kWh/km]",
    "vehicle_count": "Vehicle count",
    "electrified_termini": "Electrified termini",
    "terminus_chargers_utilized": "Terminus chargers utilised",
    "depot_chargers": "Depot chargers",
    "peak_depot_power_kw": "Peak depot power [kW]",
    "mean_rotation_duration_h": "Mean rotation duration [h]",
    "mean_depot_arrival_soc": "Mean depot-arrival SoC",
}

# Cap Dask parallelism. On a 64-core box, the LocalCluster default of one
# worker per core overwhelms the shared on-disk caches (eflips altitude cache,
# Prefect result store, OSR routing cache) and the in-process global state of
# the SimPy-based depot simulation. Eight processes saturate the depot
# simulation cleanly while keeping shared-resource contention low.
MAX_DASK_WORKERS = 8


# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------

SENSITIVITY_OUTPUT_DIR = OUTPUT_BASE / "sensitivity"
SENSITIVITY_TABLE_PATH = SENSITIVITY_OUTPUT_DIR / "sensitivity_table.xlsx"
SENSITIVITY_CSV_PATH = SENSITIVITY_OUTPUT_DIR / "sensitivity_table.csv"
SENSITIVITY_PLOTS_DIR = SENSITIVITY_OUTPUT_DIR / "plots"
SENSITIVITY_LOGS_DIR = SENSITIVITY_OUTPUT_DIR / "logs"


# ---------------------------------------------------------------------------
# Cache directory layout
# ---------------------------------------------------------------------------
#
# Two design constraints are in tension:
#
#   1. The framework's cache key uses ``work_dir.absolute().name`` (a basename),
#      not the full path — see eflips/x/framework/__init__.py. So two iterations
#      with the same work_dir *basename* compute the same cache key. On a
#      cache hit, the framework checks whether the expected output file exists;
#      if it doesn't, ``PipelineStep.execute`` silently calls ``execute_impl``
#      directly to re-run the step. That means "shared cache key, different
#      paths" produces silent recomputation — caching looks alive in the
#      Prefect UI but is dead.
#
#   2. Parallel iterations writing to the same path race on SQLite locks and
#      blow up with "attempt to write a readonly database".
#
# Resolution: serialise the common phase across iterations (it's the same
# computation for many iterations anyway), then parallelise the variant phase.
# Common phases share a single work_dir per (battery, temperature, terminus_kw)
# tuple, so the framework's content-hash caching is fully effective there.
# Variant work_dirs are unique per (factor, value, charge_type) cell, so
# parallel writers never overlap.

CommonKey = Tuple[float, float, float]  # (battery_kwh, temperature_c, terminus_kw)


def _common_params_key(cfg: Dict[str, float]) -> CommonKey:
    """Common-phase identity tuple: only the params the common phase depends on."""
    return (
        cfg["battery_capacity_kwh"],
        cfg["temperature_celsius"],
        cfg["terminus_charging_power_kw"],
    )


def _common_cache_subdir(key: CommonKey) -> str:
    bat, temp, term = key
    return f"sensitivity/common/b{int(bat)}_t{int(temp)}_c{int(term)}"


def _variant_cache_subdir(factor: str, value: float, charge_type: str) -> str:
    return f"sensitivity/variant/{factor}/{int(value)}/{charge_type}"


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _peak_depot_power_kw(context: PipelineContext) -> float:
    """Max instantaneous summed charging power across all depot charging areas.

    This is the grid-connection peak demand, **not** the count of chargers.
    PowerAndOccupancyAnalyzer's ``power`` column is already the sum across the
    given areas, so we run it once with all charging-area IDs and take ``.max()``.
    Summing per-area peaks separately would double-count if areas peak at
    different times.
    """
    with context.get_session() as session:
        charging_area_ids = [
            a.id
            for depot in session.query(Depot).all()
            for a in depot.areas
            if any(p.electric_power is not None and p.electric_power > 0 for p in a.processes)
        ]
    if not charging_area_ids:
        return 0.0
    ctx = PipelineContext(
        work_dir=context.work_dir,
        params={"PowerAndOccupancyAnalyzer.area_id": charging_area_ids},
        current_db=context.current_db,
    )
    df = cast(pd.DataFrame, PowerAndOccupancyAnalyzer().execute(context=ctx))
    if df.empty or "power" not in df.columns:
        return 0.0
    return float(df["power"].max())


def _rotation_metrics(context: PipelineContext) -> Dict[str, float]:
    """Mean rotation duration [h] and mean depot-arrival SoC across all rotations."""
    with context.get_session() as session:
        depot_station_ids = {d.station_id for d in session.query(Depot).all()}
        rotations = session.query(Rotation).options(joinedload(Rotation.trips)).all()
        durations_h: List[float] = []
        end_socs: List[float] = []
        for rot in rotations:
            trips_sorted = sorted(rot.trips, key=lambda t: t.departure_time)
            if not trips_sorted:
                continue
            duration_s = (
                trips_sorted[-1].arrival_time - trips_sorted[0].departure_time
            ).total_seconds()
            durations_h.append(duration_s / 3600.0)

            end_station_id = trips_sorted[-1].route.arrival_station_id
            if end_station_id not in depot_station_ids:
                continue

            last_drive = (
                session.query(Event)
                .filter(
                    Event.vehicle_id == rot.vehicle_id,
                    Event.event_type == EventType.DRIVING,
                    Event.time_end <= trips_sorted[-1].arrival_time,
                )
                .order_by(Event.time_end.desc())
                .first()
            )
            if last_drive is not None:
                end_socs.append(float(last_drive.soc_end))

    mean_duration = float(np.mean(durations_h)) if durations_h else float("nan")
    mean_arrival = float(np.mean(end_socs)) if end_socs else float("nan")
    return {
        "mean_rotation_duration_h": mean_duration,
        "mean_depot_arrival_soc": mean_arrival,
    }


def _extract_metrics(final_db: Path, work_dir: Path, charge_type: str) -> Dict[str, float]:
    """Run all analyzers on a finished variant DB and return a flat metric dict.

    The keys match :data:`METRIC_COLUMNS`. Both the ScenarioComparisonAnalyzer
    and the EnergyConsumptionByVehicleTypeAnalyzer return one row per vehicle
    type; SWU has one vehicle type, so we read ``.iloc[0]`` from each.
    """
    context = PipelineContext(
        work_dir=work_dir,
        params={
            "ScenarioComparisonAnalyzer.scenario_name": charge_type,
            "EnergyConsumptionByVehicleTypeAnalyzer.scenario_name": charge_type,
        },
        current_db=final_db,
    )

    comp_df = cast(pd.DataFrame, ScenarioComparisonAnalyzer().execute(context=context))
    energy_df = cast(
        pd.DataFrame, EnergyConsumptionByVehicleTypeAnalyzer().execute(context=context)
    )
    comp = comp_df.iloc[0]
    mean_consumption = (
        float(energy_df["avg_consumption_kwh_per_km"].iloc[0])
        if not energy_df.empty
        else float("nan")
    )
    rot_metrics = _rotation_metrics(context)

    return {
        "mean_energy_consumption_kwh_per_km": mean_consumption,
        "vehicle_count": int(comp["total_vehicles"]),
        "electrified_termini": int(comp["electrified_termini"]),
        "terminus_chargers_utilized": int(comp["terminus_chargers_utilized"]),
        "depot_chargers": int(comp["depot_chargers"]),
        "peak_depot_power_kw": _peak_depot_power_kw(context),
        "mean_rotation_duration_h": rot_metrics["mean_rotation_duration_h"],
        "mean_depot_arrival_soc": rot_metrics["mean_depot_arrival_soc"],
    }


# ---------------------------------------------------------------------------
# Row assembly + crash handling
# ---------------------------------------------------------------------------


def _result_row(
    *,
    factor: str,
    factor_value: float,
    charge_type: str,
    status: str,
    error_short: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Assemble a single sensitivity-table row.

    Successful rows pass ``metrics``; failed rows leave it ``None`` and provide
    ``error_short`` instead. Metric columns are always present (NaN on failure)
    so concatenating ok and error rows produces a rectangular DataFrame.
    """
    row: Dict[str, Any] = {
        "factor": factor,
        "factor_value": factor_value,
        "charge_type": charge_type,
        "status": status,
        "error_short": error_short,
    }
    for col in METRIC_COLUMNS:
        row[col] = (metrics or {}).get(col, float("nan"))
    return pd.DataFrame([row])


def _log_failure(factor: str, value: float, charge_type: str, exc: BaseException) -> None:
    """Write full traceback + provenance to OUTPUT_BASE/sensitivity/logs/."""
    SENSITIVITY_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SENSITIVITY_LOGS_DIR / f"{factor}__{value}__{charge_type}.log"
    with log_path.open("w") as f:
        f.write(f"factor={factor} value={value} charge_type={charge_type}\n")
        f.write(f"timestamp={datetime.now().isoformat()}\n\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)


# ---------------------------------------------------------------------------
# Per-iteration task
# ---------------------------------------------------------------------------


def _run_variant(
    charge_type: str,
    common_db: Path,
    cfg: Dict[str, float],
    variant_root: str,
) -> Tuple[Path, Path]:
    """Run the DEP or TERM variant and return ``(final_db, work_dir)``."""
    if charge_type == "DEP":
        final_db = run_depot_variant(
            common_db=common_db,
            enable_plots=False,
            depot_charging_power_kw=cfg["depot_charging_power_kw"],
            cache_subdir=f"{variant_root}/depot",
            output_subdir=None,
        )
        return final_db, CACHE_BASE / f"{variant_root}/depot"
    if charge_type == "TERM":
        final_db = run_opportunity_variant(
            common_db=common_db,
            enable_plots=False,
            depot_charging_power_kw=cfg["depot_charging_power_kw"],
            terminus_charging_power_kw=cfg["terminus_charging_power_kw"],
            cache_subdir=f"{variant_root}/opportunity",
            output_subdir=None,
        )
        return final_db, CACHE_BASE / f"{variant_root}/opportunity"
    raise ValueError(f"Unknown charge_type: {charge_type!r}")


@task(name="sensitivity-iteration", retries=0)
def run_one(
    factor: str,
    value: float,
    charge_type: str,
    common_db: Path,
) -> pd.DataFrame:
    """Run the variant phase for one ``(factor, value, charge_type)`` cell.

    The common phase is precomputed by the parent flow and passed in as
    ``common_db``. Only the variant phase + analyzers run here, on a Dask
    worker. On any exception, a traceback is written to the sensitivity
    ``logs/`` directory and a NaN-filled error row is returned so that the
    sweep as a whole continues.
    """
    # Each Dask worker is a fresh Python process; re-apply the kv_cache patch
    # so workers don't deadlock on the shared altitude cache.
    _patch_kv_cache_for_parallel_access()

    cfg = {**DEFAULTS, factor: value}
    variant_root = _variant_cache_subdir(factor, value, charge_type)
    try:
        final_db, work_dir = _run_variant(charge_type, common_db, cfg, variant_root)
        metrics = _extract_metrics(final_db, work_dir, charge_type)
        return _result_row(
            factor=factor,
            factor_value=value,
            charge_type=charge_type,
            status="ok",
            metrics=metrics,
        )
    except Exception as exc:
        logger.warning(
            "Sensitivity iteration failed: factor=%s value=%s charge_type=%s — %s",
            factor,
            value,
            charge_type,
            exc,
        )
        _log_failure(factor, value, charge_type, exc)
        return _result_row(
            factor=factor,
            factor_value=value,
            charge_type=charge_type,
            status="error",
            error_short=f"{type(exc).__name__}: {str(exc)[:200]}",
        )


# ---------------------------------------------------------------------------
# Parent sweep flow
# ---------------------------------------------------------------------------


def _iter_sweep_cells() -> List[Tuple[str, float, str]]:
    """All ``(factor, value, charge_type)`` triples the sweep will iterate over.

    Depot rows are skipped for the ``terminus_charging_power_kw`` sweep because
    depot-only scenarios never use terminus chargers — those values would be
    identical to the default-power row and add no information.
    """
    cells: List[Tuple[str, float, str]] = []
    for factor, values in SWEEPS.items():
        for value in values:
            for charge_type in CHARGE_TYPES:
                if charge_type == "DEP" and factor == "terminus_charging_power_kw":
                    continue
                cells.append((factor, value, charge_type))
    return cells


def _precompute_common_dbs(pre_common_db: Path) -> Dict[CommonKey, Path]:
    """Materialise every unique parameterised common DB the sweep will need.

    Run serially: parallel workers would otherwise race on shared work_dir
    output paths, and the framework's content-hash caching makes the
    serialisation harmless across re-runs.
    """
    common_dbs: Dict[CommonKey, Path] = {}
    for factor, values in SWEEPS.items():
        for value in values:
            cfg = {**DEFAULTS, factor: value}
            key = _common_params_key(cfg)
            if key in common_dbs:
                continue
            common_dbs[key] = run_common_phase(
                battery_capacity_kwh=key[0],
                charging_curve=[[0.0, key[2]], [1.0, key[2]]],
                temperature_celsius=key[1],
                cache_subdir=_common_cache_subdir(key),
                pre_common_db=pre_common_db,
            )
    return common_dbs


@flow(
    name="SWU Sensitivity Sweep",
    task_runner=DaskTaskRunner(  # type: ignore[arg-type]
        cluster_kwargs={"n_workers": MAX_DASK_WORKERS, "threads_per_worker": 1}
    ),
)
def swu_sensitivity_sweep() -> pd.DataFrame:
    """Run the full sweep: common phases serial, variant phases parallel on Dask."""
    SENSITIVITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1 — Pre-common (GTFS ingest + station merge + cleanup). Parameter-
    # independent, so it runs exactly once for the whole sweep.
    pre_common_db = run_pre_common_phase(cache_subdir="sensitivity/pre_common")
    logger.info("Pre-common database: %s", pre_common_db)

    # Phase 2 — Precompute every unique parameterised common-phase config.
    common_dbs = _precompute_common_dbs(pre_common_db)
    logger.info("Precomputed %d unique parameterised common databases", len(common_dbs))

    # Phase 3 — Fan out variant iterations to Dask workers.
    futures = [
        run_one.submit(
            factor,
            value,
            charge_type,
            common_dbs[_common_params_key({**DEFAULTS, factor: value})],
        )
        for factor, value, charge_type in _iter_sweep_cells()
    ]
    rows = [f.result() for f in futures]
    table = pd.concat([r for r in rows if not r.empty], ignore_index=True)

    table.to_excel(SENSITIVITY_TABLE_PATH, index=False)
    table.to_csv(SENSITIVITY_CSV_PATH, index=False)

    n_total = len(table)
    n_failed = int((table["status"] == "error").sum())
    logger.info(
        "Sensitivity sweep complete: %d iterations (%d failed). Table: %s. Logs: %s",
        n_total,
        n_failed,
        SENSITIVITY_TABLE_PATH,
        SENSITIVITY_LOGS_DIR,
    )
    return table


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_fig(fig: Figure, output_dir: Path, basename: str, dpi: int = 300) -> None:
    """Save a matplotlib figure as both PDF and PNG, then close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{basename}.pdf")
    fig.savefig(output_dir / f"{basename}.png", dpi=dpi)
    plt.close(fig)


def _factor_applies_to_dep(factor: str) -> bool:
    """Depot-only scenarios are invariant to terminus charging power."""
    return factor != "terminus_charging_power_kw"


def _charge_types_for_factor(factor: str) -> List[str]:
    return ["DEP", "TERM"] if _factor_applies_to_dep(factor) else ["TERM"]


# Metrics that are only meaningful under terminus charging — DEP is structurally
# zero and would render as a misleading flat panel.
_TERM_ONLY_METRICS = {"electrified_termini"}


def _charge_types_for_plot(factor: str, metric: str) -> List[str]:
    types = _charge_types_for_factor(factor)
    if metric in _TERM_ONLY_METRICS:
        types = [c for c in types if c == "TERM"]
    return types


def _padded_limits(values: List[float], pad_frac: float = 0.05) -> Tuple[float, float]:
    """Min/max of ``values`` widened by a fraction of the range (or a small floor)."""
    if not values:
        return 0.0, 1.0
    lo, hi = float(min(values)), float(max(values))
    pad = (hi - lo) * pad_frac if hi > lo else max(abs(hi), 1.0) * 0.05
    return lo - pad, hi + pad


def _final_variant_db_path(factor: str, value: float, charge_type: str) -> Optional[Path]:
    """Path to the cached final-simulation DB for one iteration, or ``None``.

    Returns ``None`` when the iteration failed before ``Simulation`` completed
    (so no ``step_*_Simulation.db`` exists in the variant work_dir).
    """
    subdir = "depot" if charge_type == "DEP" else "opportunity"
    work_dir = CACHE_BASE / _variant_cache_subdir(factor, value, charge_type) / subdir
    if not work_dir.exists():
        return None
    candidates = sorted(work_dir.glob("step_*_Simulation.db"))
    return candidates[-1] if candidates else None


def _read_arrival_socs(factor: str, value: float, charge_type: str) -> List[float]:
    """Per-rotation depot-arrival SoCs from a cached variant DB.

    Returns an empty list when the cache is missing — typically because the
    iteration failed (e.g. cold-temperature TERM infeasibility) and never
    wrote a Simulation step. The SoC is read from the last DRIVING event of
    each rotation whose final trip terminates at a depot.
    """
    final_db = _final_variant_db_path(factor, value, charge_type)
    if final_db is None:
        return []
    context = PipelineContext(work_dir=final_db.parent, current_db=final_db)
    socs: List[float] = []
    with context.get_session() as session:
        depot_station_ids = {d.station_id for d in session.query(Depot).all()}
        rotations = session.query(Rotation).options(joinedload(Rotation.trips)).all()
        for rot in rotations:
            trips_sorted = sorted(rot.trips, key=lambda t: t.departure_time)
            if not trips_sorted:
                continue
            if trips_sorted[-1].route.arrival_station_id not in depot_station_ids:
                continue
            last_drive = (
                session.query(Event)
                .filter(
                    Event.vehicle_id == rot.vehicle_id,
                    Event.event_type == EventType.DRIVING,
                    Event.time_end <= trips_sorted[-1].arrival_time,
                )
                .order_by(Event.time_end.desc())
                .first()
            )
            if last_drive is not None:
                socs.append(float(last_drive.soc_end))
    return socs


def plot_per_factor(table: pd.DataFrame, output_dir: Path) -> None:
    """For each (factor, output) pair, render a print-sized two-panel line plot.

    Left panel: DEP. Right panel: TERM. The DEP panel is omitted entirely for
    factors where depot-only is invariant (terminus charging power) — drawing a
    constant line there would be misleading. The two panels share x and y
    limits (computed across both) so they can be visually compared.
    """
    configure_latex_plotting()
    palette = sns.color_palette("Set2")
    color_by_ctype = {"DEP": palette[0], "TERM": palette[1]}

    for factor in SWEEPS:
        for output_col in METRIC_COLUMNS:
            sub = table[(table["factor"] == factor) & (table["status"] == "ok")]
            if sub.empty:
                continue

            charge_types_to_plot = _charge_types_for_plot(factor, output_col)
            if not charge_types_to_plot:
                continue
            n_panels = len(charge_types_to_plot)
            sub_for_lims = sub[sub["charge_type"].isin(charge_types_to_plot)]
            xlim = _padded_limits(sub_for_lims["factor_value"].tolist())
            ylim = _padded_limits(sub_for_lims[output_col].tolist())

            fig, axes_obj = plt.subplots(
                1,
                n_panels,
                figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH),
                layout="constrained",
            )
            # plt.subplots returns a scalar Axes when n=1; normalise to list.
            axes = list(axes_obj) if n_panels > 1 else [axes_obj]

            for ax, ctype in zip(axes, charge_types_to_plot):
                cell = sub[sub["charge_type"] == ctype].sort_values("factor_value")
                ax.plot(
                    cell["factor_value"].values,
                    cell[output_col].values,
                    marker="o",
                    color=color_by_ctype[ctype],
                    linewidth=1.0,
                    markersize=3,
                )
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_title(ctype)
                ax.set_xlabel(FACTOR_LABEL[factor])
            axes[0].set_ylabel(OUTPUT_LABEL[output_col])

            _save_fig(fig, output_dir, f"{output_col}__vs__{factor}")


def plot_arrival_soc_hist2d(table: pd.DataFrame, output_dir: Path, bins: int = 10) -> None:
    """Per-factor 2D histogram of depot-arrival SoCs (one panel per charge_type).

    Complements ``mean_depot_arrival_soc`` in :func:`plot_per_factor` by showing
    the *distribution* of per-rotation arrival SoCs rather than the mean.
    The SoCs are pulled from the cached variant simulation DBs.

    Both panels share x/y limits and a single color scale so the DEP and TERM
    distributions can be compared directly.
    """
    configure_latex_plotting()

    for factor in SWEEPS:
        sub = table[(table["factor"] == factor) & (table["status"] == "ok")]
        if sub.empty:
            continue

        charge_types_to_plot = _charge_types_for_factor(factor)

        # Collect (factor_value, soc) pairs per charge_type, one (x, y) tuple
        # per rotation.
        points: Dict[str, Tuple[List[float], List[float]]] = {}
        for ctype in charge_types_to_plot:
            xs: List[float] = []
            ys: List[float] = []
            for _, row in sub[sub["charge_type"] == ctype].iterrows():
                fv = float(row["factor_value"])
                socs = _read_arrival_socs(factor, fv, ctype)
                xs.extend([fv] * len(socs))
                ys.extend(socs)
            points[ctype] = (xs, ys)

        all_x = [v for xs, _ in points.values() for v in xs]
        all_y = [v for _, ys in points.values() for v in ys]
        if not all_x or not all_y:
            continue
        xlim = _padded_limits(all_x)
        ylim = _padded_limits(all_y)

        # Pre-compute histograms with the shared range so we can pick a global
        # color-scale max across both panels.
        hist_by_ctype: Dict[str, np.ndarray] = {}
        for ctype in charge_types_to_plot:
            xs, ys = points[ctype]
            if not xs:
                hist_by_ctype[ctype] = np.zeros((bins, bins))
                continue
            counts, _, _ = np.histogram2d(xs, ys, bins=bins, range=[list(xlim), list(ylim)])
            hist_by_ctype[ctype] = counts

        vmax = max((h.max() for h in hist_by_ctype.values()), default=0)
        if vmax <= 0:
            continue

        n_panels = len(charge_types_to_plot)
        fig, axes_obj = plt.subplots(
            1,
            n_panels,
            figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH),
            layout="constrained",
        )
        axes = list(axes_obj) if n_panels > 1 else [axes_obj]

        # Hand-rolled imshow rather than ax.hist2d so DEP and TERM share the
        # same color normalisation. ``counts.T`` because hist2d's first axis is
        # x but imshow's first axis is rows (y).
        from matplotlib.colors import LogNorm  # local: only needed here

        norm = LogNorm(vmin=1.0, vmax=float(vmax))
        cmap = plt.get_cmap("Greys")
        mappable = None
        for ax, ctype in zip(axes, charge_types_to_plot):
            counts = hist_by_ctype[ctype]
            display = np.where(counts > 0, counts, np.nan)
            mappable = ax.imshow(
                display.T,
                extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(ctype)
            ax.set_xlabel(FACTOR_LABEL[factor])
        axes[0].set_ylabel("Depot arrival SoC")
        if mappable is not None:
            fig.colorbar(mappable, ax=axes, label="Rotations / bin")

        _save_fig(fig, output_dir, f"depot_arrival_soc_hist__vs__{factor}")


def plot_vs_energy_consumption(table: pd.DataFrame, output_dir: Path) -> None:
    """Line plot of every metric except mean energy itself against mean energy.

    Treats ``mean_energy_consumption_kwh_per_km`` as if it were a swept factor
    so the per-factor view extends to the dominant emergent output. Only the
    temperature sweep is used — the other three factors don't influence
    consumption, so pooling them would stack many rows at the default
    temperature's single energy value. Layout matches :func:`plot_per_factor`:
    two panels (DEP, TERM) sharing x and y limits.
    """
    configure_latex_plotting()
    palette = sns.color_palette("Set2")
    color_by_ctype = {"DEP": palette[0], "TERM": palette[1]}
    energy_col = "mean_energy_consumption_kwh_per_km"

    ok = table[(table["status"] == "ok") & (table["factor"] == "temperature_celsius")]
    if ok.empty:
        return

    for output_col in METRIC_COLUMNS:
        if output_col == energy_col:
            continue
        allowed = _charge_types_for_plot("temperature_celsius", output_col)
        charge_types_to_plot = [c for c in allowed if c in ok["charge_type"].unique()]
        if not charge_types_to_plot:
            continue

        ok_for_lims = ok[ok["charge_type"].isin(charge_types_to_plot)]
        xlim = _padded_limits(ok_for_lims[energy_col].tolist())
        ylim = _padded_limits(ok_for_lims[output_col].tolist())

        n_panels = len(charge_types_to_plot)
        fig, axes_obj = plt.subplots(
            1,
            n_panels,
            figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH),
            layout="constrained",
        )
        axes = list(axes_obj) if n_panels > 1 else [axes_obj]

        for ax, ctype in zip(axes, charge_types_to_plot):
            cell = ok[ok["charge_type"] == ctype].sort_values(energy_col)
            ax.plot(
                cell[energy_col].values,
                cell[output_col].values,
                marker="o",
                color=color_by_ctype[ctype],
                linewidth=1.0,
                markersize=3,
            )
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(ctype)
            ax.set_xlabel(OUTPUT_LABEL[energy_col])
        axes[0].set_ylabel(OUTPUT_LABEL[output_col])

        _save_fig(fig, output_dir, f"{output_col}__vs__mean_energy_consumption")


def plot_factor_grid(table: pd.DataFrame, output_dir: Path) -> None:
    """One big seaborn grid (metric × factor), DEP/TERM as hue. Screen-sized."""
    ok_table = table[table["status"] == "ok"].copy()
    if ok_table.empty:
        logger.warning("plot_factor_grid: no successful iterations to plot.")
        return

    melted = ok_table.melt(
        id_vars=["factor", "factor_value", "charge_type"],
        value_vars=METRIC_COLUMNS,
        var_name="metric",
        value_name="value",
    )
    # Apply human-readable names for the facet labels.
    melted["factor"] = melted["factor"].map(FACTOR_LABEL)
    melted["metric"] = melted["metric"].map(OUTPUT_LABEL)

    factor_order = [FACTOR_LABEL[k] for k in SWEEPS]
    metric_order = [OUTPUT_LABEL[k] for k in METRIC_COLUMNS]

    sns.set_theme(style="whitegrid", context="notebook")
    g = sns.relplot(
        data=melted,
        x="factor_value",
        y="value",
        row="metric",
        col="factor",
        row_order=metric_order,
        col_order=factor_order,
        hue="charge_type",
        style="charge_type",
        kind="line",
        marker="o",
        facet_kws={"sharex": False, "sharey": "row", "margin_titles": True},
        palette={"DEP": "#d95f02", "TERM": "#1b9e77"},
        height=2.0,
        aspect=1.3,
    )
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("", "")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    g.fig.set_size_inches(16, 18)
    output_dir.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(output_dir / "factor_grid.pdf")
    g.fig.savefig(output_dir / "factor_grid.png", dpi=200)
    plt.close(g.fig)
    # Reset rc defaults so subsequent matplotlib calls aren't tainted by the seaborn theme.
    sns.reset_orig()


def plot_all(table_path: Path = SENSITIVITY_TABLE_PATH) -> None:
    """Load the sensitivity table from disk and write all plots."""
    if not table_path.exists():
        raise FileNotFoundError(
            f"Sensitivity table not found at {table_path}. Run 'swu_sensitivity run' first."
        )
    table = pd.read_excel(table_path)
    plot_per_factor(table, SENSITIVITY_PLOTS_DIR)
    plot_vs_energy_consumption(table, SENSITIVITY_PLOTS_DIR)
    plot_arrival_soc_hist2d(table, SENSITIVITY_PLOTS_DIR)
    plot_factor_grid(table, SENSITIVITY_PLOTS_DIR)
    logger.info("Plots written to %s", SENSITIVITY_PLOTS_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SWU sensitivity sweep")
    parser.add_argument(
        "command",
        choices=["run", "plot", "all"],
        help="run = execute the sweep; plot = generate figures from existing table; all = both",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command in ("run", "all"):
        swu_sensitivity_sweep()
    if args.command in ("plot", "all"):
        plot_all()


if __name__ == "__main__":
    main()
