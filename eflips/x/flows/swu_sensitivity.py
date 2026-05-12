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
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from eflips.model import Depot, Event, EventType, Rotation
from matplotlib.figure import Figure
from prefect import flow, task
from prefect_dask.task_runners import DaskTaskRunner
from sqlalchemy.orm import joinedload

from eflips.x.flows.swu_flow import (
    CACHE_BASE,
    OUTPUT_BASE,
    run_common_phase,
    run_depot_variant,
    run_opportunity_variant,
    run_pre_common_phase,
)
from eflips.x.framework import PipelineContext
from eflips.x.steps.analyzers.bvg_tools import (
    PLOT_HEIGHT_INCH,
    PLOT_WIDTH_INCH,
    ScenarioComparisonAnalyzer,
    configure_latex_plotting,
)
from eflips.x.steps.analyzers.output_analyzers import (
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


# ---------------------------------------------------------------------------
# Output / cache paths
# ---------------------------------------------------------------------------

SENSITIVITY_OUTPUT_DIR = OUTPUT_BASE / "sensitivity"
SENSITIVITY_TABLE_PATH = SENSITIVITY_OUTPUT_DIR / "sensitivity_table.xlsx"
SENSITIVITY_CSV_PATH = SENSITIVITY_OUTPUT_DIR / "sensitivity_table.csv"
SENSITIVITY_PLOTS_DIR = SENSITIVITY_OUTPUT_DIR / "plots"
SENSITIVITY_LOGS_DIR = SENSITIVITY_OUTPUT_DIR / "logs"


# ---------------------------------------------------------------------------
# Metric helpers
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


def _collect_row(
    final_db: Path,
    work_dir: Path,
    factor: str,
    value: float,
    charge_type: str,
) -> pd.DataFrame:
    """Run all the analyzers on the final database and assemble a single-row DataFrame."""
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
    peak_power_kw = _peak_depot_power_kw(context)
    rot_metrics = _rotation_metrics(context)

    # Both DataFrames have one row in the single-vehicle-type SWU case.
    comp = comp_df.iloc[0]
    mean_consumption = (
        float(energy_df["avg_consumption_kwh_per_km"].iloc[0])
        if not energy_df.empty
        else float("nan")
    )

    return pd.DataFrame(
        [
            {
                "factor": factor,
                "factor_value": value,
                "charge_type": charge_type,
                "status": "ok",
                "error_short": None,
                "mean_energy_consumption_kwh_per_km": mean_consumption,
                "vehicle_count": int(comp["total_vehicles"]),
                "electrified_termini": int(comp["electrified_termini"]),
                "terminus_chargers_utilized": int(comp["terminus_chargers_utilized"]),
                "depot_chargers": int(comp["depot_chargers"]),
                "peak_depot_power_kw": peak_power_kw,
                "mean_rotation_duration_h": rot_metrics["mean_rotation_duration_h"],
                "mean_depot_arrival_soc": rot_metrics["mean_depot_arrival_soc"],
            }
        ]
    )


# ---------------------------------------------------------------------------
# Crash handling
# ---------------------------------------------------------------------------


def _log_failure(factor: str, value: float, charge_type: str, exc: BaseException) -> None:
    """Write full traceback + provenance to OUTPUT_BASE/sensitivity/logs/."""
    SENSITIVITY_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SENSITIVITY_LOGS_DIR / f"{factor}__{value}__{charge_type}.log"
    with log_path.open("w") as f:
        f.write(f"factor={factor} value={value} charge_type={charge_type}\n")
        f.write(f"timestamp={datetime.now().isoformat()}\n\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)


def _failure_row(factor: str, value: float, charge_type: str, exc: BaseException) -> pd.DataFrame:
    """Build a row with NaN metrics and a short error summary."""
    row: Dict[str, Any] = {
        "factor": factor,
        "factor_value": value,
        "charge_type": charge_type,
        "status": "error",
        "error_short": f"{type(exc).__name__}: {str(exc)[:200]}",
    }
    for col in METRIC_COLUMNS:
        row[col] = float("nan")
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Cache directory layout
# ---------------------------------------------------------------------------
#
# Two design constraints are in tension:
#
#   1. The framework's cache key uses ``work_dir.absolute().name`` (a basename),
#      not the full path — see eflips/x/framework/__init__.py lines 338, 414, 501.
#      So two iterations with the same work_dir *basename* compute the same
#      cache key. On a cache hit, the framework checks whether the expected
#      output file exists; if it doesn't, ``PipelineStep.execute`` silently
#      calls ``execute_impl`` directly to re-run the step (lines 240-248).
#      That means "shared cache key, different paths" produces silent
#      recomputation — caching looks alive in the Prefect UI but is dead.
#
#   2. Parallel iterations writing to the same path race on SQLite locks and
#      blow up with "attempt to write a readonly database".
#
# Resolution: serialize the common phase across iterations (it's the same
# computation for many iterations anyway), then parallelize the variant phase.
# Common phases share a single work_dir per (battery, temp, charging_curve)
# tuple, so the framework's content-hash caching is fully effective there.
# Variant work_dirs are unique per (factor, value, charge_type) cell, so
# parallel writers never overlap.

CommonKey = Tuple[float, float, float]  # (battery_kwh, temperature_c, terminus_kw)


def _common_params_key(cfg: Dict[str, float]) -> CommonKey:
    """Common-phase identity tuple: only the params the common phase actually depends on."""
    return (
        cfg["battery_capacity_kwh"],
        cfg["temperature_celsius"],
        cfg["terminus_charging_power_kw"],
    )


def _common_cache_subdir(key: CommonKey) -> str:
    """Deterministic, unique cache subdir for one common-phase config.

    Same params → same path → real cache hit. Different params → different path.
    """
    bat, temp, term = key
    return f"sensitivity/common/b{int(bat)}_t{int(temp)}_c{int(term)}"


def _variant_cache_subdir(factor: str, value: float, charge_type: str) -> str:
    """Unique cache subdir for one variant iteration. No overlap with any other cell."""
    return f"sensitivity/variant/{factor}/{int(value)}/{charge_type}"


# ---------------------------------------------------------------------------
# Per-iteration task (variant phase only — common phase is precomputed)
# ---------------------------------------------------------------------------


@task(name="sensitivity-iteration", retries=0)
def run_one(
    factor: str,
    value: float,
    charge_type: str,
    common_db: Path,
) -> pd.DataFrame:
    """Run the variant phase for one (factor, factor_value, charge_type) cell.

    The common phase has already been precomputed by the parent flow and is
    passed in as ``common_db``. Only the variant phase + analyzers run here, on
    a Dask worker.

    On any exception, write a traceback to the sensitivity ``logs/`` directory
    and return a NaN-filled error row, so the sweep as a whole continues.
    """
    cfg = {**DEFAULTS, factor: value}
    variant_root = _variant_cache_subdir(factor, value, charge_type)
    try:
        if charge_type == "DEP":
            final_db = run_depot_variant(
                common_db=common_db,
                enable_plots=False,
                depot_charging_power_kw=cfg["depot_charging_power_kw"],
                cache_subdir=f"{variant_root}/depot",
                output_subdir=None,
            )
            work_dir = CACHE_BASE / f"{variant_root}/depot"
        elif charge_type == "TERM":
            final_db = run_opportunity_variant(
                common_db=common_db,
                enable_plots=False,
                depot_charging_power_kw=cfg["depot_charging_power_kw"],
                terminus_charging_power_kw=cfg["terminus_charging_power_kw"],
                cache_subdir=f"{variant_root}/opportunity",
                output_subdir=None,
            )
            work_dir = CACHE_BASE / f"{variant_root}/opportunity"
        else:
            raise ValueError(f"Unknown charge_type: {charge_type!r}")
        return _collect_row(final_db, work_dir, factor, value, charge_type)
    except Exception as e:
        logger.warning(
            "Sensitivity iteration failed: factor=%s value=%s charge_type=%s — %s",
            factor,
            value,
            charge_type,
            e,
        )
        _log_failure(factor, value, charge_type, e)
        return _failure_row(factor, value, charge_type, e)


# ---------------------------------------------------------------------------
# Parent sweep flow
# ---------------------------------------------------------------------------


@flow(name="SWU Sensitivity Sweep", task_runner=DaskTaskRunner())  # type: ignore[arg-type]
def swu_sensitivity_sweep() -> pd.DataFrame:
    """Run the full sweep: common phases serial, variant phases parallel on Dask.

    Depot rows are skipped for the terminus_charging_power_kw sweep because
    depot-only scenarios never use terminus chargers — those values would be
    identical to the default-power row and add no information.
    """
    SENSITIVITY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1 — Pre-common (GTFS ingest + station merge + cleanup). Parameter-
    # independent, so it runs exactly once for the whole sweep. This is the
    # expensive step we want to dodge across iterations.
    pre_common_db = run_pre_common_phase(cache_subdir="sensitivity/pre_common")
    logger.info("Pre-common database: %s", pre_common_db)

    # Phase 2 — Precompute every unique parameterised common-phase config
    # sequentially. Each call here is now cheap (CopyCreator + ConfigureVT +
    # AddTemperatures only) since the GTFS ingest is reused via pre_common_db.
    # Serialised because parallel workers would otherwise race on shared
    # work_dir output paths; the framework's content-hash caching makes the
    # serialisation harmless across re-runs.
    common_dbs: Dict[CommonKey, Path] = {}
    for factor, values in SWEEPS.items():
        for value in values:
            cfg = {**DEFAULTS, factor: value}
            ck = _common_params_key(cfg)
            if ck in common_dbs:
                continue
            common_dbs[ck] = run_common_phase(
                battery_capacity_kwh=ck[0],
                charging_curve=[[0.0, ck[2]], [1.0, ck[2]]],
                temperature_celsius=ck[1],
                cache_subdir=_common_cache_subdir(ck),
                pre_common_db=pre_common_db,
            )
    logger.info("Precomputed %d unique parameterised common databases", len(common_dbs))

    # Phase 3 — Fan out variant iterations to Dask workers.
    futures = []
    for factor, values in SWEEPS.items():
        for value in values:
            for charge_type in CHARGE_TYPES:
                if charge_type == "DEP" and factor == "terminus_charging_power_kw":
                    continue
                cfg = {**DEFAULTS, factor: value}
                futures.append(
                    run_one.submit(factor, value, charge_type, common_dbs[_common_params_key(cfg)])
                )

    rows: List[pd.DataFrame] = [f.result() for f in futures]
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


def plot_per_factor(table: pd.DataFrame, output_dir: Path) -> None:
    """For each (factor, output) pair, render a print-sized two-panel line plot.

    Left panel: DEP. Right panel: TERM. The DEP panel is omitted entirely for
    factors where depot-only is invariant (terminus charging power) — drawing a
    constant line there would be misleading.
    """
    configure_latex_plotting()
    palette = sns.color_palette("Set2")
    color_by_ctype = {"DEP": palette[0], "TERM": palette[1]}

    for factor in SWEEPS:
        for output_col in METRIC_COLUMNS:
            sub = table[(table["factor"] == factor) & (table["status"] == "ok")]
            if sub.empty:
                continue

            include_dep = _factor_applies_to_dep(factor)
            charge_types_to_plot = ["DEP", "TERM"] if include_dep else ["TERM"]
            n_panels = len(charge_types_to_plot)

            fig, axes_obj = plt.subplots(
                1,
                n_panels,
                figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH),
                layout="constrained",
                sharey=True,
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
                ax.set_title(ctype)
                ax.set_xlabel(FACTOR_LABEL[factor])
            axes[0].set_ylabel(OUTPUT_LABEL[output_col])

            _save_fig(fig, output_dir, f"{output_col}__vs__{factor}")


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
