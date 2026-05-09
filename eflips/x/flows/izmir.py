#!/usr/bin/env python3

"""
Izmir (IZBB) flow with three input-slicing variants and two charging scenarios.

Variants control how much GTFS data is ingested:

* ``"full"`` — entire Eshot GTFS feed for one week.
* ``"one_day"`` — full network for a single service day.
* ``"reduced_lines"`` — week-long subset of 12 hand-picked routes.

Scenarios (run for every variant via ``CopyCreator`` branching):

* **DEPOT** — depot-only charging (``ChargeType.DEPOT``).
* **TERMINUS** — opportunity charging via ``IntegratedScheduling`` +
  ``StationElectrification``. Battery sizes are NOT reduced (unlike the BVG
  TERM scenario).

Sibling files ``izmir_one_day.py`` / ``izmir_reduced_lines.py`` are thin
stubs that call ``main(variant=...)``.
"""
import argparse
import logging
import sys
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import pandas as pd
from eflips.model import ChargeType, Route
from prefect import flow
from sqlalchemy import event
from sqlalchemy.orm import Session


# eflips.opt bug workaround: when a Necessary depot ends up colocated with a
# passenger stop (OSR snap radius collapses both onto the same road node →
# zero-distance round-trip route), `write_optimization_results` builds the
# return route's AssocRouteStation list with the wrong order, which then
# trips eflips.model's `check_route_before_insert_or_update` validator.
# Both assoc_route_stations end up with elapsed_distance=0, sort is stable
# and preserves the wrong order. Fix at the session level: reorder assoc
# entries so the one whose station matches Route.departure_station is first.
@event.listens_for(Session, "before_flush")
def _fix_zero_distance_route_assoc_order(
    session: Session, flush_context: Any, instances: Any
) -> None:
    for obj in list(session.new) + list(session.dirty):
        if not isinstance(obj, Route):
            continue
        if obj.distance not in (0, 0.0):
            continue
        assocs = list(obj.assoc_route_stations)
        if len(assocs) != 2:
            continue
        if any(a.elapsed_distance != 0 for a in assocs):
            continue
        if assocs[0].station == obj.departure_station:
            continue
        # Wrong order — swap so the departure_station-matching entry is first.
        obj.assoc_route_stations = [assocs[1], assocs[0]]


# Add the project root to sys.path to allow importing from eflips.x
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eflips.x.flows import (
    generate_all_plots,
    run_steps,
    save_plot_to_files_in_output_dir,
)
from eflips.x.framework import PipelineContext, PipelineStep, ScenarioDisplayConfig
from eflips.x.steps.analyzers import (
    SchedulingEfficiencyAnalyzer,
    VehicleTypeDepotPlotAnalyzer,
)
from eflips.x.steps.analyzers.bvg_tools import (
    RevenueServiceTimelineAnalyzer,
    visualize_electrified_termini_map,
)
from eflips.x.steps.generators import CopyCreator, GTFSIngester
from eflips.x.steps.modifiers.bvg_tools import MergeStations
from eflips.x.steps.modifiers.general_utilities import AddTemperatures, RemoveUnusedData
from eflips.x.steps.modifiers.gtfs_utilities import (
    ConfigureVehicleTypes,
    LongDistanceVehicleType,
)
from eflips.x.steps.modifiers.scheduling import (
    DepotAssignment,
    InsufficientChargingTimeAnalyzer,
    IntegratedScheduling,
    StationElectrification,
    VehicleScheduling,
)
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation

logger = logging.getLogger(__name__)

VALID_VARIANTS = ("full", "one_day", "reduced_lines")

# Per-variant overrides. Keys absent here inherit the shared defaults below.
_VARIANT_CONFIG: Dict[str, Dict[str, Any]] = {
    "full": {
        "work_dir": Path("data/cache/eflips_izmir_full"),
        "gtfs_route_ids": None,
        "gtfs_duration": None,  # default = WEEK
        "simulation_days": 7,
        "depot_breaks": {
            "minimum_break_time": timedelta(minutes=0),
            "regular_break_time": timedelta(minutes=20),
            "maximum_break_time": timedelta(minutes=40),
        },
    },
    "one_day": {
        "work_dir": Path("data/cache/eflips_izmir_one_day"),
        "gtfs_route_ids": None,
        "gtfs_duration": "DAY",
        "simulation_days": 1,
        "depot_breaks": {},
    },
    "reduced_lines": {
        "work_dir": Path("data/cache/eflips_izmir_reduced_lines"),
        "gtfs_route_ids": [
            "53",
            "77",
            "78",
            "102",
            "125",
            "140",
            "147",
            "148",
            "154",
            "168",
            "240",
            "335",
        ],
        "gtfs_duration": None,
        "simulation_days": 7,
        "depot_breaks": {},
    },
}

# Coordinates are (longitude, latitude) — the order eflips.opt expects and
# PostGIS uses internally.
#
# The first 7 entries are Eshot's actual depots. The 9 "Necessary <region>"
# entries were derived by clustering rotations whose first AND last empty
# trips both exceeded 50 km in the simulated full-variant database — rural
# districts with no nearby Eshot depot, where the resulting >100 km dead-
# heads pushed many rotations into negative SoC. See cluster analysis in
# the design doc; coordinates are pax-trip centroids per cluster.
DEPOT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "Adatepe Depot",
        "depot_station": (27.1860, 38.3850),
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 300,
    },
    {
        "name": "Mersinli Depot",
        "depot_station": (27.1712, 38.4335),
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 300,
    },
    {
        "name": "Çiğli",
        "depot_station": (27.0704, 38.4846),
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 300,
    },
    {
        "name": "Urla",
        "depot_station": (26.7548, 38.3260),
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 150,
    },
    {
        "name": "Çakalburnu",
        "depot_station": (27.0636, 38.4049),
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 300,
    },
    {
        "name": "Bergama",
        "depot_station": (27.1180, 39.0839),
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 150,
    },
    {
        "name": "Gaziemir",
        "depot_station": (27.1199, 38.3481),
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 300,
    },
    # --- Necessary depots: derived from cluster analysis of long-deadhead
    # rotations. Each depot starts from the nearest existing passenger-served
    # station and is then offset just far enough that OSR returns a non-zero
    # depot↔stop deadhead distance — Routes have a positive-distance check
    # constraint and zero-length routes also break eflips.opt's assoc ordering.
    # Offsets were measured with /tmp/find_offset.py against the actual TU-Berlin
    # OSR instance.
    {
        "name": "Necessary Ödemiş",
        "depot_station": (28.089055, 38.182439),  # +200m N of Kurucuova Son Durak
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 150,
    },
    {
        "name": "Necessary Aliağa",
        "depot_station": (26.893710, 38.717496),  # +200m N of Kozbeyli Meydan
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 100,
    },
    {
        "name": "Necessary Torbalı",
        "depot_station": (27.584419, 38.250533),  # +200m N of Çamlıbel Son Durak
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 60,
    },
    {
        "name": "Necessary Menemen",
        "depot_station": (27.090870, 38.581830),  # +200m N of Asarlık Toki Evleri Son Durak
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 30,
    },
    {
        "name": "Necessary Tire",
        "depot_station": (27.743501, 38.096288),  # +200m N of Tire Terminal İçi
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 30,
    },
    {
        "name": "Necessary Karaburun",
        "depot_station": (26.588000, 38.436300),  # +200m N of Balıklıova
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 20,
    },
    {
        "name": "Necessary Bayındır",
        "depot_station": (27.500430, 38.107110),  # +500m N of Bülbüldere Son Durak
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 20,
    },
    {
        "name": "Necessary Sarıyurt",
        "depot_station": (27.710344, 38.285115),  # +200m N of Sarıyurt Giriş
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 15,
    },
    {
        "name": "Necessary Menderes",
        "depot_station": (27.123288, 38.215728),  # +200m N of Şehit Binbaşı Ercan Ortaokulu
        "vehicle_type": ["TEMSA_AE", "TEMSA_AE_LD"],
        "capacity": 10,
    },
]

SCENARIO_DISPLAY_CONFIG = ScenarioDisplayConfig(
    order=["DEPOT", "TERMINUS"],
    display_names={"DEPOT": "Depot Charging", "TERMINUS": "Terminus Charging"},
    baseline="DEPOT",
)


def _common_params(variant_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shared params for the ingest+vehicle-type+temperature stage."""
    params: Dict[str, Any] = {
        "log_level": "INFO",
        "GTFSIngester.agency_name": None,
        "GTFSIngester.bus_only": False,
        "AddTemperatures.temperature_celsius": 15.0,
        # TEMSA Avenue Electron. GTFSIngester produces a single placeholder
        # vehicle type ("default_bus") — rename and re-spec it here so
        # DepotAssignment can look it up by name_short.
        "ConfigureVehicleTypes.name": "TEMSA Avenue Electron",
        "ConfigureVehicleTypes.name_short": "TEMSA_AE",
        "ConfigureVehicleTypes.battery_capacity": 240,
        "ConfigureVehicleTypes.consumption": 1.5,
        "ConfigureVehicleTypes.charging_curve": [[0.0, 450.0], [1.0, 450.0]],
        "ConfigureVehicleTypes.empty_mass": 12500,
        "ConfigureVehicleTypes.allowed_mass": 18500,
        # LongDistanceVehicleType: 51 km threshold (down from 61). Adds a
        # second VehicleType (500 kWh, 1.2 kWh/km) and reassigns trips on
        # routes longer than the threshold.
        "LongDistanceVehicleType.long_distance_vehicle_threshold": 51.0,
        "LongDistanceVehicleType.battery_capacity": 500.0,
        "LongDistanceVehicleType.consumption": 1.2,
        "scenario_display_config": SCENARIO_DISPLAY_CONFIG,
    }
    if variant_cfg["gtfs_duration"] is not None:
        params["GTFSIngester.duration"] = variant_cfg["gtfs_duration"]
    if variant_cfg["gtfs_route_ids"] is not None:
        params["GTFSIngester.route_ids"] = variant_cfg["gtfs_route_ids"]
    return params


@flow(name="Izmir Common Pipeline", flow_run_name="Izmir {variant} - common")
def run_izmir_common_pipeline(variant: str) -> Path:
    """Common pipeline: GTFS ingest → MergeStations → cleanup → vehicle types.

    Returns the path to the post-common DB for branching via CopyCreator.
    """
    variant_cfg = _VARIANT_CONFIG[variant]
    work_dir: Path = variant_cfg["work_dir"] / "common"
    work_dir.mkdir(parents=True, exist_ok=True)

    params = _common_params(variant_cfg)

    gtfs_file = project_root / "data" / "input" / "GTFS" / "izmir.zip"
    steps: List[PipelineStep] = [
        GTFSIngester(input_files=[gtfs_file]),
        MergeStations(),
        RemoveUnusedData(),
        ConfigureVehicleTypes(),
        AddTemperatures(),
        LongDistanceVehicleType(),
    ]

    context = PipelineContext(work_dir=work_dir, params=params)
    run_steps(context=context, steps=steps)

    # Pre-scheduling outputs (revenue service timeline)
    analysis_dir = work_dir / "analysis"
    try:
        timeline = RevenueServiceTimelineAnalyzer().execute(context=context)
        timeline_copy = timeline.copy()
        timeline_copy.index = [idx.strftime("%Y-%m-%d %H:%M") for idx in timeline_copy.index]
        analysis_dir.mkdir(parents=True, exist_ok=True)
        timeline_copy.to_excel(analysis_dir / "revenue_service_timeline.xlsx", index=True)
        fig = RevenueServiceTimelineAnalyzer.visualize(timeline)
        save_plot_to_files_in_output_dir(fig, analysis_dir, "revenue_service_timeline")
    except Exception as exc:
        logger.warning(f"RevenueServiceTimelineAnalyzer failed: {exc}")

    assert context.current_db is not None
    return context.current_db


@flow(name="Izmir DEPOT Scenario", flow_run_name="Izmir {variant} - depot")
def run_izmir_depot_scenario(common_db: Path, variant: str) -> Tuple[Path, pd.DataFrame]:
    """DEPOT scenario: depot-only charging."""
    variant_cfg = _VARIANT_CONFIG[variant]
    work_dir = variant_cfg["work_dir"] / "depot"
    work_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "VehicleScheduling.charge_type": ChargeType.DEPOT,
        "VehicleScheduling.battery_margin": 0.2,
        "DepotAssignment.depot_config": DEPOT_CONFIGS,
        "Simulation.repetition_period": timedelta(days=variant_cfg["simulation_days"]),
        # Some rural rotations span the daily cycle boundary, so the
        # repeated-schedule simulation needs more vehicles than scheduled.
        # That's a real schedule property, not a bug — let it through.
        "Simulation.ignore_unstable_simulation": True,
        "SchedulingEfficiencyAnalyzer.scenario_name": "DEPOT",
        "scenario_display_config": SCENARIO_DISPLAY_CONFIG,
    }
    for break_key, break_val in variant_cfg["depot_breaks"].items():
        params[f"VehicleScheduling.{break_key}"] = break_val

    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[common_db]).execute(context=context)

    run_steps(
        context=context,
        steps=[VehicleScheduling(), DepotAssignment(), DepotGenerator(), Simulation()],
    )

    eff_df = cast(pd.DataFrame, SchedulingEfficiencyAnalyzer().execute(context=context))

    assert context.current_db is not None
    return context.current_db, eff_df


@flow(name="Izmir TERMINUS Scenario", flow_run_name="Izmir {variant} - terminus")
def run_izmir_terminus_scenario(common_db: Path, variant: str) -> Tuple[Path, pd.DataFrame]:
    """TERMINUS scenario: opportunity charging via IntegratedScheduling.

    Battery sizes are NOT reduced (deliberate departure from BVG TERM).
    """
    variant_cfg = _VARIANT_CONFIG[variant]
    work_dir = variant_cfg["work_dir"] / "terminus"
    work_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
        "VehicleScheduling.battery_margin": 0.1,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=0),
        "VehicleScheduling.maximum_schedule_duration": timedelta(hours=24),
        "IntegratedScheduling.max_iterations": 5,
        "InsufficientChargingTimeAnalyzer.charging_power_kw": 450.0,
        "StationElectrification.charging_power_kw": 450.0,
        "StationElectrification.max_stations_to_electrify": 999,
        "DepotAssignment.depot_config": DEPOT_CONFIGS,
        "Simulation.repetition_period": timedelta(days=variant_cfg["simulation_days"]),
        # Some rural rotations span the daily cycle boundary, so the
        # repeated-schedule simulation needs more vehicles than scheduled.
        # That's a real schedule property, not a bug — let it through.
        "Simulation.ignore_unstable_simulation": True,
        "SchedulingEfficiencyAnalyzer.scenario_name": "TERMINUS",
        "scenario_display_config": SCENARIO_DISPLAY_CONFIG,
    }

    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[common_db]).execute(context=context)

    # IntegratedScheduling rolls back its nested DepotAssignment calls, so
    # we re-run DepotAssignment afterwards.
    run_steps(context=context, steps=[IntegratedScheduling(), DepotAssignment()])

    insufficient = InsufficientChargingTimeAnalyzer().execute(context=context)
    if insufficient is not None:
        critical_ids = insufficient["rotation_ids"]
        soc_data = insufficient.get("soc_data", {})
        lines = []
        for rot_id in critical_ids:
            if rot_id in soc_data:
                soc_df, _spans, rot_start, rot_end = soc_data[rot_id]
                min_soc = float(soc_df["soc"].min())
                lines.append(
                    f"  Rotation {rot_id}: min SoC={min_soc:.3f}, window {rot_start} – {rot_end}"
                )
            else:
                lines.append(f"  Rotation {rot_id}: no SoC data available")
        raise RuntimeError(
            f"TERMINUS scenario infeasible: {len(critical_ids)} rotation(s) cannot "
            f"accumulate enough charge even with all termini electrified at 450 kW.\n"
            f"Infeasible rotations:\n" + "\n".join(lines)
        )

    run_steps(
        context=context,
        steps=[StationElectrification(), DepotGenerator(), Simulation()],
    )

    eff_df = cast(pd.DataFrame, SchedulingEfficiencyAnalyzer().execute(context=context))

    assert context.current_db is not None
    return context.current_db, eff_df


def _run_per_scenario_analysis(
    label: str, db_path: Path, work_dir: Path, params: Dict[str, Any]
) -> None:
    """Post-simulation per-scenario analysis: VehicleTypeDepotPlot + generate_all_plots."""
    analysis_dir = work_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    context = PipelineContext(work_dir=work_dir, params=params, current_db=db_path)

    # vehicle_km_by_depot_and_vehicle_type (post-simulation per user request)
    try:
        vt_depot = VehicleTypeDepotPlotAnalyzer()
        prepared = vt_depot.execute(context=context)
        prepared.to_excel(analysis_dir / "vehicle_km_by_depot_and_vehicle_type.xlsx", index=False)
        fig = vt_depot.visualize(prepared)
        save_plot_to_files_in_output_dir(fig, analysis_dir, "vehicle_km_by_depot_and_vehicle_type")
    except Exception as exc:
        logger.warning(f"[{label}] VehicleTypeDepotPlotAnalyzer failed: {exc}")

    # All standard analyzers (replaces the old hand-rolled per-rotation/per-vehicle loops).
    # include_videos=False per user instruction.
    try:
        generate_all_plots(
            context=context,
            output_dir=analysis_dir,
            include_videos=False,
            pre_simulation_only=False,
        )
    except Exception as exc:
        logger.warning(f"[{label}] generate_all_plots failed: {exc}")


@flow(name="Izmir (IZBB)", flow_run_name="Izmir {variant}")
def main(variant: str = "full") -> None:
    if variant not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")

    variant_cfg = _VARIANT_CONFIG[variant]
    work_dir: Path = variant_cfg["work_dir"]
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Working directory: {work_dir}")

    common_db = run_izmir_common_pipeline(variant)
    depot_db, depot_eff = run_izmir_depot_scenario(common_db, variant)

    # TERMINUS may be infeasible for variants whose long rural rotations can't
    # accumulate enough opportunity-charging time. Treat its failure as a real
    # finding (log + skip cross-scenario terminus outputs) rather than abort
    # the whole flow — DEPOT results are still useful on their own.
    term_db: Path | None = None
    term_eff: pd.DataFrame | None = None
    try:
        term_db, term_eff = run_izmir_terminus_scenario(common_db, variant)
    except Exception as exc:
        logger.warning(f"TERMINUS scenario failed and will be skipped: {exc}")

    # ---- Cross-scenario outputs ----
    cross_dir = work_dir / "analysis"
    cross_dir.mkdir(parents=True, exist_ok=True)

    # Combined scheduling-efficiency report
    try:
        if term_eff is not None:
            all_eff = pd.concat([depot_eff, term_eff], ignore_index=True)
        else:
            all_eff = depot_eff
        all_eff.to_excel(cross_dir / "scheduling_efficiency.xlsx", index=False)
        fig = SchedulingEfficiencyAnalyzer.visualize(all_eff)
        save_plot_to_files_in_output_dir(fig, cross_dir, "scheduling_efficiency")
        fig_hist = SchedulingEfficiencyAnalyzer.visualize_histogram(all_eff)
        save_plot_to_files_in_output_dir(fig_hist, cross_dir, "scheduling_efficiency_histogram")
    except Exception as exc:
        logger.warning(f"SchedulingEfficiency cross-scenario plots failed: {exc}")

    # Electrified termini map — only meaningful when TERMINUS ran. Function
    # expects keys "OU" and "TERM" for the split-marker palette; we map our
    # DEPOT/TERMINUS sessions onto those keys.
    if term_db is not None:
        try:
            with ExitStack() as stack:
                depot_ctx = PipelineContext(
                    work_dir=variant_cfg["work_dir"] / "depot", current_db=depot_db
                )
                term_ctx = PipelineContext(
                    work_dir=variant_cfg["work_dir"] / "terminus", current_db=term_db
                )
                sessions: Dict[str, Session] = {
                    "OU": stack.enter_context(depot_ctx.get_session()),
                    "TERM": stack.enter_context(term_ctx.get_session()),
                }
                fig = visualize_electrified_termini_map(sessions)
            save_plot_to_files_in_output_dir(fig, cross_dir, "electrified_termini_map")
        except Exception as exc:
            logger.warning(f"electrified_termini_map failed: {exc}")

    # Per-scenario post-simulation analysis (+ generate_all_plots)
    _run_per_scenario_analysis(
        "DEPOT",
        depot_db,
        variant_cfg["work_dir"] / "depot",
        params={"scenario_display_config": SCENARIO_DISPLAY_CONFIG, "log_level": "INFO"},
    )
    if term_db is not None:
        _run_per_scenario_analysis(
            "TERMINUS",
            term_db,
            variant_cfg["work_dir"] / "terminus",
            params={"scenario_display_config": SCENARIO_DISPLAY_CONFIG, "log_level": "INFO"},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Izmir (IZBB) pipeline flow")
    parser.add_argument(
        "--variant",
        choices=VALID_VARIANTS,
        default="full",
        help="Pipeline variant to run (default: full).",
    )
    args = parser.parse_args()
    main(variant=args.variant)
