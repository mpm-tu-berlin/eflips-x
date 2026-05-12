#!/usr/bin/env python3

"""
SWU (Ulm) GTFS Simulation Flow

Runs both DEPOT and OPPORTUNITY charging variants for SWU Verkehr GmbH
using the SWU GTFS feed. Exports a JSON scenario after depot assignment
(before terminus charger placement).

Usage:
    python -m eflips.x.flows.swu_flow [--plots]
"""

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from eflips.depot.api import SmartChargingStrategy  # type: ignore[import-untyped]
from eflips.model import ChargeType, Event, EventType, Line, Rotation, Trip, TripType
from geoalchemy2.shape import to_shape
from matplotlib.figure import Figure
from prefect import flow
from sqlalchemy.orm import Session

from eflips.x.flows import generate_all_plots, run_steps
from eflips.x.framework import (
    Analyzer,
    Modifier,
    PipelineContext,
    PipelineStep,
    ScenarioDisplayConfig,
)
from eflips.x.steps.analyzers.bvg_tools import (
    PLOT_HEIGHT_INCH,
    PLOT_WIDTH_INCH,
    RepresentativeVehicleSocAnalyzer,
    ScenarioComparisonAnalyzer,
    configure_latex_plotting,
    merge_scenario_comparisons,
)
from eflips.x.steps.analyzers.output_analyzers import SpecificEnergyConsumptionAnalyzer
from eflips.x.steps.generators import GTFSIngester, CopyCreator
from eflips.x.steps.modifiers.bvg_tools import MergeStations
from eflips.x.steps.modifiers.consumption_luts import ConsumptionLut
from eflips.x.steps.modifiers.general_utilities import AddTemperatures, RemoveUnusedData
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
# SWU-specific modifier: handle Line "E" (Einrückfahrten released to passengers)
# ---------------------------------------------------------------------------


class HandleOnlyETrips(Modifier):
    """Strip single-trip E-line rotations that VehicleScheduling left unembedded.

    After VehicleScheduling, some E-trip rotations contain exactly one PASSENGER
    trip on Line "E" and no other passenger service.  These rotations could not
    be merged with any regular-service rotation (their terminus has no matching
    departure in the same ±20 min window), so they are simply removed together
    with their depot EMPTY trips.

    The modifier logs how many rotations were stripped.
    """

    def __init__(self, code_version: str = "v1.0.1", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {}

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        e_line = session.query(Line).filter(Line.name_short == "E").first()
        if e_line is None:
            self.logger.info("Line 'E' not found; nothing to do")
            return

        e_route_ids = {r.id for r in e_line.routes}

        only_e_rotations = []
        for rotation in session.query(Rotation).all():
            passenger_trips = [t for t in rotation.trips if t.trip_type == TripType.PASSENGER]
            if len(passenger_trips) == 1 and passenger_trips[0].route_id in e_route_ids:
                only_e_rotations.append(rotation)

        for rotation in only_e_rotations:
            for trip in list(rotation.trips):
                for st in list(trip.stop_times):
                    session.delete(st)
                session.delete(trip)
            session.flush()
            session.delete(rotation)
        session.flush()

        self.logger.info(f"HandleOnlyETrips: stripped {len(only_e_rotations)} only-E rotations")


# ---------------------------------------------------------------------------
# SWU (Ulm) configuration — all values explicit, no implicit defaults
# ---------------------------------------------------------------------------

AGENCY_LABEL = "SWU Verkehr GmbH"
GTFS_FILE = PROJECT_ROOT / "data" / "input" / "GTFS" / "SWU.zip"
AGENCY_IDS: List[str] = ["1"]

BATTERY_CAPACITY_KWH = 600.0
CHARGING_CURVE: List[List[float]] = [[0.0, 300.0], [1.0, 300.0]]
TERMINUS_CHARGING_POWER_KW = 300.0

# 12m solo bus mass values (matching the EN values used in eflips/x/flows/bvg.py).
# allowed_mass follows the bvg_tools.py convention: empty_mass + 120 passengers * 68 kg.
EMPTY_MASS_KG = 17000.0
ALLOWED_MASS_KG = EMPTY_MASS_KG + 120 * 68

DEPOT_CONFIG: List[Dict[str, Any]] = [
    {
        "depot_station": (9.967748831666805, 48.39658695394624),  # (lon, lat)
        "name": "SWU Betriebshof",
        "vehicle_type": ["DEFAULT"],  # Matches ConfigureVehicleTypes.name_short
        "capacity": 9999,
    },
]

CACHE_BASE = PROJECT_ROOT / "data" / "cache" / "SWU" / "swu_verkehr_gmbh"
OUTPUT_BASE = PROJECT_ROOT / "data" / "output" / "SWU" / "swu_verkehr_gmbh"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _save_fig(fig: Figure, output_dir: Path, basename: str, dpi: int = 300) -> None:
    """Save a matplotlib figure as both PDF and PNG, then close it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{basename}.pdf")
    fig.savefig(output_dir / f"{basename}.png", dpi=dpi)
    plt.close(fig)


def _pick_representative_day(session: Session) -> tuple[datetime, datetime]:
    """Pick the 24-hour 03:00-to-03:00 (local) window with the most terminus-charging events.

    Returns the (start, end) bounds as Europe/Berlin–aware datetimes. The 03:00 cut matches
    the bus operator's notion of a "service day" so a vehicle's whole tour falls into one
    window even when it ends shortly after midnight.
    """
    import pytz

    tz = pytz.timezone("Europe/Berlin")
    rows = (
        session.query(Event.time_start)
        .filter(Event.event_type == EventType.CHARGING_OPPORTUNITY)
        .all()
    )
    if not rows:
        raise RuntimeError(
            "No CHARGING_OPPORTUNITY events in the simulation database — cannot select a "
            "representative day."
        )

    day_counts: Dict[Any, int] = defaultdict(int)
    for (time_start,) in rows:
        local = time_start.astimezone(tz) - timedelta(hours=3)
        day_counts[local.date()] += 1

    best_day = max(day_counts, key=lambda d: day_counts[d])
    day_start = tz.localize(datetime(best_day.year, best_day.month, best_day.day, 3, 0))
    day_end = day_start + timedelta(days=1)
    return day_start, day_end


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------


DEFAULT_TEMPERATURE_CELSIUS = 10.0
DEFAULT_DEPOT_CHARGING_POWER_KW = 75.0


@flow(name="SWU Pre-Common Phase", flow_run_name="SWU - pre-common")
def run_pre_common_phase(*, cache_subdir: str = "pre_common") -> Path:
    """Run the GTFS-only portion of the common phase.

    These three steps (GTFSIngester → MergeStations → RemoveUnusedData) depend
    only on the GTFS input file, not on any sweep parameter. Their output can
    be safely reused as the starting point for many parameterised common-phase
    runs via ``run_common_phase(pre_common_db=...)``.

    Kept in its own flow so the sensitivity sweep can compute it once and
    share it across every iteration, dodging the expensive GTFS ingest.
    """
    work_dir = CACHE_BASE / cache_subdir
    work_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "GTFSIngester.bus_only": True,
        "GTFSIngester.duration": "WEEK",
        "GTFSIngester.agency_ids": AGENCY_IDS,
    }
    steps: List[PipelineStep] = [
        GTFSIngester(input_files=[GTFS_FILE]),
        MergeStations(),
        RemoveUnusedData(),
    ]
    context = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps=steps, context=context)
    assert context.current_db is not None
    return context.current_db


@flow(name="SWU Common Phase", flow_run_name="SWU - common")
def run_common_phase(
    *,
    battery_capacity_kwh: float = BATTERY_CAPACITY_KWH,
    charging_curve: Optional[List[List[float]]] = None,
    temperature_celsius: float = DEFAULT_TEMPERATURE_CELSIUS,
    cache_subdir: str = "common",
    pre_common_db: Optional[Path] = None,
) -> Path:
    """Run the common pipeline phase (ingest → merge → cleanup → configure).

    When ``pre_common_db`` is supplied, the three GTFS-only steps are skipped and
    we instead start from ``CopyCreator(input_files=[pre_common_db])`` — only
    ``ConfigureVehicleTypes`` and ``AddTemperatures`` run on top. This is the
    fast path used by the sensitivity sweep. The canonical ``swu_flow()`` call
    leaves it ``None`` and runs all five steps.

    Note: ``AddTemperatures`` is not idempotent (it appends a Temperatures row
    rather than replacing), so we cannot simply layer it onto an existing common
    DB that has already had temperatures added — we must always start from a
    pre-common DB that has *not* been through ``AddTemperatures``.
    """
    if charging_curve is None:
        charging_curve = CHARGING_CURVE
    work_dir = CACHE_BASE / cache_subdir
    work_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "GTFSIngester.bus_only": True,
        "GTFSIngester.duration": "WEEK",
        "GTFSIngester.agency_ids": AGENCY_IDS,
        "ConfigureVehicleTypes.vehicle_type_names": ["default"],
        "ConfigureVehicleTypes.battery_capacity": battery_capacity_kwh,
        "ConfigureVehicleTypes.consumption": ConsumptionLut.NOR_BUS_12M,
        "ConfigureVehicleTypes.charging_curve": charging_curve,
        "ConfigureVehicleTypes.empty_mass": EMPTY_MASS_KG,
        "ConfigureVehicleTypes.allowed_mass": ALLOWED_MASS_KG,
        "ConfigureVehicleTypes.name_short": "DEFAULT",
        "AddTemperatures.temperature_celsius": temperature_celsius,
    }

    context = PipelineContext(work_dir=work_dir, params=params)
    if pre_common_db is None:
        steps: List[PipelineStep] = [
            GTFSIngester(input_files=[GTFS_FILE]),
            MergeStations(),
            RemoveUnusedData(),
            ConfigureVehicleTypes(),
            AddTemperatures(),
        ]
    else:
        CopyCreator(input_files=[pre_common_db]).execute(context=context)
        steps = [ConfigureVehicleTypes(), AddTemperatures()]
    run_steps(steps=steps, context=context)

    assert context.current_db is not None
    return context.current_db


@flow(name="SWU Depot Variant", flow_run_name="SWU - depot")
def run_depot_variant(
    common_db: Path,
    enable_plots: bool,
    *,
    depot_charging_power_kw: float = DEFAULT_DEPOT_CHARGING_POWER_KW,
    cache_subdir: str = "depot",
    output_subdir: Optional[str] = "depot",
) -> Path:
    """Run the DEPOT charging variant. Returns the path to the final database.

    Pass ``output_subdir=None`` to suppress all per-run output writes (used by the
    sensitivity sweep, where 55 iterations would otherwise overwrite each other).
    """
    work_dir = CACHE_BASE / cache_subdir
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir: Optional[Path] = None
    if output_subdir is not None:
        output_dir = OUTPUT_BASE / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "VehicleScheduling.charge_type": ChargeType.DEPOT,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=0),
        "VehicleScheduling.maximum_schedule_duration": timedelta(hours=24),
        "DepotAssignment.depot_config": DEPOT_CONFIG,
        "Simulation.repetition_period": timedelta(weeks=1),
        "Simulation.smart_charging": SmartChargingStrategy.EVEN,
        "Simulation.ignore_unstable_simulation": True,
        "Simulation.calculate_timeseries": True,
        "DepotGenerator.charging_power_kw": depot_charging_power_kw,
    }

    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[common_db]).execute(context=context)

    run_steps(steps=[VehicleScheduling(), DepotAssignment(), HandleOnlyETrips()], context=context)

    run_steps(steps=[DepotGenerator(), Simulation()], context=context)

    if output_dir is not None:
        # Always generate the trip profile plot after simulation
        trip_profile_result = TripProfileAnalyzer().execute(context=context)
        if trip_profile_result is not None:
            fig = TripProfileAnalyzer.visualize(trip_profile_result)
            _save_fig(fig, output_dir, "trip_profile")

        sec_result = cast(
            pd.DataFrame, SWUSpecificEnergyConsumptionAnalyzer().execute(context=context)
        )
        if not sec_result.empty:
            fig = SWUSpecificEnergyConsumptionAnalyzer.visualize(sec_result)
            _save_fig(fig, output_dir, "specific_energy_consumption")

        if enable_plots:
            plots_dir = output_dir / "visualizations"
            plots_dir.mkdir(parents=True, exist_ok=True)
            generate_all_plots(
                context=context,
                output_dir=plots_dir,
                include_videos=False,
                pre_simulation_only=False,
            )

    assert context.current_db is not None
    return context.current_db


@flow(name="SWU Opportunity Variant", flow_run_name="SWU - opportunity")
def run_opportunity_variant(
    common_db: Path,
    enable_plots: bool,
    *,
    depot_charging_power_kw: float = DEFAULT_DEPOT_CHARGING_POWER_KW,
    terminus_charging_power_kw: float = TERMINUS_CHARGING_POWER_KW,
    cache_subdir: str = "opportunity",
    output_subdir: Optional[str] = "opportunity",
) -> Path:
    """Run the OPPORTUNITY charging variant. Returns the path to the final database.

    Pass ``output_subdir=None`` to suppress all per-run output writes.
    """
    work_dir = CACHE_BASE / cache_subdir
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir: Optional[Path] = None
    if output_subdir is not None:
        output_dir = OUTPUT_BASE / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=0),
        "VehicleScheduling.maximum_schedule_duration": timedelta(hours=24),
        "IntegratedScheduling.max_iterations": 3,
        "DepotAssignment.depot_config": DEPOT_CONFIG,
        # InsufficientChargingTimeAnalyzer must use the same charging power as
        # StationElectrification — the latter validates this and refuses to run otherwise.
        "InsufficientChargingTimeAnalyzer.charging_power_kw": terminus_charging_power_kw,
        "StationElectrification.charging_power_kw": terminus_charging_power_kw,
        "StationElectrification.max_stations_to_electrify": 9999,
        "Simulation.repetition_period": timedelta(weeks=1),
        "Simulation.smart_charging": SmartChargingStrategy.EVEN,
        "Simulation.ignore_unstable_simulation": True,
        "Simulation.calculate_timeseries": True,
        "DepotGenerator.charging_power_kw": depot_charging_power_kw,
    }

    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[common_db]).execute(context=context)

    # NOTE: IntegratedScheduling returns the database in the "just after vehicle scheduling"
    # state — it rolls back its nested DepotAssignment calls, so we must re-run DepotAssignment.
    run_steps(
        steps=[IntegratedScheduling(), DepotAssignment(), HandleOnlyETrips()], context=context
    )

    # Diagnostic: identify rotations that are infeasible even with all termini electrified.
    insufficient_result = InsufficientChargingTimeAnalyzer().execute(context=context)

    if insufficient_result is not None:
        critical_ids = insufficient_result["rotation_ids"]
        soc_data = insufficient_result.get("soc_data", {})

        # Battery capacity is set in the common phase; the variant doesn't know it. Look it up.
        with context.get_session() as session:
            from eflips.model import VehicleType

            vt = session.query(VehicleType).first()
            battery_kwh_used = (
                float(vt.battery_capacity) if vt is not None else BATTERY_CAPACITY_KWH
            )

        lines = []
        for rot_id in critical_ids:
            if rot_id in soc_data:
                soc_df, _event_spans, rot_start, rot_end = soc_data[rot_id]
                min_soc = float(soc_df["soc"].min())
                deficit_kwh = abs(min_soc) * battery_kwh_used
                lines.append(
                    f"  Rotation {rot_id}: min SoC={min_soc:.3f} "
                    f"(deficit ≈{deficit_kwh:.0f} kWh), "
                    f"window {rot_start} – {rot_end}"
                )
            else:
                lines.append(f"  Rotation {rot_id}: no SoC data available")

        raise RuntimeError(
            f"[{AGENCY_LABEL}] Opportunity charging is not feasible: "
            f"{len(critical_ids)} rotation(s) cannot accumulate enough charge even with all "
            f"termini electrified at {terminus_charging_power_kw:.0f} kW "
            f"and a {battery_kwh_used:.0f} kWh battery.\n"
            f"These routes are structurally too long or have too few break opportunities.\n"
            f"Infeasible rotations:\n"
            + "\n".join(lines)
            + "\nPossible remedies: increase battery_capacity, add a minimum_break_time, "
            "or accept DEPOT-only charging for this agency."
        )

    run_steps(
        steps=[StationElectrification(), DepotGenerator(), Simulation()],
        context=context,
    )

    if output_dir is not None:
        sec_result = cast(
            pd.DataFrame, SWUSpecificEnergyConsumptionAnalyzer().execute(context=context)
        )
        if not sec_result.empty:
            fig = SWUSpecificEnergyConsumptionAnalyzer.visualize(sec_result)
            _save_fig(fig, output_dir, "specific_energy_consumption")

        # Representative service-day SoC plot: vehicle with the most terminus charging on the
        # busiest day. We pick the day from the simulated events rather than hard-coding it,
        # since the GTFS ingest auto-selects a week from the feed validity period.
        with context.get_session() as session:
            day_start, day_end = _pick_representative_day(session)

        soc_ctx = PipelineContext(
            work_dir=context.work_dir,
            params={
                **context.params,
                "RepresentativeVehicleSocAnalyzer.day_start": day_start,
                "RepresentativeVehicleSocAnalyzer.day_end": day_end,
            },
            current_db=context.current_db,
        )
        soc_result = RepresentativeVehicleSocAnalyzer().execute(context=soc_ctx)
        if soc_result is not None:
            soc_df, event_spans, soc_day_start, soc_day_end = soc_result
            fig = RepresentativeVehicleSocAnalyzer.visualize(
                soc_df, event_spans, soc_day_start, soc_day_end
            )
            _save_fig(fig, output_dir, "representative_vehicle_soc")

        if enable_plots:
            plots_dir = output_dir / "visualizations"
            plots_dir.mkdir(parents=True, exist_ok=True)
            generate_all_plots(
                context=context,
                output_dir=plots_dir,
                include_videos=False,
                pre_simulation_only=False,
            )

    assert context.current_db is not None
    return context.current_db


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


@flow(name="SWU GTFS Flow")
def swu_flow(enable_plots: bool = False) -> None:
    """Run DEPOT and OPPORTUNITY charging variants for SWU Verkehr GmbH (Ulm)."""
    logger.info(f"Processing agency: {AGENCY_LABEL}")

    common_db = run_common_phase()
    depot_db = run_depot_variant(common_db=common_db, enable_plots=enable_plots)
    opportunity_db = run_opportunity_variant(common_db=common_db, enable_plots=enable_plots)

    # Cross-scenario comparison: fleet size and charging infrastructure. DEP is the baseline
    # so the TERM row reports vehicles saved by enabling terminus charging.
    comparison_dir = OUTPUT_BASE / "comparison"
    scenario_config = ScenarioDisplayConfig(
        order=["DEP", "TERM"],
        display_names={
            "DEP": "Depot Charging Only",
            "TERM": "Terminus Charging",
        },
        baseline="DEP",
    )
    comparison_analyzer = ScenarioComparisonAnalyzer()
    comparison_rows: List[pd.DataFrame] = []
    for scenario_name, db_path, scenario_work_dir in [
        ("DEP", depot_db, CACHE_BASE / "depot"),
        ("TERM", opportunity_db, CACHE_BASE / "opportunity"),
    ]:
        ctx = PipelineContext(
            work_dir=scenario_work_dir,
            params={"ScenarioComparisonAnalyzer.scenario_name": scenario_name},
            current_db=db_path,
        )
        comparison_rows.append(cast(pd.DataFrame, comparison_analyzer.execute(context=ctx)))

    comparison_table = merge_scenario_comparisons(comparison_rows, config=scenario_config)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_table.to_excel(comparison_dir / "scenario_comparison.xlsx", index=False)
    fig = ScenarioComparisonAnalyzer.visualize(comparison_table)
    _save_fig(fig, comparison_dir, "scenario_comparison")
    logger.info(f"Scenario comparison saved to {comparison_dir}")


# ---------------------------------------------------------------------------
# Trip Profile Analyzer
# ---------------------------------------------------------------------------


DEFAULT_TRIP_PROFILE_ROUTE_ID = 6


class TripProfileAnalyzer(Analyzer):
    """
    Plots altitude (with stop labels), mean speed, and energy consumption per segment
    for a single DRIVING event from the simulation database.
    """

    def __init__(self, code_version: str = "v1.0.5", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.event_id": (
                "ID of the DRIVING Event to plot. If set, this takes precedence over "
                "route_id and the event must exist."
            ),
            f"{cls.__name__}.route_id": (
                "Route ID to filter by when event_id is not set. The first matching "
                f"DRIVING Event is picked. Default: {DEFAULT_TRIP_PROFILE_ROUTE_ID} "
                "(manually chosen for the SWU feed)."
            ),
        }

    def analyze(self, session: Session, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Build a per-stop DataFrame for one DRIVING event.

        Returns a DataFrame with columns: distance_km, altitude_m, stop_name,
        arrival_time, speed_kmh, consumption_kwh_per_km.
        """
        event_id = params.get(f"{self.__class__.__name__}.event_id", None)
        route_id = params.get(f"{self.__class__.__name__}.route_id", DEFAULT_TRIP_PROFILE_ROUTE_ID)

        if event_id is not None:
            event = session.query(Event).filter(Event.id == event_id).one_or_none()
            if event is None:
                raise ValueError(
                    f"{self.__class__.__name__}: no Event found with id={event_id}. "
                    f"Set {self.__class__.__name__}.event_id to an existing DRIVING "
                    f"Event id, or leave it unset to fall back to route_id."
                )
        else:
            event = session.query(Event).join(Trip).filter(Trip.route_id == route_id).first()
            if event is None:
                raise ValueError(
                    f"{self.__class__.__name__}: no Event found for route_id={route_id}. "
                    f"Set {self.__class__.__name__}.route_id to a route present in the "
                    f"current database, or pin a specific event via "
                    f"{self.__class__.__name__}.event_id."
                )

        def _altitude(station: Any) -> float:
            return to_shape(station.geom).z if station.geom is not None else float("nan")

        battery_kwh: float = event.vehicle_type.battery_capacity
        event_start = event.time_start
        # timeseries["time"] is documented as ISO 8601 strings with TZ (eflips.model.Event);
        # cast narrows the column's Union type. fromisoformat returns a tz-aware datetime,
        # event_start is tz-aware, so the subtraction stays tz-consistent.
        ts_times_s = np.array(
            [
                (datetime.fromisoformat(t) - event_start).total_seconds()
                for t in cast(List[str], event.timeseries["time"])
            ]
        )
        ts_socs = np.array(event.timeseries["soc"], dtype=float)

        # Build ordered list of (elapsed_distance, station) from the route associations so
        # that stations appearing more than once on a circular route get their correct
        # per-occurrence distance rather than a single overwritten value.
        route_seq = [
            (a.elapsed_distance, a.station_id, a.station)
            for a in event.trip.route.assoc_route_stations
        ]

        dist_by_station: Dict[int, List[float]] = defaultdict(list)
        for elapsed, sid, sta in route_seq:
            dist_by_station[sid].append(elapsed)

        rows = []
        for st in event.trip.stop_times:
            distances = dist_by_station.get(st.station_id)
            if not distances:
                continue
            t_s = (st.arrival_time - event_start).total_seconds()
            # For stations that appear multiple times (e.g. circular terminus),
            # pick the distance occurrence whose position along the route best
            # matches the elapsed time fraction.
            trip_duration_s = (event.time_end - event_start).total_seconds()
            route_distance_m = event.trip.route.distance
            if len(distances) > 1 and trip_duration_s > 0:
                time_fraction = t_s / trip_duration_s
                expected_m = time_fraction * route_distance_m
                elapsed_dist = min(distances, key=lambda d: abs(d - expected_m))
            else:
                elapsed_dist = distances[0]

            rows.append(
                {
                    "distance_km": elapsed_dist / 1000.0,
                    "soc": float(np.interp(t_s, ts_times_s, ts_socs)),
                    "stop_name": st.station.name,
                    "altitude_m": _altitude(st.station),
                    "arrival_time": st.arrival_time,
                }
            )

        first_assoc = event.trip.route.assoc_route_stations[0]
        last_assoc = event.trip.route.assoc_route_stations[-1]
        rows.insert(
            0,
            {
                "distance_km": first_assoc.elapsed_distance / 1000.0,
                "soc": event.soc_start,
                "stop_name": first_assoc.station.name,
                "altitude_m": _altitude(first_assoc.station),
                "arrival_time": event.time_start,
            },
        )
        rows.append(
            {
                "distance_km": last_assoc.elapsed_distance / 1000.0,
                "soc": event.soc_end,
                "stop_name": last_assoc.station.name,
                "altitude_m": _altitude(last_assoc.station),
                "arrival_time": event.time_end,
            }
        )

        # Sort by arrival_time so circular-route duplicates are ordered correctly,
        # then drop stops outside the event window and any with non-advancing time.
        df = pd.DataFrame(rows).sort_values("arrival_time").reset_index(drop=True)
        df = df[
            (df["arrival_time"] >= event.time_start) & (df["arrival_time"] <= event.time_end)
        ].reset_index(drop=True)
        # Keep only rows where arrival_time strictly advances (removes same-second duplicates)
        arrival_diff = pd.to_timedelta(df["arrival_time"].diff())
        df = df[arrival_diff.dt.total_seconds().fillna(1) > 0].reset_index(drop=True)

        df["delta_soc"] = df["soc"].diff()
        df["delta_km"] = df["distance_km"].diff()
        df["delta_h"] = pd.to_timedelta(df["arrival_time"].diff()).dt.total_seconds() / 3600.0
        df["consumption_kwh_per_km"] = -battery_kwh * df["delta_soc"] / df["delta_km"]
        df["speed_kmh"] = df["delta_km"] / df["delta_h"]

        # Suppress segments with implausibly short dwell time (GTFS data artefacts where
        # consecutive stop_times are < 10 s apart produce physically impossible speeds).
        min_segment_h = 10.0 / 3600.0
        mask = df["delta_h"] < min_segment_h
        df.loc[mask, ["speed_kmh", "consumption_kwh_per_km"]] = float("nan")

        return df

    @staticmethod
    def visualize(df: pd.DataFrame) -> "Any":
        """
        Create a 3-panel figure: altitude with stop labels, mean speed, energy consumption.

        Returns:
            matplotlib Figure
        """
        configure_latex_plotting()

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH * 1.5),
            layout="constrained",
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]},
        )

        palette = sns.color_palette("Set2")
        x = df["distance_km"].values

        # --- Panel 0: Altitude with alternating stop-name labels ---
        axes[0].plot(
            x,
            df["altitude_m"].values,
            color=palette[1],
            linewidth=1.0,
            marker=".",
            markersize=3,
        )
        axes[0].set_ylabel("Altitude [m]")

        alt_vals = df["altitude_m"].dropna()
        if len(alt_vals) > 0:
            alt_min, alt_max = float(alt_vals.min()), float(alt_vals.max())
            alt_range = max(alt_max - alt_min, 10.0)
            # Reserve equal padding above and below for the alternating labels
            axes[0].set_ylim(alt_min - 0.35 * alt_range, alt_max + 0.35 * alt_range)

        for idx, (_, row) in enumerate(df.iterrows()):
            if np.isnan(row["altitude_m"]):
                continue
            label = row["stop_name"].replace(" ", "\n")
            above = idx < 9
            axes[0].annotate(
                label,
                xy=(row["distance_km"], row["altitude_m"]),
                textcoords="offset points",
                xytext=(2, 5) if above else (2, -5),
                fontsize=4.5,
                rotation=60 if above else -60,
                ha="left",
                va="bottom" if above else "top",
                multialignment="left",
            )

        # --- Panel 1: Mean speed per segment ---
        seg_x = df["distance_km"].iloc[1:].values
        seg_w = np.asarray(df["delta_km"].iloc[1:].values, dtype=float)
        axes[1].bar(
            seg_x,
            df["speed_kmh"].iloc[1:].values,
            width=seg_w * 0.85,
            align="center",
            color=palette[0],
            edgecolor="white",
            linewidth=0.4,
        )
        axes[1].set_ylabel("Speed\n[km/h]")

        # --- Panel 2: Energy consumption per segment ---
        axes[2].bar(
            seg_x,
            df["consumption_kwh_per_km"].iloc[1:].values,
            width=seg_w * 0.85,
            align="center",
            color=palette[2],
            edgecolor="white",
            linewidth=0.4,
        )
        axes[2].axhline(0, color="black", linewidth=0.5)
        axes[2].set_ylabel("Consumption\n[kWh/km]")
        axes[2].set_xlabel("Distance [km]")

        return fig


# ---------------------------------------------------------------------------
# Specific Energy Consumption Analyzer (matplotlib visualization)
# ---------------------------------------------------------------------------


class SWUSpecificEnergyConsumptionAnalyzer(SpecificEnergyConsumptionAnalyzer):
    """Specific energy consumption with a LaTeX-styled matplotlib histogram.

    Inherits the parent's per-trip kWh and distance computation; only the visualisation
    is replaced. The bars use the same Set2-palette / constrained-layout styling as
    :class:`TripProfileAnalyzer` so the two plots compose well in a publication.
    """

    @staticmethod
    def visualize(prepared_data: pd.DataFrame) -> Figure:
        """Histogram of specific energy consumption (kWh/km) per vehicle type."""
        configure_latex_plotting()

        df = prepared_data.copy()
        df = df[df["distance"] > 0]
        df["specific_energy_consumption"] = df["energy_consumption"] / df["distance"]
        df = df[np.isfinite(df["specific_energy_consumption"])]

        fig, ax = plt.subplots(
            figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH),
            layout="constrained",
        )

        if df.empty:
            ax.text(0.5, 0.5, "No driving events", ha="center", va="center")
            return fig

        palette = sns.color_palette("Set2")

        vehicle_types = sorted(df["vehicle_type_name"].unique())
        # Aim for ~10 trips per bin, clamped to a sensible range.
        n_bins = max(10, min(50, len(df) // 10))
        v_min = float(df["specific_energy_consumption"].min())
        v_max = float(df["specific_energy_consumption"].max())
        if v_min == v_max:
            v_min, v_max = v_min - 0.5, v_max + 0.5
        bins = np.linspace(v_min, v_max, n_bins + 1).tolist()

        single = len(vehicle_types) <= 1
        for idx, vt_name in enumerate(vehicle_types):
            data = df.loc[df["vehicle_type_name"] == vt_name, "specific_energy_consumption"]
            # Start at palette[2] (the same lavender as TripProfileAnalyzer's consumption bars)
            # so a single-VT histogram reads as the consumption-themed plot it is.
            ax.hist(
                data,
                bins=bins,
                color=palette[(2 + idx) % len(palette)],
                edgecolor="white",
                linewidth=0.4,
                alpha=1.0 if single else 0.7,
                label=vt_name,
            )

        ax.set_xlabel("Specific Energy Consumption [kWh/km]")
        ax.set_ylabel("Trip Count")
        if not single:
            ax.legend()

        return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWU (Ulm) GTFS Simulation Flow")
    parser.add_argument("--plots", action="store_true", help="Enable plot generation")
    args = parser.parse_args()

    swu_flow(enable_plots=args.plots)
