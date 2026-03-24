#!/usr/bin/env python3

"""
BVG Three-Scenario Flow

This flow implements the BVG (Berlin public transport) three-scenario analysis:
- OU (Originalumläufe): Original blocks with depot + terminus charging
- DEP (Depotlader): Depot-only charging with large batteries
- TERM (Fokus Endhaltestellen): Terminal focus with IntegratedScheduling and smaller batteries
- DIESEL: Diesel baseline for comparison

The scenarios run in parallel after a common pipeline
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, cast

import pandas as pd
from eflips.depot.api import SmartChargingStrategy  # type: ignore[import-untyped]
from eflips.model import ChargeType, Depot, Station, VehicleType
from matplotlib.figure import Figure
from prefect import flow
from sqlalchemy.orm import Session

from eflips.x.flows import run_steps
from eflips.x.framework import Modifier
from eflips.x.framework import PipelineContext, PipelineStep
from eflips.x.steps.analyzers import (
    GeographicTripPlotAnalyzer,
    PowerAndOccupancyAnalyzer,
    RevenueServiceTimelineAnalyzer,
    SchedulingEfficiencyAnalyzer,
    TCOAnalyzer,
    VehicleTypeDepotPlotAnalyzer,
    merge_tco_results,
)
from eflips.x.steps.analyzers.bvg_tools import (
    RepresentativeVehicleSocAnalyzer,
    ScenarioComparisonAnalyzer,
    merge_scenario_comparisons,
    visualize_depot_and_terminus_power,
    visualize_power_comparison,
    visualize_routes_by_depot_cartopy,
    visualize_tco_comparison,
)
from eflips.x.steps.generators import BVGXMLIngester, CopyCreator
from eflips.x.steps.modifiers.bvg_tools import (
    MergeStations,
    ReduceToNDaysNDepots,
    RemoveUnusedRotations,
    SetUpBvgVehicleTypes,
    depots_for_bvg,
)
from eflips.x.steps.modifiers.general_utilities import (
    AddTemperatures,
    CalculateConsumptionScaling,
    RemoveConsumptionLuts,
    RemoveUnusedData,
)
from eflips.x.steps.modifiers.scheduling import (
    DepotAssignment,
    StationElectrification,
    VehicleScheduling,
)
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation, SmartCharging

logger = logging.getLogger(__name__)

# ============================================================================
# Module-Level Configuration
# ============================================================================

# Module-level switch for testing
REDUCED_DATA = False  # Set to True for quick testing
LOG_LEVEL = "INFO"

# Derived configuration
if REDUCED_DATA:
    NUM_DAYS: int | None = 1
    NUM_DEPOTS: int | None = 2
    SIMULATION_DAYS = 1
else:
    NUM_DAYS = None  # All days
    NUM_DEPOTS = None  # All depots
    SIMULATION_DAYS = 7

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / "data" / "input" / "Berlin 2025-06"
if REDUCED_DATA:
    WORK_DIR_BASE = PROJECT_ROOT / "data" / "cache" / "bvgmini"
else:
    WORK_DIR_BASE = PROJECT_ROOT / "data" / "cache" / "bvg"


# ============================================================================
# TCO Configuration
# ============================================================================

# Per-vehicle-type TCO parameters for BVG fleet
BVG_TCO_VEHICLE_TYPES: Dict[str, Any] = {
    "EN": {"useful_life": 14, "procurement_cost": 580_000.0, "cost_escalation": 0.02},
    "GN": {"useful_life": 14, "procurement_cost": 780_000.0, "cost_escalation": 0.02},
    "DD": {"useful_life": 14, "procurement_cost": 780_000.0, "cost_escalation": 0.02},
}

# Per-vehicle-type battery TCO parameters (procurement_cost is EUR per kWh)
BVG_TCO_BATTERY_TYPES: Dict[str, Any] = {
    "EN": {
        "name": "Ebusco 3.0 12 large battery",
        "procurement_cost": 190,
        "useful_life": 7,
        "cost_escalation": -0.03,
    },
    "GN": {
        "name": "Solaris Urbino 18 large battery",
        "procurement_cost": 190,
        "useful_life": 7,
        "cost_escalation": -0.03,
    },
    "DD": {
        "name": "Alexander Dennis Enviro500EV large battery",
        "procurement_cost": 190,
        "useful_life": 7,
        "cost_escalation": -0.03,
    },
}

# Average-day energy consumption factors in kWh/km.
# These are LOWER than the simulated worst-case consumption, because the simulation
# plans for extreme conditions (e.g. -12C) while TCO should reflect average operations.
BVG_TCO_ENERGY_CONSUMPTION_FACTORS: Dict[str, float] = {
    "EN": 1.48,
    "GN": 2.16,
    "DD": 2.16,
}

BVG_TCO_CHARGING_POINT_TYPES: List[Dict[str, Any]] = [
    {
        "type": "depot",
        "name": "Depot Charging Point",
        "procurement_cost": 119_899.50,
        "useful_life": 20,
        "cost_escalation": 0.02,
    },
    {
        "type": "opportunity",
        "name": "Opportunity Charging Point",
        "procurement_cost": 299_748.74,
        "useful_life": 20,
        "cost_escalation": 0.02,
    },
]

BVG_TCO_CHARGING_INFRASTRUCTURE: List[Dict[str, Any]] = [
    {
        "type": "depot",
        "name": "Depot Charging Infrastructure",
        "procurement_cost": 2_397_989.95,
        "useful_life": 20,
        "cost_escalation": 0.02,
    },
    {
        "type": "station",
        "name": "Opportunity Charging Infrastructure",
        "procurement_cost": 269_773.87,
        "useful_life": 20,
        "cost_escalation": 0.02,
    },
]


def _bvg_tco_params() -> Dict[str, Any]:
    """Return BVG-specific TCO parameters for the TCOAnalyzer."""
    return {
        "TCOAnalyzer.vehicle_type_tco_params": BVG_TCO_VEHICLE_TYPES,
        "TCOAnalyzer.battery_type_tco_params": BVG_TCO_BATTERY_TYPES,
        "TCOAnalyzer.energy_consumption_factor": BVG_TCO_ENERGY_CONSUMPTION_FACTORS,
        "TCOAnalyzer.charging_point_type_params": BVG_TCO_CHARGING_POINT_TYPES,
        "TCOAnalyzer.charging_infrastructure_params": BVG_TCO_CHARGING_INFRASTRUCTURE,
        # Financial defaults from document_params() are used (matching BVG values)
    }


# ============================================================================
# Helper Functions
# ============================================================================


class UpdateBatteryCapacity(Modifier):
    """
    Lightweight modifier to update battery capacities without re-creating vehicle types.

    This modifier only updates the battery_capacity attribute of existing vehicle types,
    preserving their consumption LUTs that were created by CalculateConsumptionScaling.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, Any]:
        return {
            f"{cls.__name__}.battery_capacities": """
Dictionary mapping vehicle type short names to new battery capacities (in kWh).
Example: {"EN": 500.0, "GN": 640.0, "DD": 472.0}
            """.strip(),
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Update battery capacities for specified vehicle types.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session
        params : Dict[str, Any]
            Pipeline parameters containing battery_capacities
        """
        param_key = f"{self.__class__.__name__}.battery_capacities"
        battery_capacities: Dict[str, float] = params.get(param_key, {})

        if not battery_capacities:
            logger.warning(f"No battery capacities specified in {param_key}, skipping update")
            return

        for name_short, new_capacity in battery_capacities.items():
            vehicle_type = (
                session.query(VehicleType).filter(VehicleType.name_short == name_short).first()
            )
            if vehicle_type:
                old_capacity = vehicle_type.battery_capacity
                vehicle_type.battery_capacity = new_capacity
                logger.info(
                    f"Updated battery capacity for {name_short}: "
                    f"{old_capacity} kWh -> {new_capacity} kWh"
                )
            else:
                logger.warning(f"Vehicle type {name_short} not found in database")


class CleanSimulationResults(Modifier):
    """
    Lightweight modifier to clean up simulation results from a previous run.

    This modifier removes all vehicles and events for the scenario and
    disconnects vehicles from rotations. Useful when re-running simulations on
    an existing database (e.g., branching from a completed simulation).

    Assumes exactly one scenario exists in the database.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, Any]:
        return {}

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Clean up simulation results for the scenario in the database.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session
        params : Dict[str, Any]
            Pipeline parameters (unused)
        """
        from eflips.model import Scenario, Rotation, Event, Vehicle

        # Get all scenarios - should be exactly one
        scenarios = session.query(Scenario).all()
        if len(scenarios) == 0:
            logger.error("No scenario found in database, cannot clean up")
            raise ValueError("No scenario found in database")
        elif len(scenarios) > 1:
            logger.error(f"Found {len(scenarios)} scenarios in database, expected exactly one")
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        scenario = scenarios[0]
        logger.info(f"Cleaning up simulation results for scenario ID: {scenario.id}")

        # Delete all vehicles and events, also disconnect the vehicles from the rotations
        rotation_q = session.query(Rotation).filter(Rotation.scenario_id == scenario.id)
        updated_count = rotation_q.update({"vehicle_id": None})
        logger.info(f"Disconnected {updated_count} rotations from vehicles")

        events_count = session.query(Event).filter(Event.scenario_id == scenario.id).delete()
        logger.info(f"Deleted {events_count} events")

        vehicles_count = session.query(Vehicle).filter(Vehicle.scenario_id == scenario.id).delete()
        logger.info(f"Deleted {vehicles_count} vehicles")


def save_plot_to_files_in_output_dir(fig: Figure, basename: str) -> None:
    """
    Utility mehtod to save a figure to the output directory with given basename.

    :param fig: A Matplotlib Figure object
    :param basename: The base name for the output files (without extension)
    :return: None
    """
    data_dir = output_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = data_dir / f"{basename}.pdf"
    fig.savefig(output_pdf, bbox_inches="tight")
    logger.info(f"Saved plot to: {output_pdf}")
    output_png = data_dir / f"{basename}.png"
    fig.savefig(output_png, bbox_inches="tight", dpi=300)
    logger.info(f"Saved plot to: {output_png}")


def output_dir() -> Path:
    project_root = PipelineStep.find_project_root()
    data_dir = project_root / "data" / "output" / "bvg"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _save_timeseries_to_excel(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame with a 'time' column to Excel, formatting time as string."""
    copy = df.copy()
    copy["time"] = copy["time"].dt.strftime("%Y-%m-%d %H:%M")
    copy.to_excel(output_dir() / filename, index=False)


# ============================================================================
# Common Pipeline
# ============================================================================


@flow(name="Common Pipeline")
def run_common_pipeline() -> Path:
    """
    Run the common pipeline that all scenarios branch from.

    Returns:
    --------
    Path
        Path to the final database from the common pipeline
    """
    logger.info("Starting common pipeline...")

    # Set up working directory
    work_dir = WORK_DIR_BASE / "common"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Locate XML input files
    if not INPUT_DATA_DIR.exists():
        raise FileNotFoundError(f"Input data directory not found: {INPUT_DATA_DIR}")

    xml_files = list(INPUT_DATA_DIR.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {INPUT_DATA_DIR}")

    logger.info(f"Found {len(xml_files)} XML files")

    # Configure parameters
    params = {
        "log_level": LOG_LEVEL,
        "BVGXMLIngester.multithreading": True,
        "AddTemperatures.temperature_celsius": -12.0,
        "Settings.use_reduced_data": REDUCED_DATA,
    }

    # Add reduction parameters if REDUCED_DATA is True
    if REDUCED_DATA:
        params["ReduceToNDaysNDepots.num_days"] = NUM_DAYS
        params["ReduceToNDaysNDepots.num_depots"] = NUM_DEPOTS
        logger.info(f"REDUCED_DATA mode: {NUM_DAYS} day(s), {NUM_DEPOTS} depot(s)")

    # Build pipeline steps
    steps: List[PipelineStep] = [
        BVGXMLIngester(input_files=xml_files),
        SetUpBvgVehicleTypes(),  # Default large batteries
        RemoveUnusedRotations(),
        MergeStations(),
    ]

    # Add reduction step if REDUCED_DATA is True
    if REDUCED_DATA:
        steps.append(ReduceToNDaysNDepots())

    steps.extend(
        [
            RemoveUnusedData(),
            AddTemperatures(),
            CalculateConsumptionScaling(),  # NEW: Calculate and apply BVG empirical scaling
        ]
    )

    # Execute pipeline
    context = PipelineContext(work_dir=work_dir, params=params)
    run_steps(context=context, steps=steps)

    vehicle_type_depot_ploter = (
        VehicleTypeDepotPlotAnalyzer()
    )  # Generate vehicle type/depot distribution plot
    prepared_data = vehicle_type_depot_ploter.execute(context=context)
    prepared_data.to_excel(
        output_dir() / "vehicle_km_by_depot_and_vehicle_type.xlsx",
        index=False,
    )
    fig = vehicle_type_depot_ploter.visualize(prepared_data)
    save_plot_to_files_in_output_dir(fig, "vehicle_km_by_depot_and_vehicle_type")

    # Geographic route visualization by depot
    geographic_analyzer = GeographicTripPlotAnalyzer()
    route_data = geographic_analyzer.execute(context=context)
    with context.get_session() as session:
        route_fig = visualize_routes_by_depot_cartopy(route_data, session)
    save_plot_to_files_in_output_dir(route_fig, "routes_by_depot")

    # Revenue service timeline visualization
    revenue_analyzer = RevenueServiceTimelineAnalyzer()
    revenue_data = revenue_analyzer.execute(context=context)
    revenue_data_copy = revenue_data.copy()
    revenue_data_copy.index = [idx.strftime("%Y-%m-%d %H:%M") for idx in revenue_data_copy.index]

    revenue_data_copy.to_excel(output_dir() / "revenue_service_timeline.xlsx", index=True)
    fig = RevenueServiceTimelineAnalyzer.visualize(revenue_data)
    save_plot_to_files_in_output_dir(fig, "revenue_service_timeline")

    logger.info(f"Common pipeline complete. Database: {context.current_db}")
    assert context.current_db is not None
    return context.current_db


# ============================================================================
# Scenario Tasks
# ============================================================================


def reduce_depots_for_bvg() -> List[Dict[str, Any]]:
    """
    If we're running in REDUCED_DATA mode, we cannot assign to 9 depots. So we take the
    default depots_for_bvg and reduce them to only the last three.

    Then, we modify the last three to allow all vehicle types and have a cpacity of 400 each.

    :return: A reduced depot configuration list
    """
    ALL_VEHICLE_TYPES = ["EN", "GN"]  # No DD in reduced data

    depot_list = []
    depot_list.append(
        {
            "depot_station": (13.5053889, 52.4714167),
            "name": "Betriebshof Rummelsburger Landstraße",
            "capacity": 200,
            "vehicle_type": ALL_VEHICLE_TYPES,
        }
    )

    # "Betriebshof Säntisstraße" will have a capacity of 230
    depot_list.append(
        {
            "depot_station": (13.3844563, 52.416735),
            "name": "Betriebshof Säntisstraße",
            "capacity": 200,
            "vehicle_type": ALL_VEHICLE_TYPES,
        }
    )

    # "Betriebshof Alt Friedrichsfelde" will have a capacity of 0
    depot_list.append(
        {
            "depot_station": (13.5401389, 52.5123056),
            "name": "Betriebshof Alt Friedrichsfelde",
            "capacity": 0,
            "vehicle_type": ALL_VEHICLE_TYPES,
        }
    )
    return depot_list


def _run_power_analyzer(
    context: PipelineContext,
    area_ids: "List[int] | None" = None,
    station_ids: "List[int] | None" = None,
) -> pd.DataFrame:
    """Execute PowerAndOccupancyAnalyzer with given area/station IDs."""
    params = context.params.copy()
    if area_ids is not None:
        params["PowerAndOccupancyAnalyzer.area_id"] = area_ids
    if station_ids is not None:
        params["PowerAndOccupancyAnalyzer.station_id"] = station_ids
        if area_ids is None:
            params["PowerAndOccupancyAnalyzer.area_id"] = None
    analysis_context = PipelineContext(
        work_dir=context.work_dir, params=params, current_db=context.current_db
    )
    return cast(pd.DataFrame, PowerAndOccupancyAnalyzer().execute(context=analysis_context))


def extract_power_for_depot(context: PipelineContext, depot_short_name: str) -> pd.DataFrame:
    """Extract power timeseries for a single depot by its short name."""
    with context.get_session() as session:
        station = session.query(Station).filter(Station.name_short == depot_short_name).one()
        depot = session.query(Depot).filter(Depot.station_id == station.id).one()
        area_ids = [area.id for area in depot.areas]
    return _run_power_analyzer(context, area_ids=area_ids)


def extract_power_for_all_depots(context: PipelineContext) -> pd.DataFrame:
    """Extract summed power timeseries across all depots."""
    with context.get_session() as session:
        all_area_ids = [area.id for depot in session.query(Depot).all() for area in depot.areas]
    return _run_power_analyzer(context, area_ids=all_area_ids)


def extract_power_for_all_termini(context: PipelineContext) -> pd.DataFrame:
    """Extract summed power timeseries across all electrified terminus stations."""
    with context.get_session() as session:
        depot_station_ids = {d.station_id for d in session.query(Depot).all()}
        terminus_stations = (
            session.query(Station)
            .filter(
                Station.is_electrified == True,  # noqa: E712
                ~Station.id.in_(depot_station_ids),
            )
            .all()
        )
        station_ids = [s.id for s in terminus_stations]
    return _run_power_analyzer(context, station_ids=station_ids)


@flow(name="OU Scenario: Original Blocks")
def run_ou_scenario(common_db: Path) -> Tuple[Path, pd.DataFrame, pd.DataFrame]:
    """
    Run the OU (Originalumläufe - Original Blocks) scenario.

    Baseline scenario with depot + terminus charging using large batteries.

    Parameters:
    -----------
    common_db : Path
        Path to the common database to branch from
    """
    logger.info("Starting OU scenario...")

    # Set up working directory
    work_dir = WORK_DIR_BASE / "ou"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Configure parameters
    params: Dict[str, Any] = {
        "log_level": LOG_LEVEL,
        "StationElectrification.charging_power_kw": 450.0,
        "DepotGenerator.charging_power_kw": 90.0,
        "Simulation.repetition_period": timedelta(days=SIMULATION_DAYS),
        "Simulation.ignore_unstable_simulation": True,
        "SchedulingEfficiencyAnalyzer.scenario_name": "OU",
    }

    # Create context and copy common database as baseline
    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[common_db]).execute(context=context)

    # Get depot configuration (need a session)
    with context.get_session() as session:
        if REDUCED_DATA:
            depots = reduce_depots_for_bvg()
        else:
            depots = depots_for_bvg(session)
        params["DepotAssignment.depot_config"] = depots

    # Continue with non-scheduling steps (can run in parallel with other scenarios)
    steps = [
        DepotAssignment(),
        StationElectrification(),
        DepotGenerator(),
        Simulation(),
    ]
    run_steps(context=context, steps=steps)

    efficiency_data = cast(pd.DataFrame, SchedulingEfficiencyAnalyzer().execute(context=context))
    efficiency_data.to_excel(output_dir() / "ou_scheduling_efficiency.xlsx", index=False)

    # Extract BFI power for comparison with OU-EVEN
    bfi_power_none = extract_power_for_depot(context, "BFI")
    _save_timeseries_to_excel(bfi_power_none, "ou_bfi_power_none.xlsx")

    logger.info(f"OU scenario complete. Database: {context.current_db}")

    assert context.current_db is not None
    return context.current_db, efficiency_data, bfi_power_none


@flow(name="DEP Scenario: Depot Only")
def run_dep_scenario(common_db: Path) -> Tuple[Path, pd.DataFrame]:
    """
    Run the DEP (Depotlader - Depot Only) scenario.

    Depot-only charging with large batteries.

    Parameters:
    -----------
    common_db : Path
        Path to the common database to branch from
    """
    logger.info("Starting DEP scenario...")

    # Set up working directory
    work_dir = WORK_DIR_BASE / "dep"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Configure parameters
    params: Dict[str, Any] = {
        "log_level": LOG_LEVEL,
        "VehicleScheduling.charge_type": ChargeType.DEPOT,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=0),
        "VehicleScheduling.battery_margin": 0.2,  # 20% for delta-SoC safety
        "DepotGenerator.charging_power_kw": 90.0,
        "Simulation.repetition_period": timedelta(days=SIMULATION_DAYS),
        "Simulation.ignore_unstable_simulation": True,
        "SchedulingEfficiencyAnalyzer.scenario_name": "DEP",
    }

    # Create context and copy common database as baseline
    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[common_db]).execute(context=context)

    # Get depot configuration
    with context.get_session() as session:
        if REDUCED_DATA:
            depots = reduce_depots_for_bvg()
        else:
            depots = depots_for_bvg(session)
        params["DepotAssignment.depot_config"] = depots

    # DEP uses large batteries (same as common pipeline defaults), no need to update

    steps = [
        VehicleScheduling(),
        DepotAssignment(),
        DepotGenerator(),
        Simulation(),
    ]

    run_steps(context=context, steps=steps)

    efficiency_data = cast(pd.DataFrame, SchedulingEfficiencyAnalyzer().execute(context=context))
    efficiency_data.to_excel(output_dir() / "dep_scheduling_efficiency.xlsx", index=False)

    logger.info(f"DEP scenario complete. Database: {context.current_db}")
    assert context.current_db is not None
    return context.current_db, efficiency_data


@flow(name="TERM Scenario: Terminal Focus")
def run_term_scenario(common_db: Path) -> Tuple[Path, pd.DataFrame]:
    """
    Run the TERM (Fokus Endhaltestellen - Terminal Focus) scenario.

    Opportunity charging with smaller batteries and IntegratedScheduling.

    Parameters:
    -----------
    common_db : Path
        Path to the common database to branch from
    """
    logger.info("Starting TERM scenario...")

    # Set up working directory
    work_dir = WORK_DIR_BASE / "term"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Configure parameters
    params: Dict[str, Any] = {
        "log_level": LOG_LEVEL,
        "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=0),
        "VehicleScheduling.maximum_schedule_duration": (
            timedelta(hours=24) if not REDUCED_DATA else timedelta(hours=4)
        ),  # Shorter schedules for reduced data
        "VehicleScheduling.battery_margin": 0.1,
        "IntegratedScheduling.max_iterations": 2,
        "StationElectrification.charging_power_kw": 450.0,
        "DepotGenerator.charging_power_kw": 90.0,
        "Simulation.repetition_period": timedelta(days=SIMULATION_DAYS),
        "Simulation.ignore_unstable_simulation": (
            True if REDUCED_DATA else False
        ),  # Reduced data has unstable simulation, for some reason.
        "SchedulingEfficiencyAnalyzer.scenario_name": "TERM",
    }

    # Create context and copy common database as baseline
    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[common_db]).execute(context=context)

    # Get depot configuration
    with context.get_session() as session:
        if REDUCED_DATA:
            depots = reduce_depots_for_bvg()
        else:
            depots = depots_for_bvg(session)
        params["DepotAssignment.depot_config"] = depots

    # Set up battery capacities (small batteries for TERM)
    params["UpdateBatteryCapacity.battery_capacities"] = {
        "EN": 250.0,  # 50% of large battery (500.0)
        "GN": 320.0,  # 50% of large battery (640.0)
        "DD": 320.0,  # 68% of large battery (472.0)
    }

    # NOTE: IntegratedScheduling returns database in "just after vehicle scheduling" state
    # It rolls back its nested DepotAssignment calls, so we must run DepotAssignment again
    steps = [
        UpdateBatteryCapacity(),
        VehicleScheduling(),  # TODO: Why does IntegratedScheduling not work?
        DepotAssignment(),  # Re-run since IntegratedScheduling rolls back
        StationElectrification(),
        DepotGenerator(),
        Simulation(),
    ]
    run_steps(context=context, steps=steps)

    efficiency_data = cast(pd.DataFrame, SchedulingEfficiencyAnalyzer().execute(context=context))
    efficiency_data.to_excel(output_dir() / "term_scheduling_efficiency.xlsx", index=False)

    logger.info(f"TERM scenario complete. Database: {context.current_db}")
    assert context.current_db is not None
    return context.current_db, efficiency_data


@flow(name="OU-EVEN Scenario: Original Blocks with Even Charging")
def run_ou_even_scenario(
    finished_ou_db: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the OU-EVEN scenario: re-simulate OU with SmartChargingStrategy.EVEN.

    Applies even smart charging to the finished OU scenario and re-runs
    consumption simulation.

    Parameters:
    -----------
    finished_ou_db : Path
        Path to the finished OU scenario database

    Returns:
    --------
    Tuple of (bfi_power, all_depots_power, all_termini_power) DataFrames
    """
    logger.info("Starting OU-EVEN scenario...")

    work_dir = WORK_DIR_BASE / "ou_even"
    work_dir.mkdir(parents=True, exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": LOG_LEVEL,
        "SmartCharging.smart_charging_strategy": SmartChargingStrategy.EVEN,
    }

    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[finished_ou_db]).execute(context=context)

    steps = [
        SmartCharging(),
    ]
    run_steps(context=context, steps=steps)

    # Extract power data
    bfi_power = extract_power_for_depot(context, "BFI")
    _save_timeseries_to_excel(bfi_power, "ou_bfi_power_even.xlsx")

    all_depots_power = extract_power_for_all_depots(context)
    _save_timeseries_to_excel(all_depots_power, "ou_even_all_depots_power.xlsx")

    all_termini_power = extract_power_for_all_termini(context)
    _save_timeseries_to_excel(all_termini_power, "ou_even_all_termini_power.xlsx")

    # Representative vehicle SoC day plots (terminus + depot modes)
    soc_analyzer = RepresentativeVehicleSocAnalyzer()
    for mode, filename in [
        ("terminus", "representative_vehicle_soc_day"),
        ("depot", "representative_depot_vehicle_soc_day"),
    ]:
        soc_ctx = PipelineContext(
            work_dir=context.work_dir,
            params={**context.params, "RepresentativeVehicleSocAnalyzer.mode": mode},
            current_db=context.current_db,
        )
        soc_data, event_spans, day_start, day_end = soc_analyzer.execute(context=soc_ctx)
        fig = RepresentativeVehicleSocAnalyzer.visualize(soc_data, event_spans, day_start, day_end)
        save_plot_to_files_in_output_dir(fig, filename)

    logger.info(f"OU-EVEN scenario complete. Database: {context.current_db}")
    return bfi_power, all_depots_power, all_termini_power


@flow(name="DIESEL Scenario: Diesel Reference")
def run_diesel_scenario(finished_ou_db: Path) -> Path:
    """
    Run the DIESEL scenario (diesel baseline for comparison).

    This scenario branches from OU's final state (after simulation).

    Parameters:
    -----------
    finished_ou_db : Path
        Path to the finished OU scenario database to branch from
    """
    logger.info("Starting DIESEL scenario...")

    # Set up working directory
    work_dir = WORK_DIR_BASE / "diesel"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Configure diesel-specific parameters
    params = {
        "log_level": LOG_LEVEL,
        "RemoveConsumptionLuts.minimal_consumption": 0.001,
        "DepotGenerator.charging_power_kw": 90.0,
        "Simulation.repetition_period": timedelta(days=SIMULATION_DAYS),
        "Simulation.ignore_unstable_simulation": False,
    }

    # Create context and copy OU final database as baseline
    context = PipelineContext(work_dir=work_dir, params=params)
    CopyCreator(input_files=[finished_ou_db]).execute(context=context)

    # Run diesel-specific steps
    steps = [
        CleanSimulationResults(),  # Clean up previous simulation results
        RemoveConsumptionLuts(),  # Remove LUTs and set minimal consumption
        DepotGenerator(),
        Simulation(),
    ]
    run_steps(context=context, steps=steps)

    logger.info(f"DIESEL scenario complete. Database: {context.current_db}")
    assert context.current_db is not None
    return context.current_db


# ============================================================================
# Main Flow
# ============================================================================


@flow(name="BVG Three-Scenario Flow")
def bvg_three_scenario_flow() -> None:
    """
    Main flow orchestrating the BVG three-scenario analysis.

    Runs a common pipeline followed by four scenarios in parallel:
    - OU, DEP, TERM start together after common pipeline
    - DIESEL starts after OU completes (branching dependency)

    """
    logger.info("Starting BVG three-scenario flow...")

    # Phase 1: Common pipeline (sequential)
    common_db = run_common_pipeline()

    # Submit OU, DEP, TERM and collect efficiency data
    ou_db, ou_eff, bfi_power_none = run_ou_scenario(common_db)
    dep_db, dep_eff = run_dep_scenario(common_db)
    term_db, term_eff = run_term_scenario(common_db)

    # DIESEL: diesel baseline branching from finished OU
    diesel_db = run_diesel_scenario(ou_db)

    # OU-EVEN: re-simulate OU with SmartChargingStrategy.EVEN
    bfi_power_even, all_depots_power, all_termini_power = run_ou_even_scenario(ou_db)

    # BFI power comparison plot (NONE vs EVEN)
    fig = visualize_power_comparison(bfi_power_none, bfi_power_even, depot_name="BFI")
    save_plot_to_files_in_output_dir(fig, "bfi_power_none_vs_even")

    # Depot vs terminus total power plot (EVEN only)
    fig = visualize_depot_and_terminus_power(all_depots_power, all_termini_power)
    save_plot_to_files_in_output_dir(fig, "ou_even_depot_vs_terminus_power")

    # Aggregate and save combined scheduling efficiency report
    all_eff = pd.concat([ou_eff, dep_eff, term_eff], ignore_index=True)
    all_eff.to_excel(output_dir() / "scheduling_efficiency.xlsx", index=False)
    fig = SchedulingEfficiencyAnalyzer.visualize(all_eff)
    save_plot_to_files_in_output_dir(fig, "scheduling_efficiency")
    fig_hist = SchedulingEfficiencyAnalyzer.visualize_histogram(all_eff)
    save_plot_to_files_in_output_dir(fig_hist, "scheduling_efficiency_histogram")

    # Scenario comparison table (fleet size, chargers, additional vehicles vs DIESEL)
    comparison_analyzer = ScenarioComparisonAnalyzer()
    comparison_rows: List[pd.DataFrame] = []
    for scenario_name, db_path, work_dir in [
        ("OU", ou_db, WORK_DIR_BASE / "ou"),
        ("DEP", dep_db, WORK_DIR_BASE / "dep"),
        ("TERM", term_db, WORK_DIR_BASE / "term"),
        ("DIESEL", diesel_db, WORK_DIR_BASE / "diesel"),
    ]:
        ctx = PipelineContext(
            work_dir=work_dir,
            params={"ScenarioComparisonAnalyzer.scenario_name": scenario_name},
            current_db=db_path,
        )
        row = cast(pd.DataFrame, comparison_analyzer.execute(context=ctx))
        comparison_rows.append(row)

    comparison_table = merge_scenario_comparisons(comparison_rows)
    comparison_table.to_excel(output_dir() / "scenario_comparison.xlsx", index=False)
    logger.info("Scenario comparison table saved to scenario_comparison.xlsx")
    fig = ScenarioComparisonAnalyzer.visualize(comparison_table)
    save_plot_to_files_in_output_dir(fig, "scenario_comparison")

    # TCO analysis (OU, DEP, TERM — DIESEL excluded as non-electric baseline)
    tco_analyzer = TCOAnalyzer()
    tco_params = _bvg_tco_params()
    tco_rows: List[pd.DataFrame] = []
    for scenario_name, db_path, work_dir in [
        ("OU", ou_db, WORK_DIR_BASE / "ou"),
        ("DEP", dep_db, WORK_DIR_BASE / "dep"),
        ("TERM", term_db, WORK_DIR_BASE / "term"),
    ]:
        ctx = PipelineContext(
            work_dir=work_dir,
            params={**tco_params, "TCOAnalyzer.scenario_name": scenario_name},
            current_db=db_path,
        )
        row = cast(pd.DataFrame, tco_analyzer.execute(context=ctx))
        tco_rows.append(row)

    tco_table = merge_tco_results(tco_rows)
    tco_table.to_excel(output_dir() / "tco_results.xlsx", index=False)
    logger.info("TCO comparison table saved to tco_results.xlsx")
    fig = visualize_tco_comparison(tco_table)
    save_plot_to_files_in_output_dir(fig, "tco_comparison")

    logger.info("BVG three-scenario flow complete!")
    logger.info(f"Results available in: {WORK_DIR_BASE}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    bvg_three_scenario_flow()
