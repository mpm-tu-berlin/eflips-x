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
from typing import List, Dict, Union, Tuple, Any

from eflips.model import ChargeType, VehicleType
from prefect import flow
from sqlalchemy.orm import Session

from eflips.x.flows import run_steps
from eflips.x.framework import Modifier
from eflips.x.framework import PipelineContext, PipelineStep
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
    IntegratedScheduling,
    StationElectrification,
    VehicleScheduling,
)
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation

logger = logging.getLogger(__name__)

# ============================================================================
# Module-Level Configuration
# ============================================================================

# Module-level switch for testing
REDUCED_DATA = False  # Set to True for quick testing
LOG_LEVEL = "INFO"

# Derived configuration
if REDUCED_DATA:
    NUM_DAYS = 1
    NUM_DEPOTS = 2
    SIMULATION_DAYS = 1
else:
    NUM_DAYS = None  # All days
    NUM_DEPOTS = None  # All depots
    SIMULATION_DAYS = 7

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INPUT_DATA_DIR = PROJECT_ROOT / "data" / "input" / "Berlin 2025-06"
WORK_DIR_BASE = PROJECT_ROOT / "data" / "cache" / "bvg"


# ============================================================================
# Helper Functions
# ============================================================================


class UpdateBatteryCapacity(Modifier):
    """
    Lightweight modifier to update battery capacities without re-creating vehicle types.

    This modifier only updates the battery_capacity attribute of existing vehicle types,
    preserving their consumption LUTs that were created by CalculateConsumptionScaling.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
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

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
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

    logger.info(f"Common pipeline complete. Database: {context.current_db}")
    return context.current_db


# ============================================================================
# Scenario Tasks
# ============================================================================


def reduce_depots_for_bvg() -> List[Dict[str, Union[int, Tuple[float, float], List[int], str]]]:
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
            "capacity": 200,
            "vehicle_type": ALL_VEHICLE_TYPES,
        }
    )
    return depot_list


@flow(name="OU Scenario: Original Blocks")
def run_ou_scenario(common_db: Path) -> Path:
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
    params = {
        "log_level": LOG_LEVEL,
        "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=10),
        "VehicleScheduling.battery_margin": 0.1,
        "StationElectrification.charging_power_kw": 450.0,
        "DepotGenerator.charging_power_kw": 90.0,
        "Simulation.repetition_period": timedelta(days=SIMULATION_DAYS),
        "Simulation.ignore_unstable_simulation": False,
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
        VehicleScheduling(),
        DepotAssignment(),
        StationElectrification(),
        DepotGenerator(),
        Simulation(),
    ]
    run_steps(context=context, steps=steps)

    logger.info(f"OU scenario complete. Database: {context.current_db}")

    return context.current_db  # Needed for DIESEL branching


@flow(name="DEP Scenario: Depot Only")
def run_dep_scenario(common_db: Path) -> None:
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
    params = {
        "log_level": LOG_LEVEL,
        "VehicleScheduling.charge_type": ChargeType.DEPOT,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=0),
        "VehicleScheduling.battery_margin": 0.2,  # 20% for delta-SoC safety
        "DepotGenerator.charging_power_kw": 90.0,
        "Simulation.repetition_period": timedelta(days=SIMULATION_DAYS),
        "Simulation.ignore_unstable_simulation": True,
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

    logger.info(f"DEP scenario complete. Database: {context.current_db}")


@flow(name="TERM Scenario: Terminal Focus")
def run_term_scenario(common_db: Path) -> None:
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
    params = {
        "log_level": LOG_LEVEL,
        "VehicleScheduling.charge_type": ChargeType.OPPORTUNITY,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=10),
        "VehicleScheduling.battery_margin": 0.1,
        "IntegratedScheduling.max_iterations": 2,
        "StationElectrification.charging_power_kw": 450.0,
        "DepotGenerator.charging_power_kw": 90.0,
        "Simulation.repetition_period": timedelta(days=SIMULATION_DAYS),
        "Simulation.ignore_unstable_simulation": False,
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
        IntegratedScheduling(),
        DepotAssignment(),  # Re-run since IntegratedScheduling rolls back
        StationElectrification(),
        DepotGenerator(),
        Simulation(),
    ]
    run_steps(context=context, steps=steps)

    logger.info(f"TERM scenario complete. Database: {context.current_db}")


@flow(name="DIESEL Scenario: Diesel Reference")
def run_diesel_scenario(finished_ou_db: Path) -> None:
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

    # Submit OU, DEP, TERMl
    run_ou_scenario(common_db)
    run_dep_scenario(common_db)
    run_term_scenario(common_db)

    logger.info("BVG three-scenario flow complete!")
    logger.info(f"Results available in: {WORK_DIR_BASE}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    bvg_three_scenario_flow()
