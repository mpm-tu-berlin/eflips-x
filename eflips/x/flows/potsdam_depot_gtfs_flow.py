#!/usr/bin/env python3

"""
ViP (Potsdam) GTFS flow demonstrating GTFS ingestion and scheduling.

This flow shows how to:
1. Ingest data from GTFS files
2. Configure vehicle types with battery capacity and consumption
3. Run scheduling and simulation
4. Analyze and visualize results

The pipeline follows a simple pattern:
- Ingest GTFS data
- Configure vehicle parameters
- Run scheduling and simulation
- Save visualization outputs
"""
import logging
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from tempfile import gettempdir
from typing import List, Any, Dict

import dash_cytoscape as cyto  # type: ignore[import-untyped]
import folium  # type: ignore[import-untyped]
import plotly  # type: ignore[import-untyped]
import sqlalchemy.orm
from eflips.depot.api import SmartChargingStrategy  # type: ignore[import-untyped]
from eflips.model import ChargeType, VehicleType, Rotation, Station
from prefect import flow

from eflips.x.flows import generate_all_plots
from eflips.x.flows import run_steps
from eflips.x.framework import PipelineStep, PipelineContext, Modifier
from eflips.x.steps.generators import GTFSIngester
from eflips.x.steps.modifiers.bvg_tools import MergeStations
from eflips.x.steps.modifiers.general_utilities import RemoveUnusedData
from eflips.x.steps.modifiers.scheduling import (
    VehicleScheduling,
    DepotAssignment,
    StationElectrification,
)
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation

# Constants for vehicle configuration
BATTERY_CAPACITY = 360.0  # kWh - large for electric buses
ENERGY_CONSUMPTION = 1.5  # kWh/km - typical consumption per km
CONSTANT_CHARGING_CURVE = [[0.0, 450.0], [1.0, 450.0]]  # kW: constant 450kW from 0-100% SoC

logger = logging.getLogger(__name__)


class ConfigureVehicleTypes(Modifier):
    """Lightweight modifier to configure vehicle types with hardcoded values.

    This modifier sets battery capacity, energy consumption, opportunity charging
    capability, and charging curves for all vehicle types in the database. It also
    enables opportunity charging for all rotations.

    The values are hardcoded constants defined at the module level for the VBB use case.
    """

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        raise NotImplementedError("This modifier does not take any parameters.")

    def modify(self, session: sqlalchemy.orm.Session, params: Dict[str, Any]) -> None:
        """Apply vehicle type configuration to the database.

        Args:
            session: SQLAlchemy session for database access
            params: Pipeline parameters (unused by this modifier)
        """
        # Configure all vehicle types
        vehicle_types = session.query(VehicleType).all()
        if not vehicle_types:
            logger.warning("No vehicle types found in database!")
        for vt in vehicle_types:
            vt.battery_capacity = BATTERY_CAPACITY
            vt.consumption = ENERGY_CONSUMPTION
            vt.opportunity_charging_capable = True
            vt.charging_curve = CONSTANT_CHARGING_CURVE
            logger.info(
                f"Set vehicle type '{vt.name}': "
                f"battery_capacity={BATTERY_CAPACITY} kWh, "
                f"consumption={ENERGY_CONSUMPTION} kWh/km"
            )

        # Enable opportunity charging for rotations
        session.query(Rotation).update({"allow_opportunity_charging": True})
        session.flush()


@flow
def vbb_gtfs_flow() -> None:
    """EFLiPS flow for GTFS ingestion, vehicle scheduling, and simulation for VBB.

    This flow demonstrates a complete electric bus fleet analysis workflow:

    Phase 1 - Data Ingestion and Setup:
        - Ingest GTFS data for VBB bus network
        - Clean up unused data
        - Configure vehicle types with battery capacity and charging parameters

    Phase 2 - Scheduling and Simulation:
        - Generate vehicle schedules with opportunity charging
        - Assign vehicles to depot
        - Determine station electrification needs
        - Run simulation to analyze charging and energy consumption

    Phase 3 - Visualization:
        - Generate comprehensive analysis plots and reports

    The flow uses a two-phase execution pattern to ensure data is properly
    loaded before scheduling operations begin.
    """
    ### Step 1: Initialize Pipeline ###
    # Create a unique working directory for this pipeline run

    work_dir = Path(gettempdir()) / ("eflips_vbb_gtfs_flow")
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {work_dir}")

    # Pipeline parameters
    params: Dict[str, Any] = {
        "log_level": "DEBUG",
    }

    # Phase 1: Data ingestion and initial setup
    initial_steps: List[PipelineStep] = []

    ### Step 2: Configure GTFS Data Ingestion ###
    # Locate and load GTFS input file
    path_to_this_file = Path(__file__).resolve()
    path_to_gtfs_file = (
        path_to_this_file.parent.parent.parent.parent / "data" / "input" / "GTFS" / "VBB.zip"
    )

    if not path_to_gtfs_file.exists():
        raise FileNotFoundError(f"GTFS file not found: {path_to_gtfs_file}")

    # Configure GTFS ingestion parameters
    params["GTFSIngester.bus_only"] = True
    params["GTFSIngester.duration"] = "WEEK"  # Use one week of data
    params["GTFSIngester.agency_name"] = "Verkehrsbetrieb Potsdam GmbH"
    # start_date will be auto-selected

    initial_steps.append(GTFSIngester(input_files=[path_to_gtfs_file]))

    ### Step 2: Merge Duplicate Stations ###
    station_merger = MergeStations()
    initial_steps.append(station_merger)

    ### Step 3: General Data Cleanup ###
    # Remove unnecessary data to improve processing speed
    initial_steps.append(RemoveUnusedData())

    ### Step 4: Configure Vehicle Battery Capacity and Energy Consumption ###
    # Set battery capacity, energy consumption, and opportunity charging via Modifier
    initial_steps.append(ConfigureVehicleTypes())

    ### Step 5: Execute initial steps ###
    # We need to execute the ingestion, cleanup, and vehicle configuration before scheduling
    pipeline = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps=initial_steps, context=pipeline)

    # Phase 2: Scheduling and simulation
    scheduling_steps: List[PipelineStep] = []

    ### Step 6: Vehicle Scheduling ###
    # Configure scheduling parameters
    params["VehicleScheduling.charge_type"] = ChargeType.DEPOT
    params["VehicleScheduling.minimum_break_time"] = timedelta(minutes=0)
    params["VehicleScheduling.maximum_schedule_duration"] = timedelta(hours=24)
    scheduling_steps.append(VehicleScheduling())

    ### Step 7: Depot Assignment ###
    # Configure a depot with coordinates and capacity
    # ViP operates in Potsdam, Germany - using approximate coordinates
    depot_config_1 = {
        "depot_station": (13.1122607, 52.3759956),  # (lon, lat) - ViP Betriebshof
        "name": "ViP Betriebshof",
        "vehicle_type": ["default_bus"],  # Will be created by GTFS ingester
        "capacity": 9999,
    }
    depot_config_2 = {
        "depot_station": (13.0393382, 52.4036622),  # (lon, lat) - ViP Betriebshof
        "name": "Fiktiver Betriebshof Sanssouci",
        "vehicle_type": ["default_bus"],  # Will be created by GTFS ingester
        "capacity": 9999,
    }
    params["DepotAssignment.depot_config"] = [depot_config_1, depot_config_2]
    scheduling_steps.append(DepotAssignment())

    ### Step 7.5: Station Electrification ###
    # Skipped here, since we're doing depot charging only
    if params["VehicleScheduling.charge_type"] == ChargeType.OPPORTUNITY:
        params["InsufficientChargingTimeAnalyzer.charging_power_kw"] = 450  # kW
        # TODO: Enable InsufficientChargingTimeAnalyzer when ready
        # is_possible_analyzer = InsufficientChargingTimeAnalyzer()
        # scheduling_steps.append(is_possible_analyzer)
        params["StationElectrification.max_stations_to_electrify"] = 236
        station_electrification = StationElectrification()
        scheduling_steps.append(station_electrification)

    ### Step 8: Run Simulation ###
    # Generate depot infrastructure objects
    scheduling_steps.append(DepotGenerator())

    # Run the actual vehicle and charging simulation
    # Custom setting the repetion period to 1 week because auto-detection may fail
    params["Simulation.repetition_period"] = timedelta(weeks=1)
    # Custom setting to enable smart charging
    params["Simulation.smart_charging"] = (
        SmartChargingStrategy.NONE
    )  # TODO: Change to EVEN when available
    scheduling_steps.append(Simulation())

    ### Step 9: Execute scheduling and simulation steps ###
    run_steps(steps=scheduling_steps, context=pipeline)

    ### Step 10: Generate Visualizations ###
    output_dir = work_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_all_plots(
        context=pipeline, output_dir=output_dir, include_videos=False, pre_simulation_only=False
    )


if __name__ == "__main__":
    vbb_gtfs_flow()
