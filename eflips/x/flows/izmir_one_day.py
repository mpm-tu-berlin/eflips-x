#!/usr/bin/env python3

"""
Izmir (IZBB) one-day flow.

Runs the full Eshot network but only for a single day. Faster to iterate on
than the week-long ``izmir`` (full) flow while still exercising every route.
For a week-long subset see ``izmir_reduced_lines``.
"""
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import gettempdir
from typing import List, Any, Dict, Type, Optional
from eflips.x.steps.modifiers.gtfs_utilities import ConfigureVehicleTypes

# Add the project root to sys.path to allow importing from eflips.x
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import dash_cytoscape as cyto  # type: ignore[import-untyped]
import eflips.model
import folium  # type: ignore[import-untyped]
import matplotlib
import plotly  # type: ignore[import-untyped]
from eflips.model import ChargeType, Vehicle, Depot, Area, Rotation, Event
from prefect import flow

from eflips.x.flows import run_steps
from eflips.x.framework import PipelineStep, PipelineContext, Analyzer
from eflips.x.steps.analyzers import (
    RotationInfoAnalyzer,
    DepartureArrivalSocAnalyzer,
    SpecificEnergyConsumptionAnalyzer,
    VehicleSocAnalyzer,
    DepotActivityAnalyzer,
    GeographicTripPlotAnalyzer,
    SingleRotationInfoAnalyzer,
    DepotEventAnalyzer,
    PowerAndOccupancyAnalyzer,
    InteractiveMapAnalyzer,
)
from eflips.x.steps.generators import GTFSIngester
from eflips.x.steps.modifiers.general_utilities import RemoveUnusedData, AddTemperatures
from eflips.x.steps.modifiers.scheduling import VehicleScheduling, DepotAssignment
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation


def save_visualization(vis: Any, output_file: Path, analyzer: Optional[Analyzer] = None) -> None:
    """Save a visualization to a file based on its type."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(vis, plotly.graph_objs._figure.Figure):
        vis.write_html(output_file)
    elif isinstance(vis, folium.Map):
        vis.save(str(output_file))
    elif isinstance(vis, cyto.Cytoscape):
        if analyzer is None:
            raise ValueError("Analyzer instance required for Cytoscape export")
        assert hasattr(
            analyzer, "export_cytoscape_html"
        ), "Analyzer must have export_cytoscape_html method for Cytoscape export"
        analyzer.export_cytoscape_html(cytoscape=vis, filename=str(output_file))
    elif isinstance(vis, matplotlib.animation.FuncAnimation):
        writer = matplotlib.animation.FFMpegWriter(
            fps=30,
            codec="libx264",
            extra_args=["-preset", "fast", "-tune", "animation", "-crf", "18"],
        )
        vis.save(output_file, writer=writer)
    else:
        raise TypeError(f"Unknown visualization type: {type(vis)}")

    print(f"Wrote analysis output to: {output_file}")


def execute_and_save_simple_analyzer(
    analyzer_class: Type[Analyzer],
    context: PipelineContext,
    output_dir: Path,
) -> None:
    """Execute a simple analyzer and save its visualization."""
    analyzer = analyzer_class()
    result = analyzer.execute(context=context)

    if not hasattr(analyzer, "visualize"):
        raise AttributeError(f"{analyzer_class.__name__} has no visualize method")

    vis = analyzer.visualize(result)
    output_file = output_dir / f"{analyzer_class.__name__}.html"
    save_visualization(vis, output_file, analyzer)


def query_all_ids(
    context: PipelineContext, model_class: Type[eflips.model.Base], id_attr: str = "id"
) -> List[int]:
    """Query all IDs for a given model class."""
    with context.get_session() as session:
        return [getattr(obj, id_attr) for obj in session.query(model_class).all()]


@flow(name="Izmir One Day (IZBB)")
def main() -> None:
    ### Step 1: Initialize Pipeline ###
    work_dir = Path("data/cache/eflips_izmir_one_day")
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {work_dir}")

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "GTFSIngester.duration": 'DAY',
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
    }

    steps: List[PipelineStep] = []

    ### Step 2: Configure Data Ingestion ###
    path_to_this_file = Path(__file__).resolve()
    gtfs_file = (
        path_to_this_file.parent.parent.parent.parent / "data" / "input" / "GTFS" / "izmir.zip"
    )
    steps.append(GTFSIngester(input_files=[gtfs_file]))

    ### Step 3: General Data Cleanup ###
    steps.append(RemoveUnusedData())
    steps.append(ConfigureVehicleTypes())
    steps.append(AddTemperatures())

    ### Step 4: Vehicle Scheduling ###
    params["VehicleScheduling.charge_type"] = ChargeType.DEPOT
    steps.append(VehicleScheduling())

    ### Step 5: Depot Assignment ###
    # Izmir Depot Locations
    # Note: coordinates are (longitude, latitude)
    depot_configs = [
        {
            "name": "Adatepe",
            "depot_station": (27.1860, 38.3850),
            "vehicle_type": ["TEMSA_AE",],
            "capacity": 200,
        },
        {
            "name": "Mersinli",
            "depot_station": (27.1712, 38.4335),
            "vehicle_type": ["TEMSA_AE",],
            "capacity": 200,
        },
        {
            "name": "Çiğli",
            "depot_station": (27.0704, 38.4846),
            "vehicle_type": ["TEMSA_AE",],
            "capacity": 200,
        },
        {
            "name": "Urla",
            "depot_station": (26.7548, 38.3260),
            "vehicle_type": ["TEMSA_AE",],
            "capacity": 100,
        },
        {
            "name": "Çakalburnu",
            "depot_station": (27.0636, 38.4049),
            "vehicle_type": ["TEMSA_AE",],
            "capacity": 200,
        },
        {
            "name": "Bergama",
            "depot_station": (27.1180, 39.0839),
            "vehicle_type": ["TEMSA_AE",],
            "capacity": 100,
        },
        {
            "name": "Gaziemir",
            "depot_station": (27.1199, 38.3481),
            "vehicle_type": ["TEMSA_AE",],
            "capacity": 200,
        },
    ]
    params["DepotAssignment.depot_config"] = depot_configs
    steps.append(DepotAssignment())

    ### Step 6: Run Simulation ###
    steps.append(DepotGenerator())
    steps.append(Simulation())

    ### Step 7: Execute Pipeline ###
    pipeline = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps=steps, context=pipeline)

    ### Step 8: Analyze and Visualize Results ###
    output_directory = work_dir / "analysis"

    simple_analyzers = [
        RotationInfoAnalyzer,
        GeographicTripPlotAnalyzer,
        DepartureArrivalSocAnalyzer,
        DepotEventAnalyzer,
        SpecificEnergyConsumptionAnalyzer,
        InteractiveMapAnalyzer,
    ]

    for analyzer_class in simple_analyzers:
        try:
            execute_and_save_simple_analyzer(analyzer_class, pipeline, output_directory)  # type: ignore[type-abstract]
        except Exception as e:
            print(f"Error running analyzer {analyzer_class.__name__}: {e}")

    # PowerAndOccupancyAnalyzer
    try:
        with pipeline.get_session() as session:
            area_ids_by_depot = {
                depot.name: [area.id for area in depot.areas]
                for depot in session.query(Depot).all()
            }

        for depot_name, area_ids in area_ids_by_depot.items():
            pipeline.params["PowerAndOccupancyAnalyzer.area_id"] = area_ids
            analyzer = PowerAndOccupancyAnalyzer()
            result = analyzer.execute(context=pipeline)
            vis = analyzer.visualize(result)
            output_file = output_directory / f"PowerAndOccupancyAnalyzer_{depot_name}.html"
            save_visualization(vis, output_file)
    except Exception as e:
        print(f"Error running PowerAndOccupancyAnalyzer: {e}")

    # SingleRotationInfoAnalyzer
    try:
        rotation_ids = query_all_ids(pipeline, Rotation)
        for rotation_id in rotation_ids[:10]:  # Limit for Izmir
            pipeline.params["SingleRotationInfoAnalyzer.rotation_id"] = rotation_id
            analyzer = SingleRotationInfoAnalyzer()
            result = analyzer.execute(context=pipeline)
            vis = analyzer.visualize(result)
            output_file = (
                output_directory / f"SingleRotationInfoAnalyzer_rotation_{rotation_id}.html"
            )
            save_visualization(vis, output_file, analyzer)
    except Exception as e:
        print(f"Error running SingleRotationInfoAnalyzer: {e}")

    # VehicleSocAnalyzer
    try:
        vehicle_ids = query_all_ids(pipeline, Vehicle)
        for vehicle_id in vehicle_ids[:10]:  # Limit for Izmir
            pipeline.params["VehicleSocAnalyzer.vehicle_id"] = vehicle_id
            analyzer = VehicleSocAnalyzer()
            result = analyzer.execute(context=pipeline)
            vis = analyzer.visualize(*result)
            output_file = output_directory / f"VehicleSocAnalyzer_vehicle_{vehicle_id}.html"
            save_visualization(vis, output_file)
    except Exception as e:
        print(f"Error running VehicleSocAnalyzer: {e}")


if __name__ == "__main__":
    main()
