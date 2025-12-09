#!/usr/bin/env python3

"""
Example flow demonstrating the eflips-x pipeline framework.

This example shows how to:
1. Ingest data from XML files
2. Clean and prepare the data
3. Run scheduling and simulation
4. Analyze and visualize results

The pipeline follows a simple pattern:
- Create pipeline steps (ingester, modifiers, analyzers)
- Execute them in sequence
- Save visualization outputs
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import gettempdir
from typing import List, Any, Dict, Type, Optional

import dash_cytoscape as cyto  # type: ignore[import-untyped]
import eflips.model
import folium  # type: ignore[import-untyped]
import matplotlib
import plotly  # type: ignore[import-untyped]
from eflips.model import ChargeType, Vehicle, Depot, Area, Rotation, Event
from prefect import flow

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
)
from eflips.x.steps.generators import BVGXMLIngester
from eflips.x.steps.modifiers.bvg_tools import (
    SetUpBvgVehicleTypes,
    RemoveUnusedRotations,
    MergeStations,
)
from eflips.x.steps.modifiers.general_utilities import RemoveUnusedData, AddTemperatures
from eflips.x.steps.modifiers.scheduling import VehicleScheduling, DepotAssignment
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation


def save_visualization(vis: Any, output_file: Path, analyzer: Optional[Analyzer] = None) -> None:
    """
    Save a visualization to a file based on its type.

    Args:
        vis: The visualization object (Plotly Figure, Folium Map, or Cytoscape)
        output_file: Path where the visualization should be saved
        analyzer: Optional analyzer instance (needed for Cytoscape exports)
    """
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
    """
    Execute a simple analyzer and save its visualization.

    Simple analyzers are those that don't require special parameters or
    have unique visualization handling.

    Args:
        analyzer_class: The analyzer class to instantiate and execute
        context: The pipeline context
        output_dir: Directory where output files should be saved
    """
    analyzer = analyzer_class()
    result = analyzer.execute(context=context)

    # Verify the analyzer has a visualize method
    if not hasattr(analyzer, "visualize"):
        raise AttributeError(f"{analyzer_class.__name__} has no visualize method")

    vis = analyzer.visualize(result)
    output_file = output_dir / f"{analyzer_class.__name__}.html"
    save_visualization(vis, output_file, analyzer)


def query_all_ids(
    context: PipelineContext, model_class: Type[eflips.model.Base], id_attr: str = "id"
) -> List[int]:
    """
    Query all IDs for a given model class.

    Args:
        context: The pipeline context
        model_class: The SQLAlchemy model class to query
        id_attr: The attribute name for the ID (default: "id")

    Returns:
        List of IDs
    """
    with context.get_session() as session:
        return [getattr(obj, id_attr) for obj in session.query(model_class).all()]


@flow
def run_steps(context: PipelineContext, steps: List[PipelineStep]) -> None:
    """Run a sequence of pipeline steps."""
    for step in steps:
        step.execute(context=context)


if __name__ == "__main__":
    ### Step 1: Initialize Pipeline ###
    # Create a unique working directory for this pipeline run
    work_dir = Path(gettempdir()) / (
        "eflips_example_flow_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {work_dir}")

    # Pipeline parameters can be discovered using the document_params() method
    # Global parameters have no prefix, step-specific parameters use the class name as prefix
    print("\nExample parameters for BVGXMLIngester:")
    for param, desc in BVGXMLIngester.document_params().items():
        print(f"  - {param}: {desc}")

    params: Dict[str, Any] = {
        "log_level": "INFO",
        "BVGXMLIngester.multithreading": True,
    }

    # Collect all pipeline steps in a list
    steps: List[PipelineStep] = []

    ### Step 2: Configure Data Ingestion ###
    # Locate and load XML input files
    path_to_this_file = Path(__file__).resolve()
    path_to_input_files = (
        path_to_this_file.parent.parent.parent.parent / "data" / "input" / "Berlin Testing"
    )
    input_files = list(path_to_input_files.glob("*.xml"))
    steps.append(BVGXMLIngester(input_files=input_files))

    ### Step 3: Clean and Prepare BVG Data ###
    # Set up vehicle types specific to BVG
    vehicle_types_setup = SetUpBvgVehicleTypes()
    steps.append(vehicle_types_setup)

    # Remove rotations that won't be used
    steps.append(RemoveUnusedRotations())

    # Merge nearby stations to simplify the network
    steps.append(MergeStations())

    ### Step 4: General Data Cleanup ###
    # Remove unnecessary data to improve processing speed
    steps.append(RemoveUnusedData())

    # Add temperature data (required for extended consumption simulation)
    # Skip this step if using constant consumption simulation
    steps.append(AddTemperatures())

    ### Step 5: Vehicle Scheduling (Optional) ###
    # The BVG data already contains schedules, but we re-schedule here
    # to demonstrate the scheduling functionality
    # Note: IntegratedScheduling is available for opportunity charging edge cases,
    # but is slower than regular scheduling
    params["VehicleScheduling.charge_type"] = ChargeType.DEPOT
    params["VehicleScheduling.minimum_break_time"] = timedelta(minutes=10)
    params["VehicleScheduling.maximum_schedule_duration"] = timedelta(hours=4)
    steps.append(VehicleScheduling())

    ### Step 6: Depot Assignment (Optional) ###
    # Since we re-scheduled, we need to re-assign depots
    # Configure a depot with coordinates and capacity
    vehicle_types = vehicle_types_setup._get_default_conversion_mapping().keys()
    depot_config = {
        "depot_station": (13.3509814, 52.5145556),  # (lon, lat)
        "name": "Manual Depot 1",
        "vehicle_type": vehicle_types,
        "capacity": 9999,
    }
    params["DepotAssignment.depot_config"] = [depot_config]
    steps.append(DepotAssignment())

    ### Step 7: Run Simulation ###
    # Generate depot infrastructure objects
    steps.append(DepotGenerator())

    # Run the actual vehicle and charging simulation
    steps.append(Simulation())

    ### Step 8: Execute Pipeline ###
    pipeline = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps=steps, context=pipeline)

    ### Step 9: Analyze and Visualize Results ###
    # Analyzers are run separately because they return output data rather than
    # modifying the database
    output_directory = work_dir / "analysis"

    # Simple analyzers that don't require special parameters
    simple_analyzers = [
        # Can be run before simulation
        RotationInfoAnalyzer,
        GeographicTripPlotAnalyzer,
        # Must be run after simulation
        DepartureArrivalSocAnalyzer,
        DepotEventAnalyzer,
        SpecificEnergyConsumptionAnalyzer,
    ]

    for analyzer_class in simple_analyzers:
        execute_and_save_simple_analyzer(analyzer_class, pipeline, output_directory)  # type: ignore[type-abstract]

    # PowerAndOccupancyAnalyzer requires area IDs (one analysis per depot)
    with pipeline.get_session() as session:
        area_ids_by_depot = {
            depot.name: [area.id for area in depot.areas] for depot in session.query(Depot).all()
        }

    analyzer: Analyzer  # To make Mypy happy, keep it generic
    for depot_name, area_ids in area_ids_by_depot.items():
        pipeline.params["PowerAndOccupancyAnalyzer.area_id"] = area_ids
        analyzer = PowerAndOccupancyAnalyzer()
        result = analyzer.execute(context=pipeline)
        vis = analyzer.visualize(result)
        output_file = output_directory / f"PowerAndOccupancyAnalyzer_{depot_name}.html"
        save_visualization(vis, output_file)

    # SingleRotationInfoAnalyzer requires a rotation ID (one analysis per rotation)
    rotation_ids = query_all_ids(pipeline, Rotation)
    for rotation_id in rotation_ids:
        pipeline.params["SingleRotationInfoAnalyzer.rotation_id"] = rotation_id
        analyzer = SingleRotationInfoAnalyzer()
        result = analyzer.execute(context=pipeline)
        vis = analyzer.visualize(result)
        output_file = output_directory / f"SingleRotationInfoAnalyzer_rotation_{rotation_id}.html"
        save_visualization(vis, output_file, analyzer)

    # VehicleSocAnalyzer requires a vehicle ID (one analysis per vehicle)
    vehicle_ids = query_all_ids(pipeline, Vehicle)
    for vehicle_id in vehicle_ids:
        pipeline.params["VehicleSocAnalyzer.vehicle_id"] = vehicle_id
        analyzer = VehicleSocAnalyzer()
        result = analyzer.execute(context=pipeline)
        vis = analyzer.visualize(*result)  # Note: unpacks result tuple
        output_file = output_directory / f"VehicleSocAnalyzer_vehicle_{vehicle_id}.html"
        save_visualization(vis, output_file)

    # DepotActivityAnalyzer generates videos (disabled by default as it's slow)
    GENERATE_DEPOT_VIDEOS = False  # Set to True to enable video generation

    if GENERATE_DEPOT_VIDEOS:
        logger = logging.getLogger(__name__)
        logger.info("Generating depot activity videos (this may take a while)...")

        depot_ids = query_all_ids(pipeline, Depot)
        for depot_id in depot_ids:
            # Query time range for this depot's events
            with pipeline.get_session() as session:
                first_event = (
                    session.query(Event)
                    .join(Area)
                    .filter(Area.depot_id == depot_id)
                    .order_by(Event.time_start.asc())
                    .first()
                )
                last_event = (
                    session.query(Event)
                    .join(Area)
                    .filter(Area.depot_id == depot_id)
                    .order_by(Event.time_end.desc())
                    .first()
                )

            if not first_event or not last_event:
                logger.warning(f"No events found for depot ID {depot_id}, skipping")
                continue

            # Configure and execute analyzer
            pipeline.params["DepotActivityAnalyzer.depot_id"] = depot_id
            pipeline.params["DepotActivityAnalyzer.animation_start"] = first_event.time_start
            pipeline.params["DepotActivityAnalyzer.animation_end"] = last_event.time_end

            analyzer = DepotActivityAnalyzer()
            result = analyzer.execute(context=pipeline)

            with pipeline.get_session() as session:
                vis = analyzer.visualize(
                    result,
                    animation_range=(first_event.time_start, last_event.time_end),
                    depot_id=depot_id,
                    session=session,
                )

            output_file = output_directory / f"DepotActivityAnalyzer_depot_{depot_id}.mp4"
            save_visualization(vis, output_file)
    else:
        logger = logging.getLogger(__name__)
        logger.info(
            "Skipping DepotActivityAnalyzer (video generation disabled). "
            "Set GENERATE_DEPOT_VIDEOS = True to enable."
        )
