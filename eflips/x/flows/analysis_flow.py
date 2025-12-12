#!/usr/bin/env python3

"""
Reusable analysis flow for generating visualizations from eflips-x pipeline results.

This flow provides a parallelized approach to generating all analysis plots from an
existing PipelineContext. It organizes plots into a sensible folder structure and
uses process-based parallelism for maximum performance.
"""
import logging
import multiprocessing
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type

import dash_cytoscape as cyto  # type: ignore[import-untyped]
import eflips.model
import folium  # type: ignore[import-untyped]
import matplotlib
import plotly  # type: ignore[import-untyped]
from eflips.model import Area, Depot, Event, Rotation, Station, Vehicle
from prefect import flow, task
from prefect.futures import wait
from prefect.task_runners import ProcessPoolTaskRunner

from eflips.x.framework import Analyzer, PipelineContext
from eflips.x.steps.analyzers import (
    DepartureArrivalSocAnalyzer,
    DepotActivityAnalyzer,
    DepotEventAnalyzer,
    GeographicTripPlotAnalyzer,
    InteractiveMapAnalyzer,
    PowerAndOccupancyAnalyzer,
    RotationInfoAnalyzer,
    SingleRotationInfoAnalyzer,
    SpecificEnergyConsumptionAnalyzer,
    VehicleSocAnalyzer,
)

logger = logging.getLogger(__name__)


def _optional_path_to_str(path: Optional[Path]) -> Optional[str]:
    """Convert optional Path to optional string.

    Args:
        path: Path object or None

    Returns:
        String representation of path, or None if path is None
    """
    return str(path) if path is not None else None


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

    logger.info(f"Wrote analysis output to: {output_file}")


@task
def execute_simple_analyzer(
    analyzer_class: Type[Analyzer],
    context: PipelineContext,
    output_file: Path,
) -> None:
    """
    Execute a simple analyzer and save its visualization.

    This task is designed to run in parallel using ProcessPoolTaskRunner.

    Args:
        analyzer_class: The analyzer class to instantiate and execute
        context: The pipeline context
        output_file: Full path where the output file should be saved
    """
    analyzer = analyzer_class()

    # Create temporary database for the analyzer
    result = analyzer.execute(context=context)

    # Verify the analyzer has a visualize method
    if not hasattr(analyzer, "visualize"):
        raise AttributeError(f"{analyzer_class.__name__} has no visualize method")

    vis = analyzer.visualize(result)
    save_visualization(vis, output_file, analyzer)


@task
def execute_interactive_map_analyzer(
    context: PipelineContext,
    output_file: Path,
    depot_plot_dir: Optional[Path] = None,
    station_plot_dir: Optional[Path] = None,
) -> None:
    """
    Execute InteractiveMapAnalyzer with optional directory parameters.

    This task is designed to run in parallel using ProcessPoolTaskRunner.

    Args:
        context: The pipeline context
        output_file: Full path where the output file should be saved
        depot_plot_dir: Optional directory containing depot plots (depot_{id}.html)
        station_plot_dir: Optional directory containing station plots (station_{id}.html)
    """
    analyzer = InteractiveMapAnalyzer()
    result = analyzer.execute(context=context)

    # Convert Path to string for visualize method
    depot_dir_str = _optional_path_to_str(depot_plot_dir)
    station_dir_str = _optional_path_to_str(station_plot_dir)

    vis = analyzer.visualize(
        result, station_plot_dir=station_dir_str, depot_plot_dir=depot_dir_str
    )
    save_visualization(vis, output_file, analyzer)


@task
def execute_rotation_analyzer(
    rotation_id: int, context: PipelineContext, output_file: Path
) -> None:
    """
    Execute SingleRotationInfoAnalyzer for a specific rotation.

    Args:
        rotation_id: The rotation ID to analyze
        context: The pipeline context
        output_file: Full path where the output file should be saved
    """
    # Create a copy of the context params to avoid mutation
    params = context.params.copy()
    params["SingleRotationInfoAnalyzer.rotation_id"] = rotation_id

    # Create new context with updated params
    analysis_context = PipelineContext(
        work_dir=context.work_dir,
        params=params,
        current_db=context.current_db,
    )

    analyzer = SingleRotationInfoAnalyzer()
    result = analyzer.execute(context=analysis_context)
    vis = analyzer.visualize(result)
    save_visualization(vis, output_file, analyzer)


@task
def execute_vehicle_analyzer(vehicle_id: int, context: PipelineContext, output_file: Path) -> None:
    """
    Execute VehicleSocAnalyzer for a specific vehicle.

    Args:
        vehicle_id: The vehicle ID to analyze
        context: The pipeline context
        output_file: Full path where the output file should be saved
    """
    params = context.params.copy()
    params["VehicleSocAnalyzer.vehicle_id"] = vehicle_id

    analysis_context = PipelineContext(
        work_dir=context.work_dir,
        params=params,
        current_db=context.current_db,
    )

    analyzer = VehicleSocAnalyzer()
    result = analyzer.execute(context=analysis_context)
    vis = analyzer.visualize(*result)  # Note: unpacks result tuple
    save_visualization(vis, output_file)


@task
def execute_power_occupancy_analyzer(
    depot_name: str, area_ids: List[int], context: PipelineContext, output_file: Path
) -> None:
    """
    Execute PowerAndOccupancyAnalyzer for a specific depot.

    Args:
        depot_name: Name of the depot (used in filename)
        area_ids: List of area IDs for this depot
        context: The pipeline context
        output_file: Full path where the output file should be saved
    """
    params = context.params.copy()
    params["PowerAndOccupancyAnalyzer.area_id"] = area_ids

    analysis_context = PipelineContext(
        work_dir=context.work_dir,
        params=params,
        current_db=context.current_db,
    )

    analyzer = PowerAndOccupancyAnalyzer()
    result = analyzer.execute(context=analysis_context)
    vis = analyzer.visualize(result)
    save_visualization(vis, output_file)


@task
def execute_station_analyzer(station_id: int, context: PipelineContext, output_file: Path) -> None:
    """
    Execute PowerAndOccupancyAnalyzer for a specific station.

    Args:
        station_id: ID of the electrified station
        context: The pipeline context
        output_file: Full path where the output file should be saved
    """
    params = context.params.copy()
    params["PowerAndOccupancyAnalyzer.station_id"] = station_id
    params["PowerAndOccupancyAnalyzer.area_id"] = None  # Will become [] in analyzer

    analysis_context = PipelineContext(
        work_dir=context.work_dir,
        params=params,
        current_db=context.current_db,
    )

    analyzer = PowerAndOccupancyAnalyzer()
    result = analyzer.execute(context=analysis_context)
    vis = analyzer.visualize(result)
    save_visualization(vis, output_file)


@task
def execute_depot_activity_analyzer(
    depot_id: int,
    time_range: Tuple[datetime, datetime],
    context: PipelineContext,
    output_file: Path,
) -> None:
    """
    Execute DepotActivityAnalyzer for a specific depot (generates video).

    Args:
        depot_id: The depot ID to analyze
        time_range: Tuple of (start_time, end_time) for the animation
        context: The pipeline context
        output_file: Full path where the output file should be saved
    """
    params = context.params.copy()
    params["DepotActivityAnalyzer.depot_id"] = depot_id
    params["DepotActivityAnalyzer.animation_start"] = time_range[0]
    params["DepotActivityAnalyzer.animation_end"] = time_range[1]

    analysis_context = PipelineContext(
        work_dir=context.work_dir,
        params=params,
        current_db=context.current_db,
    )

    analyzer = DepotActivityAnalyzer()
    result = analyzer.execute(context=analysis_context)

    # Need a session for visualization
    with analysis_context.get_session() as session:
        vis = analyzer.visualize(
            result,
            animation_range=time_range,
            depot_id=depot_id,
            session=session,
        )

    save_visualization(vis, output_file)


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


@flow(
    name="analysis-flow",
    task_runner=ProcessPoolTaskRunner(max_workers=multiprocessing.cpu_count()),  # type: ignore[arg-type]
)
def generate_all_plots(
    context: PipelineContext,
    output_dir: Path,
    include_videos: bool = False,
    pre_simulation_only: bool = False,
) -> None:
    """
    Generate all analysis plots from a pipeline context.

    This flow runs analyzers in parallel where possible and organizes outputs
    into a structured directory hierarchy:

    output_dir/
        simple/           - Simple analyzers (no parameters)
        rotations/        - Per-rotation analyses
        vehicles/         - Per-vehicle analyses
        depots/           - Per-depot power and occupancy
        videos/           - Depot activity videos (optional)

    Args:
        context: The pipeline context containing the database to analyze
        output_dir: Root directory for saving all analysis outputs
        include_videos: Whether to generate depot activity videos (slow)
        pre_simulation_only: If True, only run analyzers that work before simulation
    """
    logger.info(f"Starting analysis flow, outputs will be saved to: {output_dir}")

    # Define output directories
    simple_dir = output_dir / "simple"
    rotations_dir = output_dir / "rotations"
    vehicles_dir = output_dir / "vehicles"
    depots_dir = output_dir / "depots"
    stations_dir = output_dir / "stations"
    videos_dir = output_dir / "videos"

    # Create directories
    for directory in [
        simple_dir,
        rotations_dir,
        vehicles_dir,
        depots_dir,
        stations_dir,
        videos_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # Define simple analyzers by category
    pre_simulation_analyzers = [
        RotationInfoAnalyzer,
        GeographicTripPlotAnalyzer,
    ]

    post_simulation_analyzers = [
        DepartureArrivalSocAnalyzer,
        DepotEventAnalyzer,
        SpecificEnergyConsumptionAnalyzer,
    ]

    # Run simple analyzers in parallel
    if pre_simulation_only:
        analyzers_to_run = pre_simulation_analyzers
        logger.info("Running pre-simulation analyzers only")
    else:
        analyzers_to_run = pre_simulation_analyzers + post_simulation_analyzers
        logger.info("Running all simple analyzers")

    all_futures = []
    for analyzer_class in analyzers_to_run:
        output_file = simple_dir / f"{analyzer_class.__name__}.html"
        future = execute_simple_analyzer.submit(analyzer_class, context, output_file)  # type: ignore[type-abstract]
        all_futures.append(future)

    # Execute InteractiveMapAnalyzer with depot and station plot directories
    logger.info("Submitting InteractiveMapAnalyzer...")
    map_output_file = simple_dir / "InteractiveMapAnalyzer.html"

    # Determine plot directories based on mode
    depot_dir = None if pre_simulation_only else depots_dir
    station_dir = None if pre_simulation_only else stations_dir

    future = execute_interactive_map_analyzer.submit(
        context,
        map_output_file,
        depot_plot_dir=depot_dir,
        station_plot_dir=station_dir,
    )
    all_futures.append(future)

    # If pre-simulation only, skip the rest
    if pre_simulation_only:
        # Wait for simple analyzers to finish
        logger.info("Waiting for pre-simulation analyzers to complete...")
        wait(all_futures)
        logger.info("Pre-simulation analysis complete")
        return

    # Run rotation-specific analyzers in parallel
    logger.info("Querying rotation IDs for analysis...")
    rotation_ids = query_all_ids(context, Rotation)
    logger.info(f"Found {len(rotation_ids)} rotations to analyze")

    for rotation_id in rotation_ids:
        output_file = rotations_dir / f"rotation_{rotation_id}.html"
        future = execute_rotation_analyzer.submit(rotation_id, context, output_file)
        all_futures.append(future)

    # Run vehicle-specific analyzers in parallel
    logger.info("Querying vehicle IDs for analysis...")
    vehicle_ids = query_all_ids(context, Vehicle)
    logger.info(f"Found {len(vehicle_ids)} vehicles to analyze")

    for vehicle_id in vehicle_ids:
        output_file = vehicles_dir / f"vehicle_{vehicle_id}.html"
        future = execute_vehicle_analyzer.submit(vehicle_id, context, output_file)
        all_futures.append(future)

    # Run depot-specific power and occupancy analyzers in parallel
    logger.info("Querying depot information for analysis...")
    with context.get_session() as session:
        depot_info = {
            depot.id: {"name": depot.name, "area_ids": [area.id for area in depot.areas]}
            for depot in session.query(Depot).all()
        }

    logger.info(f"Found {len(depot_info)} depots to analyze")
    for depot_id, info in depot_info.items():
        output_file = depots_dir / f"depot_{depot_id}.html"
        future = execute_power_occupancy_analyzer.submit(  # type: ignore[call-overload]
            info["name"], info["area_ids"], context, output_file
        )
        all_futures.append(future)

    # Run station-specific power and occupancy analyzers for electrified stations
    if not pre_simulation_only:
        logger.info("Querying electrified stations for analysis...")
        with context.get_session() as session:
            electrified_stations = session.query(Station).filter_by(is_electrified=True).all()
            station_ids = [station.id for station in electrified_stations]

        logger.info(f"Found {len(station_ids)} electrified stations to analyze")
        for station_id in station_ids:
            output_file = stations_dir / f"station_{station_id}.html"
            future = execute_station_analyzer.submit(station_id, context, output_file)
            all_futures.append(future)

    # Optionally run depot activity analyzers (videos)
    if include_videos:
        logger.info("Generating depot activity videos (this may take a while)...")
        depot_ids = query_all_ids(context, Depot)

        for depot_id in depot_ids:
            # Query time range for this depot's events
            with context.get_session() as session:
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

            time_range = (first_event.time_start, last_event.time_end)
            output_file = videos_dir / f"depot_{depot_id}_activity.mp4"
            future = execute_depot_activity_analyzer.submit(
                depot_id, time_range, context, output_file
            )
            all_futures.append(future)
    else:
        logger.info("Skipping depot activity videos. Set include_videos=True to enable.")

    logger.info("All analysis tasks submitted. Waiting for completion...")
    wait(all_futures)
    logger.info("Analysis flow complete. All outputs saved.")
