#! /usr/bin/env python3
import multiprocessing
import os
from datetime import datetime, timedelta
from operator import or_
from queue import Queue
from typing import List, Tuple

import eflips.eval.output.prepare
import eflips.eval.output.visualize
import pytz
from eflips.model import (
    Scenario,
    Depot,
    Vehicle,
    Area,
    AreaType,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm.auto import tqdm

from util import _progress_process_method

PARALLELISM = True
MAKE_MOVIES = False

tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)

OUTPUT_FOLDER = "output_interactive"


def progress_callback(current_frame: int, total_frames: int) -> None:
    print(f"Processing frame {current_frame}/{total_frames}")


def inner_generate_vehicle_plot(
    vehicle_id: int, progress_queue: Queue | None = None
) -> None:
    logger = multiprocessing.get_logger()

    DATABASE_URL = os.environ["DATABASE_URL"]
    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    scenario = (
        session.query(Scenario).join(Vehicle).filter(Vehicle.id == vehicle_id).one()
    )
    vehicle = session.query(Vehicle).filter(Vehicle.id == vehicle_id).one()
    folder_name = os.path.join(OUTPUT_FOLDER, scenario.name_short)
    vehicle_folder = os.path.join(folder_name, "vehicles")
    assert os.path.exists(vehicle_folder), f"Folder {vehicle_folder} does not exist"
    output_file_name = os.path.join(vehicle_folder, f"{vehicle.id}.html")
    if not os.path.exists(output_file_name):
        df, descriptions = eflips.eval.output.prepare.vehicle_soc(vehicle.id, session)
        fig = eflips.eval.output.visualize.vehicle_soc(df, descriptions)
        fig.update_layout(
            title=f"Vehicle {vehicle.id} of type {vehicle.vehicle_type.name_short}"
        )
        fig.update_layout(xaxis_range=[START_OF_SIMULATION, END_OF_SIMULATION])
        fig.write_html(output_file_name)
    else:
        logger.info(f"Skipping vehicle {vehicle.id} as it already exists")

    session.close()
    engine.dispose()

    if progress_queue is not None:
        progress_queue.put(1)


if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    SCENARIO_NAMES = ["TERM", "DEP", "OU"]

    for scenario_name_short in SCENARIO_NAMES:
        folder_name = os.path.join(OUTPUT_FOLDER, scenario_name_short)
        os.makedirs(folder_name, exist_ok=True)

        # For each vehiclle plot what it's doing
        scenario = (
            session.query(Scenario)
            .filter(Scenario.name_short == scenario_name_short)
            .first()
        )
        vehicles_q = session.query(Vehicle).filter(Vehicle.scenario == scenario)

        vehicle_folder = os.path.join(folder_name, "vehicles")
        os.makedirs(vehicle_folder, exist_ok=True)

        # This is my "parallelism with progress" pattern
        # Use my "parallelism with progress" boilerplate
        parallelism = True
        if parallelism:
            # Set up progress reporting
            progress_manager = multiprocessing.Manager()
            progress_queue = progress_manager.Queue()
            progress_process = multiprocessing.Process(
                target=_progress_process_method,
                args=(vehicles_q.count(), progress_queue),
            )
            progress_process.start()

            pool_args: List[Tuple[int, multiprocessing.Queue]] = []
            for vehicle in vehicles_q:
                pool_args.append((vehicle.id, progress_queue))

            with multiprocessing.Pool() as p:
                results = p.starmap(inner_generate_vehicle_plot, pool_args)

            # Remember to kill the progress process
            progress_process.kill()
        else:
            results = list()
            for vehicle in tqdm(vehicles_q.all(), smoothing=0):
                results.append(inner_generate_vehicle_plot(vehicle.id, None))

        if MAKE_MOVIES:
            # Depot Movies
            depot_folder = os.path.join(folder_name, "depots")
            depots = session.query(Depot).filter(Depot.scenario == scenario).all()
            os.makedirs(depot_folder, exist_ok=True)
            for depot in tqdm(depots, desc=f"Depot {scenario_name_short}"):
                # We will need to modify the depot a little. We will need to add to cap the size of the direct areas at
                # their maximum usage
                direct_area_q = (
                    session.query(Area)
                    .filter(Area.depot_id == depot.id)
                    .filter(
                        or_(
                            Area.area_type == AreaType.DIRECT_ONESIDE,
                            Area.area_type == AreaType.DIRECT_TWOSIDE,
                        )
                    )
                )
                for direct_area in direct_area_q:
                    direct_area: Area
                    area_id = direct_area.id
                    try:
                        occupancy_df = eflips.eval.output.prepare.power_and_occupancy(
                            area_id,
                            session,
                            sim_start_time=START_OF_SIMULATION,
                            sim_end_time=END_OF_SIMULATION,
                        )
                        max_occupancy = max(occupancy_df["occupancy"])
                        direct_area.capacity = max_occupancy
                    except ValueError:
                        # This means nothing is happening at the area. Set the capacity to 1
                        direct_area.capacity = 1
                session.flush()
                session.expire_all()

                time_range = (START_OF_SIMULATION, END_OF_SIMULATION)
                area_blocks = eflips.eval.output.prepare.depot_layout(depot.id, session)
                depot_activity = eflips.eval.output.prepare.depot_activity(
                    depot.id, session, time_range
                )
                animation = eflips.eval.output.visualize.depot_activity_animation(
                    area_blocks, depot_activity, time_range
                )
                animation.save(
                    filename=os.path.join(depot_folder, f"{depot.name}.mp4"),
                    writer="ffmpeg",
                    fps=5,
                    progress_callback=progress_callback,
                )
