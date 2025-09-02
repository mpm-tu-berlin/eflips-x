import glob
from datetime import timedelta
from pathlib import Path
from tempfile import gettempdir

from prefect import flow

from eflips.x.steps import (
    bvgxml_ingest_2025_06,
    remove_unused_vehicle_types,
    remove_unused_rotations,
    merge_stations,
    remove_unused_data,
    vehicle_type_and_depot_plot,
    reduce_to_one_day_two_depots,
    add_temperatures_and_consumption,
    depot_assignment,
    is_station_electrification_possible,
    do_station_electrification,
    run_simulation,
    vehicle_scheduling,
    calculate_tco,
)


@flow(name="prepare-dataset-bvg-brazil")
def prepare_dataset_bvg_brazil() -> Path:
    """
    Prepare the BVG Brazil dataset.
    :return: The path to the final database file.
    """
    base_dir = Path(__file__).parent.parent.parent.parent
    cache_dir = base_dir / "data" / "cache" / "bvg_brazil"
    output_dir = base_dir / "data" / "output" / "bvg_brazil"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    post_import_db_path = cache_dir / "01_post_import.db"
    path_as_str = Path(base_dir / "data" / "input" / "Berlin 2025-06" / "*.xml").as_posix()
    input_files = glob.glob(path_as_str)
    bvgxml_ingest_2025_06(input_db=None, output_db=post_import_db_path, input_files=input_files)

    post_remove_unused_vehicle_types_db_path = cache_dir / "02_post_remove_unused_vehicle_types.db"
    remove_unused_vehicle_types(
        input_db=post_import_db_path, output_db=post_remove_unused_vehicle_types_db_path
    )

    post_remove_unused_rotations_db_path = cache_dir / "03_post_remove_unused_rotations.db"
    remove_unused_rotations(
        input_db=post_remove_unused_vehicle_types_db_path,
        output_db=post_remove_unused_rotations_db_path,
    )

    post_more_station_merging_db_path = cache_dir / "04_post_more_station_merging.db"
    merge_stations(
        input_db=post_remove_unused_rotations_db_path,
        output_db=post_more_station_merging_db_path,
        max_distance_meters=200,
        match_percentage=50,
    )

    post_cleanup_path = cache_dir / "05_removed_unused_data.db"
    remove_unused_data(
        input_db=post_more_station_merging_db_path,
        output_db=post_cleanup_path,
    )
    return post_cleanup_path


@flow(name="prepare-dataset-one-day-two-depots")
def prepare_dataset_one_day_two_depots() -> None:
    """
    This flow
    - prepares simulation data
    - reduces the data to one day and two depots
    - creates plots of vehicle types and depots
    """
    base_dir = Path(__file__).parent.parent.parent.parent
    cache_dir = base_dir / "data" / "cache" / "one_day_two_depots"
    cache_dir.mkdir(parents=True, exist_ok=True)

    db_path = prepare_dataset_bvg_brazil()
    one_day_two_depot_path = cache_dir / "one_day_two_depots.db"
    reduce_to_one_day_two_depots(
        input_db=db_path,
        output_db=one_day_two_depot_path,
    )
    return one_day_two_depot_path


@flow(name="existing-depots-unchanged")
def prepare_and_simulate_existing_depots_unchanged() -> None:
    """
    This flow
    - prepares simulation data
    -
    """
    base_dir = Path(__file__).parent.parent.parent.parent
    output_dir = base_dir / "data" / "output" / "bvg_brazil" / "existing_depots_unchanged"
    cache_dir = base_dir / "data" / "cache" / "existing_depots_unchanged"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = prepare_dataset_one_day_two_depots()
    vehicle_type_and_depot_plot(db_path=db_path, output_path=output_dir)

    post_add_temperature_consumption_path = cache_dir / "01_temperature_consumption.db"
    add_temperatures_and_consumption(
        input_db=db_path,
        output_db=post_add_temperature_consumption_path,
    )

    post_scheduling_path = cache_dir / "01b_scheduling.db"
    vehicle_scheduling(
        input_db=post_add_temperature_consumption_path,
        output_db=post_scheduling_path,
        minimum_break_time=timedelta(seconds=0),
        maximum_schedule_duration=timedelta(hours=24),
        battery_margin=0.1,
    )

    post_depot_assignment_path = cache_dir / "02_depot_assignment.db"
    depot_assignment(
        input_db=post_scheduling_path,
        output_db=post_depot_assignment_path,
    )

    post_is_electrification_possible_path = (
        Path(gettempdir()) / "03_is_electrification_possible.db"
    )
    is_station_electrification_possible(
        input_db=post_depot_assignment_path,
        output_db=post_is_electrification_possible_path,
    )

    post_do_station_electrification_path = cache_dir / "04_do_station_electrification.db"
    do_station_electrification(
        input_db=post_depot_assignment_path,
        output_db=post_do_station_electrification_path,
    )

    post_run_simulation_path = cache_dir / "05_run_simulation.db"
    run_simulation(
        input_db=post_do_station_electrification_path,
        output_db=post_run_simulation_path,
    )

    post_calculate_tco_path = cache_dir / "06_calculate_tco.db"
    calculate_tco(
        input_db=post_run_simulation_path,
        output_db=post_calculate_tco_path,
    )

    return post_calculate_tco_path


if __name__ == "__main__":
    prepare_and_simulate_existing_depots_unchanged()
