import glob
from pathlib import Path

from prefect import task, flow

from eflips.x.steps import (
    bvgxml_ingest_2025_06,
    remove_unused_vehicle_types,
    remove_unused_rotations,
    merge_stations,
    remove_unused_data,
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


@flow(name="existing-depots-unchanged")
def existing_depots_unchanged() -> None:
    """
    This flow
    - prepares simulation data
    -
    """
    raise NotImplementedError


if __name__ == "__main__":
    prepare_dataset_bvg_brazil()
