import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from tempfile import gettempdir
from typing import Tuple, Dict, List, Any
from urllib.parse import urlparse

import eflips.model
import pytz
import sqlalchemy
from alembic import command
from alembic.config import Config
from eflips.depot.api import (
    group_rotations_by_start_end_stop,
    ConsumptionResult,
    simple_consumption_simulation,
    generate_depot_layout,
    init_simulation,
    run_simulation,
    add_evaluation_to_database,
    insert_dummy_standby_departure_events,
    apply_even_smart_charging,
    generate_consumption_result,
)
from eflips.model import (
    Event,
    Rotation,
    Scenario,
    Vehicle,
    Station,
    VehicleType,
    ChargeType,
    VoltageLevel,
    Trip,
    EventType,
    Area,
    Process,
    Depot,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload
from tqdm.asyncio import tqdm

tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)


def database_url_components(database_url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extracts the components of a database URL.
    :param database_url: The URL of the database.
    :return: A tuple with the components of the URL: protocol, user, password, host, port, database name.
    """
    o = urlparse(database_url)
    if o.scheme != "postgresql":
        raise ValueError("Only PostgreSQL databases are supported.")
    if o.port is None:
        port = "5432"
    else:
        port = str(o.port)
    return o.scheme, o.username, o.password, o.hostname, port, o.path[1:]


def clear_previous_simulation_results(scenario: Scenario, session: Session) -> None:
    session.query(Rotation).filter(Rotation.scenario_id == scenario.id).update(
        {"vehicle_id": None}
    )
    session.query(Event).filter(Event.scenario_id == scenario.id).delete()
    session.query(Vehicle).filter(Vehicle.scenario_id == scenario.id).delete()


def make_depot_stations_electrified(scenario: Scenario, session: Session):
    """
    Before runing SimBA for the first time, we need to make sure that the depot stations (The ones where rotations
    start and end) are electrified.
    :param scenario: The scenario to electrify.
    :param session: An open database session.
    :return: Nothing. The function modifies the database.
    """
    rotations_by_start_end_stop: Dict[
        Tuple[Station, Station], Dict[VehicleType, List[Rotation]]
    ] = group_rotations_by_start_end_stop(scenario.id, session)
    for (start, end), _ in rotations_by_start_end_stop.items():
        if start != end:
            raise ValueError(f"Start and end station are not the same: {start} != {end}")
        if not start.is_electrified:
            start.is_electrified = True
            start.amount_charging_places = 100
            start.power_per_charger = 300
            start.power_total = start.amount_charging_places * start.power_per_charger
            start.charge_type = ChargeType.DEPOT
            start.voltage_level = VoltageLevel.MV


def get_alembic_ini_path() -> str:
    """
    Retrieve the path to the `alembic.ini` file located within the `eflips.model` package.

    This function constructs the path to the `alembic.ini` file by determining
    the directory of the `eflips.model` package and appending the relative path
    to the `alembic.ini` file. It checks if the file exists and raises a
    FileNotFoundError if it does not.

    Returns:
        str: The absolute path to the `alembic.ini` file.

    Raises:
        FileNotFoundError: If the `alembic.ini` file does not exist in the expected location.
    """
    # Get the directory of the `eflips.model` package
    package_dir = Path(eflips.model.__file__).parent

    # Construct the path to `alembic.ini` (relative to the package)
    alembic_ini_path = package_dir / "alembic.ini"

    # Verify the file exists (optional but recommended)
    if not alembic_ini_path.exists():
        raise FileNotFoundError(f"'{alembic_ini_path}' does not exist. Check package structure.")
    return str(alembic_ini_path)


def update_database_to_most_recent_schema() -> None:
    """
    Update the database schema to the most recent state using Alembic migrations.

    This function performs the following steps:
    1. Checks if the `DATABASE_URL` environment variable is set.
    2. Retrieves the path to the `alembic.ini` file using the `get_alembic_ini_path` function.
    3. Configures Alembic with the retrieved `alembic.ini` path.
    4. Upgrades the database schema to the latest version ("head").

    Raises:
        ValueError: If the `DATABASE_URL` environment variable is not set.
        Exception: If any other error occurs during the migration process, such as a configuration error or database connection issues.
    """

    logger = logging.getLogger(__name__)

    if "DATABASE_URL" not in os.environ:
        raise ValueError(
            "Please set the DATABASE_URL environment variable."
            "Other ways to set the database URL are not supported."
        )

    try:
        alembic_ini_path = get_alembic_ini_path()
        config = Config(alembic_ini_path)
        # The "migrations" folder is in "migrations" *relative* to the `alembic.ini` file
        alembic_ini_dir = Path(alembic_ini_path).parent
        config.set_main_option("script_location", str(alembic_ini_dir / "migrations"))
        command.upgrade(config, "head")
        logger.info("Database upgraded successfully.")
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        raise e


def clear_sim_results(scenario, session):
    session.query(Event).filter(Event.scenario_id == scenario.id).delete()
    session.query(Rotation).filter(Rotation.scenario_id == scenario.id).update(
        {"vehicle_id": None}
    )
    session.query(Vehicle).filter(Vehicle.scenario_id == scenario.id).delete()


def update_trip_loaded_masses(scenario: Scenario, session: Session) -> None:
    """
    Set all trip's loaded mass to half of the maximum capacity.
    :param scenario: The scenario to use.
    :param session: The database session.
    """
    # We need to set the loaded mass for each trip
    # Set all trip's loaded mass to 17.6 passengers, the average for the Germany
    PASSENGER_MASS = 68  # kg
    PASSENGER_COUNT = 17.6  # German-wide average
    payload = PASSENGER_COUNT * PASSENGER_MASS
    session.query(Trip).filter(Trip.scenario == scenario).update({"loaded_mass": payload})


def _progress_process_method(total_count: int, queue: Queue):
    """
    This method reports progress. It sets up a tqdm, gets a count from the progress queue and updates the command line.
     It may later be overridden / monkey-patched to do analytics.
    Args:
        total_count: The total amount of iterations (may be routes, trips, generations for evolutionary algorithms…
                     Each iteration should take a similar amount of wallclock time.
        queue: the progress queue object. Set automatically from `train_in_parallel()`

    Returns:
        Nothing.

    """
    progress_reporter = tqdm(total=total_count, smoothing=0)
    while True:
        incremental_progress = queue.get()
        progress_reporter.update(incremental_progress)
