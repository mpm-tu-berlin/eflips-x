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
from ds_wrapper import DjangoSimbaWrapper
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
            raise ValueError(
                f"Start and end station are not the same: {start} != {end}"
            )
        if not start.is_electrified:
            start.is_electrified = True
            start.amount_charging_places = 100
            start.power_per_charger = 300
            start.power_total = start.amount_charging_places * start.power_per_charger
            start.charge_type = ChargeType.DEPOT
            start.voltage_level = VoltageLevel.MV


def create_consumption_results(
    scenario: Scenario, session: sqlalchemy.orm.session.Session, database_url: str
) -> Dict[int, ConsumptionResult]:
    logger = logging.getLogger(__name__)

    CACHE_FILE_PATH = os.path.join(
        gettempdir(), f"consumption_cache_{scenario.id}_{scenario.name_short}.pkl"
    )
    if os.path.exists(CACHE_FILE_PATH):
        logger.info("Loading consumption results from cache")
        with open(CACHE_FILE_PATH, "rb") as f:
            consumption_results = pickle.load(f)
        return consumption_results
    else:
        # We need to remove all electric charging stations from the database
        # SO we can look at "which rotations will be okay without terminus charging"?
        # As well, django-simba messes up otherwise.
        # We will need to remember them, including their electrification parameters. Which are:
        # is_electrified
        # amount_charging_places
        # power_per_charger
        # power_total
        # charge_type
        already_elecrified_stations: List[Dict[str, Any]] = []
        non_terminus_elecrified_stations_q = (
            session.query(Station)
            .filter(Station.scenario_id == scenario.id)
            .filter(Station.is_electrified)
            .filter(Station.charge_type != ChargeType.DEPOT)
        )
        for station in tqdm(
            non_terminus_elecrified_stations_q.all(), desc="Un-Electrifying stations"
        ):
            already_elecrified_stations.append(
                {
                    "id": station.id,
                    "is_electrified": station.is_electrified,
                    "amount_charging_places": station.amount_charging_places,
                    "power_per_charger": station.power_per_charger,
                    "power_total": station.power_total,
                    "charge_type": station.charge_type,
                    "voltage_level": station.voltage_level,
                }
            )
            station.is_electrified = False
            station.amount_charging_places = None
            station.power_per_charger = None
            station.power_total = None
            station.charge_type = None
            station.voltage_level = None

        logger.info("Running consumption simulation with Django-Simba")
        session.commit()
        ds_wrapper = DjangoSimbaWrapper(database_url)
        ds_wrapper.run_simba_scenario(scenario.id, assign_vehicles=True)
        del ds_wrapper
        session.commit()
        session.expire_all()

        # Now, we need to re-electrify the stations
        for station_data in tqdm(
            already_elecrified_stations, desc="Re-electrifying stations"
        ):
            station = (
                session.query(Station).filter(Station.id == station_data["id"]).one()
            )
            station.is_electrified = station_data["is_electrified"]
            station.amount_charging_places = station_data["amount_charging_places"]
            station.power_per_charger = station_data["power_per_charger"]
            station.power_total = station_data["power_total"]
            station.charge_type = station_data["charge_type"]
            station.voltage_level = station_data["voltage_level"]

        consumption_results: Dict[int, ConsumptionResult] = {}
        for trip in (
            session.query(Trip)
            .filter(Trip.scenario == scenario)
            .options(joinedload(Trip.events))
        ):
            event = trip.events[0]
            event: Event
            assert event.event_type == EventType.DRIVING
            consumption_results[trip.id] = ConsumptionResult(
                delta_soc_total=event.soc_end
                - event.soc_start,  # So the number is negative
                timestamps=None,
                delta_soc=None,
            )
            if event.timeseries is not None:
                consumption_results[trip.id].timestamps = [
                    datetime.fromisoformat(t) for t in event.timeseries["time"]
                ]
                consumption_results[trip.id].soc = event.timeseries["soc"]

        with open(CACHE_FILE_PATH, "wb") as f:
            pickle.dump(consumption_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return consumption_results


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
        raise FileNotFoundError(
            f"'{alembic_ini_path}' does not exist. Check package structure."
        )
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
    session.query(Trip).filter(Trip.scenario == scenario).update(
        {"loaded_mass": payload}
    )


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


def _inner_run_simulation(
    scenario_id: int, database_url: str, USE_SIMBA_CONSUMPTION: bool = True
):
    engine = create_engine(database_url)
    session = Session(engine)
    scenario = session.query(Scenario).filter(Scenario.id == scenario_id).one()

    #### Step 0: Make depot stations electrified #####
    update_trip_loaded_masses(scenario, session)
    make_depot_stations_electrified(scenario, session)

    ##### Step 1: Consumption simulation
    clear_previous_simulation_results(scenario, session)
    if USE_SIMBA_CONSUMPTION:
        # If DATABASE_URL is not defined, take it from the environment
        if "DATABASE_URL" not in locals():
            DATABASE_URL = os.environ["DATABASE_URL"]

        # If using simba consumption, we will run it once to determine the delta SoC for each trip
        # Then store these in a dictionary and use simple consumption simulation in "Predefined delta SoC" mode
        consumption_results = create_consumption_results(
            scenario, session, DATABASE_URL
        )
        clear_previous_simulation_results(scenario, session)

        simple_consumption_simulation(
            initialize_vehicles=True,
            scenario=scenario,
            consumption_result=consumption_results,
        )
    else:
        simple_consumption_simulation(scenario, initialize_vehicles=True)

    ##### Step 2: Generate the depot layout
    generate_depot_layout(
        scenario=scenario, charging_power=120, delete_existing_depot=True
    )

    # TODO: Remove if unneeded
    # octuple the capacity of all areas
    for area in session.query(Area).filter(Area.scenario_id == scenario.id):
        area.capacity *= 4

    # Set cleaning area capacity to 20
    for area in (
        session.query(Area)
        .filter(Area.scenario_id == scenario.id)
        .filter(Area.name == "Cleaning Area")
    ):
        area.capacity = 20

    # Set cleaning duration to 30 minutes
    for process in (
        session.query(Process)
        .filter(Process.scenario_id == scenario.id)
        .filter(Process.name == "Arrival Cleaning")
    ):
        process.duration = timedelta(minutes=30)

    # Set shunting area capacity to 20
    for area in (
        session.query(Area)
        .filter(Area.scenario_id == scenario.id)
        .filter(Area.name.like("Shunting Area%"))
    ):
        area.capacity = 20

    ##### Step 3: Run the simulation
    # This can be done using eflips.api.run_simulation. Here, we use the three steps of
    # eflips.api.init_simulation, eflips.api.run_simulation, and eflips.api.add_evaluation_to_database
    # in order to show what happens "under the hood".
    simulation_host = init_simulation(
        scenario=scenario,
        session=session,
        repetition_period=timedelta(days=7),
    )
    depot_evaluations = run_simulation(simulation_host)

    add_evaluation_to_database(scenario, depot_evaluations, session)
    session.flush()
    session.expire_all()

    ##### Step 4: Consumption simulation
    if USE_SIMBA_CONSUMPTION:
        simple_consumption_simulation(
            initialize_vehicles=False,
            scenario=scenario,
            consumption_result=consumption_results,
        )
    else:
        simple_consumption_simulation(scenario, initialize_vehicles=False)

    ##### Step 4.5: Insert dummy standby departure events
    # This is something we need to do due to an unclear reason. The depot simulation sometimes does not
    # generate the correct departure events for the vehicles. Therefore, we insert dummy standby departure events
    for depot in session.query(Depot).filter(Depot.scenario_id == scenario.id).all():
        insert_dummy_standby_departure_events(
            depot.id, session, sim_time_end=END_OF_SIMULATION
        )

    ##### Step 5: Apply even smart charging
    # This step is optional. It can be used to apply even smart charging to the vehicles, reducing the peak power
    # consumption. This is done by shifting the charging times of the vehicles. The method is called
    # apply_even_smart_charging and is part of the eflips.depot.api module.
    apply_even_smart_charging(scenario)
    session.commit()

    print(f"Simulation complete for scenario {scenario.name_short}.")
