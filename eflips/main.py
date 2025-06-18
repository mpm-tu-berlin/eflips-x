import logging
import os
import uuid
import warnings
from datetime import timedelta
from os import wait3
from typing import Union, List, Dict, Optional

from prefect.concurrency.sync import concurrency
from prefect.context import FlowRunContext, TaskRunContext

import sqlalchemy
from eflips.depot.api import (
    generate_depot_layout,
    simple_consumption_simulation,
    init_simulation,
    run_simulation,
    add_evaluation_to_database,
    apply_even_smart_charging,
)
from eflips.model import (
    VehicleType,
    Scenario,
    Block,
    ConsistencyWarning,
    Event,
    Vehicle,
    Trip,
    Area,
    Depot,
)
from prefect import flow, task
from prefect.futures import wait
from prefect.logging import get_run_logger
from prefect_dask import DaskTaskRunner
from prefect_sqlalchemy import SqlAlchemyConnector, ConnectionComponents, SyncDriver
from configparser import ConfigParser

from sqlalchemy.orm import Session

from eflips.db_util import (
    clear_database_schema,
    update_database_schema,
    create_db_snapshot,
    restore_db_snapshot,
    SimulationStage,
)


@task(
    cache_key_fn=lambda context, parameters: f"{parameters['database_and_schema_id']}_"
    f"scenario_{parameters['scenario_id']}_"
    f"constant_value_{parameters['constant_value']}",
    version="1",
    name="Set Consumption for all vehicles to a constant value",
)
def set_consumption_constant(
    database_and_schema_id: str,
    scenario_id: int,
    snapshot_folder: str,
    constant_value: float = 1.0,
) -> str:
    """
    Set the energy consumption for all vehicles to a constant value.
    :param database_and_schema_id: The name of the database connection block to use.
    :param constant_value: The constant value to set the energy consumption to.
    :param scenario_id: The ID of the scenario to set the energy consumption for. If None, it will set for all scenarios.
    """

    logger = get_run_logger()
    logger.info(f"Setting energy consumption for all vehicles to {constant_value} kWh/km")

    flow_run_ctx = FlowRunContext.get()
    task_run_ctx = TaskRunContext.get()

    if scenario_id is None:
        raise ValueError("scenario_id must be provided to set energy consumption.")

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            # Update the energy consumption for all vehicles in the specified scenario
            session.query(VehicleType).filter(VehicleType.scenario_id == scenario_id).update(
                {VehicleType.consumption: constant_value}, synchronize_session=False
            )
            # Remove group membership of the VehicleTypes in VehicleClasses that have a ConsumptionLut
            all_vts = session.query(VehicleType).filter(VehicleType.scenario_id == scenario_id)
            for vt in all_vts:
                if len(vt.vehicle_classes) > 0:
                    for vc in vt.vehicle_classes:
                        if vc.consumption_lut is not None:
                            logger.info(
                                f"Removing VehicleType {vt.id} from VehicleClass {vc.id} "
                                "because it has a ConsumptionLut."
                            )
                            vc.vehicle_types.remove(vt)
            session.commit()

    logger.info("Energy consumption set successfully.")

    snapshot_path = create_db_snapshot(
        database_and_schema_id=database_and_schema_id,
        checkpoint_name="consumption_constant_set",
        snapshot_folder=snapshot_folder,
    )
    return snapshot_path


@task(
    cache_key_fn=lambda context, parameters: f"{parameters['database_and_schema_id']}_"
    f"scenario_{parameters['scenario_id']}_"
    f"charging_power_{parameters['charging_power']}",
    name="Generate Depot Layout",
)
def generate_depot_layout_task(
    database_and_schema_id: str,
    scenario_id: int,
    snapshot_folder: str,
    charging_power: float = 300.0,
) -> str:
    """Generate the depot layout for the scenario."""
    logger = get_run_logger()
    logger.info(f"Generating depot layout with {charging_power}kW charging power")

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            scenario = session.query(Scenario).filter(Scenario.id == scenario_id).one()

            generate_depot_layout(
                scenario=scenario, charging_power=charging_power, delete_existing_depot=True
            )
            session.commit()

    logger.info("Depot layout generated")

    snapshot_path = create_db_snapshot(
        database_and_schema_id=database_and_schema_id,
        checkpoint_name=SimulationStage.DEPOT_LAYOUT_GENERATED,
        snapshot_folder=snapshot_folder,
    )
    return snapshot_path


@task(name="Clear preious simulation results")
def clear_previous_simulation_results(database_and_schema_id: str, scenario_id: int) -> None:
    """Clear previous simulation results for the scenario."""
    logger = get_run_logger()
    logger.info("Clearing previous simulation results")

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            all_blocks = session.query(Block).filter(Block.scenario_id == scenario_id)
            all_blocks.update({"vehicle_id": None})  # Clear vehicle assignments in blocks
            session.query(Event).filter(Event.scenario_id == scenario_id).delete()
            session.query(Vehicle).filter(Vehicle.scenario_id == scenario_id).delete()

            session.commit()

    logger.info("Previous simulation results cleared")


@task(
    cache_key_fn=lambda context, parameters: f"{parameters['database_and_schema_id']}_"
    f"scenario_{parameters['scenario_id']}",
    name="Initial Consumption Simulation",
)
def initial_consumption_simulation(
    database_and_schema_id: str, scenario_id: int, snapshot_folder: str
) -> str:
    """Run initial consumption simulation with vehicle initialization."""
    logger = get_run_logger()
    logger.info("Running initial consumption simulation")

    # Suppress ConsistencyWarning for BVG data
    warnings.simplefilter("ignore", category=ConsistencyWarning)

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            scenario = session.query(Scenario).filter(Scenario.id == scenario_id).one()
            simple_consumption_simulation(scenario=scenario, initialize_vehicles=True)
            session.commit()

    logger.info("Initial consumption simulation completed")

    snapshot_path = create_db_snapshot(
        database_and_schema_id=database_and_schema_id,
        checkpoint_name=SimulationStage.ENERGY_CONSUMPTION_INITIAL,
        snapshot_folder=snapshot_folder,
    )
    return snapshot_path


@task(name="Generate Simulation Diagram")
def generate_simulation_diagram(scenario: Scenario, depot_evaluations: Dict) -> None:
    """Generate simulation core diagram for debugging."""
    logger = get_run_logger()
    logger.info("Generating simulation diagrams")

    OUTPUT_DIR = os.path.join("output", scenario.name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        for depot in scenario.depots:
            DEPOT_NAME = depot.station.name
            DEPOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, DEPOT_NAME)
            os.makedirs(DEPOT_OUTPUT_DIR, exist_ok=True)

            depot_evaluation = depot_evaluations[str(depot.id)]
            depot_evaluation.path_results = DEPOT_OUTPUT_DIR

            depot_evaluation.vehicle_periods(
                periods={
                    "depot general": "darkgray",
                    "park": "lightgray",
                    "Arrival Cleaning": "steelblue",
                    "Charging": "forestgreen",
                    "Standby Pre-departure": "darkblue",
                    "precondition": "black",
                    "trip": "wheat",
                },
                save=True,
                show=False,
                formats=("pdf", "png"),
                show_total_power=True,
                show_annotates=True,
            )
    except AssertionError as e:
        logger.warning(
            "There are vehicles waiting for entering the depot. "
            "Please make the capacity of the first area larger to avoid this issue."
        )


@task(
    cache_key_fn=lambda context, parameters: f"{parameters['database_and_schema_id']}_"
    f"scenario_{parameters['scenario_id']}_"
    f"repetition_period_{parameters.get('repetition_period', 'None')}_"
    f"generate_diagram_{uuid.uuid4().hex if parameters.get('generate_diagram') else 'False'}",  # If generate_diagram is True, we want to generate a new UUID the code is run again
    name="Run Depot Simulation",
)
def run_depot_simulation(
    database_and_schema_id: str,
    scenario_id: int,
    snapshot_folder: str,
    repetition_period: Optional[timedelta] = None,
    generate_diagram: bool = False,
) -> str:
    """Run the core depot simulation."""
    logger = get_run_logger()
    logger.info("Starting core simulation")

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            scenario = session.query(Scenario).filter(Scenario.id == scenario_id).one()

            # Initialize simulation
            simulation_host = init_simulation(
                scenario=scenario, session=session, repetition_period=repetition_period
            )

            # Run simulation
            depot_evaluations = run_simulation(simulation_host)

            # Generate diagram if requested
            if generate_diagram:
                generate_simulation_diagram(scenario, depot_evaluations)

            # Add evaluation to database
            add_evaluation_to_database(scenario, depot_evaluations, session)
            session.expire_all()
            session.commit()

    logger.info("Core simulation completed")

    snapshot_path = create_db_snapshot(
        database_and_schema_id=database_and_schema_id,
        checkpoint_name=SimulationStage.SIMULATED,
        snapshot_folder=snapshot_folder,
    )
    return snapshot_path


@task(name="Final Consumption Simulation")
def final_consumption_simulation(database_and_schema_id: str, scenario_id: int) -> None:
    """Run final consumption simulation after vehicle merging."""
    logger = get_run_logger()
    logger.info("Running final consumption simulation")

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            scenario = session.query(Scenario).filter(Scenario.id == scenario_id).one()
            simple_consumption_simulation(scenario=scenario, initialize_vehicles=False)
            session.commit()

    logger.info("Final consumption simulation completed")


@task(name="Apply Smart Charging")
def apply_smart_charging(database_and_schema_id: str, scenario_id: int) -> None:
    """Apply even smart charging to reduce peak power consumption."""
    logger = get_run_logger()
    logger.info("Applying smart charging optimization")

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            scenario = session.query(Scenario).filter(Scenario.id == scenario_id).one()
            apply_even_smart_charging(scenario)
            session.commit()

    logger.info("Smart charging applied")


@task(name="Generate Visualizations")
def generate_visualizations(database_and_schema_id: str, scenario_id: int) -> None:
    """Generate visualization outputs if eflips.eval is available."""
    logger = get_run_logger()

    try:
        import eflips.eval.input.prepare
        import eflips.eval.input.visualize
        import eflips.eval.output.prepare
        import eflips.eval.output.visualize
    except ImportError:
        logger.warning(
            "The eflips.eval package is not installed. Visualization is not possible. "
            "Install it using: pip install eflips-eval"
        )
        return

    logger.info("Generating visualizations")
    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            scenario = session.query(Scenario).filter(Scenario.id == scenario_id).one()
            OUTPUT_DIR = os.path.join("output", scenario.name)
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            futures = []

            for depot in scenario.depots:
                DEPOT_NAME = depot.station.name
                DEPOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, DEPOT_NAME)
                os.makedirs(DEPOT_OUTPUT_DIR, exist_ok=True)

                # Generate rotation info visualization
                futures.append(
                    generate_rotation_info_viz.submit(
                        database_and_schema_id=database_and_schema_id,
                        scenario_id=scenario_id,
                        depot_id=depot.id,
                        output_dir=DEPOT_OUTPUT_DIR,
                    )
                )

                # Generate power and occupancy visualization
                futures.append(
                    generate_power_occupancy_viz.submit(
                        database_and_schema_id=database_and_schema_id,
                        scenario_id=scenario_id,
                        depot_id=depot.id,
                        output_dir=DEPOT_OUTPUT_DIR,
                    )
                )

                # Generate depot event timeline
                futures.append(
                    generate_depot_event_viz.submit(
                        database_and_schema_id=database_and_schema_id,
                        scenario_id=scenario_id,
                        depot_id=depot.id,
                        output_dir=DEPOT_OUTPUT_DIR,
                    )
                )

                # Generate vehicle SoC visualizations
                vehicles = (
                    session.query(Vehicle)
                    .join(Event)
                    .join(Area)
                    .filter(Area.depot_id == depot.id)
                    .all()
                )
                for vehicle in vehicles:
                    futures.append(
                        generate_vehicle_soc_viz.submit(
                            database_and_schema_id=database_and_schema_id,
                            vehicle_id=vehicle.id,
                            output_dir=DEPOT_OUTPUT_DIR,
                        )
                    )
            # Wait for all tasks to complete
            wait(futures)

    logger.info("Visualizations generated")


@task(
    name="Generate Rotation Information Visualization",
    cache_key_fn=lambda context, parameters: f"{parameters['database_and_schema_id']}_"
    f"scenario_{parameters['scenario_id']}_"
    f"depot_{parameters['depot_id']}",
)
def generate_rotation_info_viz(
    database_and_schema_id: str, scenario_id: int, depot_id: int, output_dir: str
):
    """Generate rotation information visualization."""
    import eflips.eval.input.prepare
    import eflips.eval.input.visualize

    logger = get_run_logger()

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            # Find rotations using this depot
            rotations = (
                session.query(Block)
                .filter(Block.scenario_id == scenario_id)
                .options(sqlalchemy.orm.joinedload(Block.trips).joinedload(Trip.route))
            )
            depot = session.query(Depot).filter(Depot.id == depot_id).one()

            rotation_ids = set()
            for rotation in rotations:
                if rotation.trips[0].route.departure_station_id == depot.station.id:
                    rotation_ids.add(rotation.id)
            rotation_ids = list(rotation_ids)

            rotation_info = eflips.eval.input.prepare.rotation_info(
                scenario_id=scenario_id, session=session, rotation_ids=rotation_ids
            )
            fig = eflips.eval.input.visualize.rotation_info(rotation_info)
            fig.update_layout(title=f"Rotation information for {depot.station.name}")
            fig.write_html(os.path.join(output_dir, "rotation_info.html"))


@task(name="Generate Power and Occupancy Visualization")
def generate_power_occupancy_viz(
    database_and_schema_id: str, scenario_id: int, depot_id: int, output_dir: str
):
    """Generate power and occupancy visualization."""
    import eflips.eval.output.prepare
    import eflips.eval.output.visualize

    logger = get_run_logger()

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            depot = session.query(Depot).filter(Depot.id == depot_id).one()

            areas = session.query(Area).filter(Area.depot_id == depot_id).all()
            area_ids = [area.id for area in areas]
            df = eflips.eval.output.prepare.power_and_occupancy(area_ids, session)
            fig = eflips.eval.output.visualize.power_and_occupancy(df)
            fig.update_layout(title=f"Power and occupancy for {depot.station.name}")
            fig.write_html(os.path.join(output_dir, "power_and_occupancy.html"))


@task(name="Generate Depot Event Timeline Visualization")
def generate_depot_event_viz(
    database_and_schema_id: str, scenario_id: int, depot_id: int, output_dir: str
):
    """Generate depot event timeline visualization."""
    import eflips.eval.output.prepare
    import eflips.eval.output.visualize

    logger = get_run_logger()

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            depot = session.query(Depot).filter(Depot.id == depot_id).one()
            vehicles = (
                session.query(Vehicle)
                .join(Event)
                .join(Area)
                .filter(Area.depot_id == depot_id)
                .all()
            )
            vehicle_ids = [vehicle.id for vehicle in vehicles]
            df = eflips.eval.output.prepare.depot_event(scenario_id, session, vehicle_ids)

            for color_scheme in ["event_type", "soc", "location"]:
                fig = eflips.eval.output.visualize.depot_event(df, color_scheme=color_scheme)
                fig.update_layout(
                    title=f"Depot events for {depot.station.name}, color scheme: {color_scheme}"
                )
                fig.write_html(os.path.join(output_dir, f"depot_event_{color_scheme}.html"))


@task(name="Generate Vehicle State of Charge Visualization")
def generate_vehicle_soc_viz(database_and_schema_id: str, vehicle_id: int, output_dir: str):
    """Generate vehicle State of Charge visualizations."""
    import eflips.eval.output.prepare
    import eflips.eval.output.visualize

    logger = get_run_logger()

    VEHICLE_OUTPUT_DIR = os.path.join(output_dir, "vehicles")
    os.makedirs(VEHICLE_OUTPUT_DIR, exist_ok=True)

    with concurrency("database", occupy=1, strict=True):
        with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
            engine = db_connector.get_engine()
            # Open a session to the database
            with Session(engine) as session:
                df, descriptions = eflips.eval.output.prepare.vehicle_soc(vehicle_id, session)
                fig = eflips.eval.output.visualize.vehicle_soc(df, descriptions)
                fig.update_layout(title=f"Vehicle {vehicle_id} SoC over time")
                fig.write_html(os.path.join(VEHICLE_OUTPUT_DIR, f"vehicle_{vehicle_id}_soc.html"))


@flow(persist_result=True, name="Core Simulation", version="1")
def core_simulation_flow(
    database_and_schema_id: str, scenario_id: int, snapshot_folder: str
) -> None:
    """
    The "core simulation" turns a "simulation-ready" scenario into a "simulated" scenario. It runs energy consumption
    and depot simulation, and produces a "simulated" scenario with the results of the simulation.
    """
    logger = get_run_logger()
    logger.info("Starting core simulation flow...")

    # Set the energy consumption for all vehicles in the scenario to a constant value
    snapshot_path = set_consumption_constant(
        database_and_schema_id,
        constant_value=1.0,
        scenario_id=scenario_id,
        snapshot_folder=snapshot_folder,
    )
    restore_db_snapshot(database_and_schema_id=database_and_schema_id, snapshot_path=snapshot_path)

    # Generate the depot layout for the scenario
    snapshot_path = generate_depot_layout_task(
        database_and_schema_id, scenario_id, charging_power=300.0, snapshot_folder=snapshot_folder
    )
    restore_db_snapshot(database_and_schema_id=database_and_schema_id, snapshot_path=snapshot_path)

    # Run the initial consumption simulation
    snapshot_path = initial_consumption_simulation(
        database_and_schema_id=database_and_schema_id,
        scenario_id=scenario_id,
        snapshot_folder=snapshot_folder,
    )
    restore_db_snapshot(database_and_schema_id=database_and_schema_id, snapshot_path=snapshot_path)

    # Run the core depot simulation
    snapshot_path = run_depot_simulation(
        database_and_schema_id=database_and_schema_id,
        scenario_id=scenario_id,
        snapshot_folder=snapshot_folder,
        repetition_period=timedelta(days=1),  # Example repetition period
        generate_diagram=False,  # Set to True if you want to generate a diagram
    )
    restore_db_snapshot(database_and_schema_id=database_and_schema_id, snapshot_path=snapshot_path)

    # Run the final consumption simulation
    final_consumption_simulation(database_and_schema_id, scenario_id)


@task(name="List Available Scenarios")
def list_scenarios_task(database_and_schema_id: str) -> List[Dict[str, Union[int, str]]]:
    """List all available scenarios in the database."""
    logger = get_run_logger()
    scenarios_info = []

    with SqlAlchemyConnector.load(database_and_schema_id) as db_connector:
        engine = db_connector.get_engine()
        # Open a session to the database
        with Session(engine) as session:
            scenarios = session.query(Scenario).all()
            for scenario in scenarios:
                rotation_count = (
                    session.query(Block).filter(Block.scenario_id == scenario.id).count()
                )
                scenarios_info.append(
                    {"id": scenario.id, "name": scenario.name, "rotation_count": rotation_count}
                )
                logger.info(
                    f"Scenario {scenario.id}: {scenario.name} with {rotation_count} rotations."
                )

    return scenarios_info


@flow(
    name="Sample Run", version="1", task_runner=DaskTaskRunner(cluster_kwargs={"processes": True})
)
def sample_run_flow(database_and_schema_id: str, snapshot_folder: str) -> None:
    """
    Runs a full simulation of the included `sample_db.pg_dump` database.
    :param database_and_schema_id: The name of the database connection block to use. "default" is a common choice.
    :return: None
    """
    logger = get_run_logger()
    logger.info("Starting sample run flow...")

    # Clear the database
    clear_database_schema(database_and_schema_id)

    # Restore the sample database snapshot
    snapshot_path = "sample_db.pg_dump"
    restore_db_snapshot(database_and_schema_id, snapshot_path=snapshot_path, checkpoint_name=None)

    # Update the database schema
    update_database_schema(database_and_schema_id)
    logger.info("Sample run flow completed successfully.")

    scenarios = list_scenarios_task(database_and_schema_id)

    # Now run the core simulation for each scenario (in parallel if needed)
    for scenario in scenarios:
        scenario_id = scenario["id"]
        logger.info(f"Running core simulation for scenario ID: {scenario_id}")
        clear_previous_simulation_results(
            database_and_schema_id=database_and_schema_id,
            scenario_id=scenario_id,
        )

        core_simulation_flow(
            database_and_schema_id=database_and_schema_id,
            scenario_id=scenario_id,
            snapshot_folder=snapshot_folder,
        )
        # apply_smart_charging(
        #    database_and_schema_id=database_and_schema_id, scenario_id=scenario_id, snapshot_folder=snapshot_folder
        # )

        generate_visualizations(
            database_and_schema_id=database_and_schema_id,
            scenario_id=scenario_id,
        )


def save_database_connector_from_config_for_schema(schema_name: str) -> SqlAlchemyConnector:
    """
    Create a database connector from the configuration.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating database connector from config...")

    config = ConfigParser()
    config.read("config.ini")
    db_name = config.get("database", "name")
    db_host = config.get("database", "host")
    db_user = config.get("database", "user")
    db_password = config.get("database", "password")
    db_port = config.get("database", "port")

    logger.info(f"Connecting to database: {db_name} at {db_host}:{db_port} as {db_user}")
    db_connector = SqlAlchemyConnector(
        connection_info=ConnectionComponents(
            driver=SyncDriver.POSTGRESQL_PSYCOPG2,
            database=db_name,
            username=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        ),
        connect_args={
            "options": f"-c search_path=common,{schema_name}"
        },  # Set the schema search path
    )

    db_connector.save(SCHEMA_NAME, overwrite=True)


if __name__ == "__main__":
    # Set up the database connector for our simulation run
    SCHEMA_NAME = "example2"  # Replace with your schema name
    save_database_connector_from_config_for_schema(SCHEMA_NAME)

    config = ConfigParser()
    config.read("config.ini")
    snapshot_folder = config.get("paths", "snapshot_folder")

    # Run the sample run flow
    sample_run_flow(database_and_schema_id=SCHEMA_NAME, snapshot_folder=snapshot_folder)
