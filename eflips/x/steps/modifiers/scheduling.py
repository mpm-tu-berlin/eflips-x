"""
Vehicle scheduling modifiers for rotation optimization.

This module contains modifiers that perform vehicle scheduling operations,
such as creating optimal rotation plans for different vehicle types.
"""

import logging
import os
from collections import defaultdict
from datetime import timedelta
from typing import Any, Dict, List

import eflips.model
from eflips.depot.api import generate_consumption_result, ConsumptionResult
from eflips.model import VehicleType, Trip, TripType, Rotation, Scenario, Route, Station
from eflips.opt.depot_rotation_matching import DepotRotationOptimizer
from eflips.opt.scheduling import create_graph, solve, write_back_rotation_plan
from sqlalchemy.orm import Session, joinedload

from eflips.x.framework import Modifier


class VehicleScheduling(Modifier):
    """
    Generate vehicle schedules by optimizing rotation plans.

    This modifier creates optimal rotation plans for each vehicle type in the scenario
    by solving a vehicle scheduling optimization problem. It takes into account:
    - Energy consumption (SOC) for each trip
    - Maximum schedule duration constraints
    - Minimum break time requirements between trips
    - Battery margin for safety

    The optimization is performed separately for each vehicle type that has rotations
    in the scenario.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_minimum_break_time() -> timedelta:
        """Get the default minimum break time."""
        return timedelta(seconds=0)

    @staticmethod
    def _get_default_maximum_schedule_duration() -> timedelta:
        """Get the default maximum schedule duration."""
        return timedelta(hours=24)

    @staticmethod
    def _get_default_battery_margin() -> float:
        """Get the default battery margin."""
        return 0.1

    @staticmethod
    def _get_default_longer_break_time_trips() -> List[int]:
        """Get the default list of trips requiring longer break time."""
        return []

    @staticmethod
    def _get_default_longer_break_time_duration() -> timedelta:
        """Get the default additional break time duration."""
        return timedelta(minutes=5)

    def document_params(self) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary describing the configurable parameters:
            - VehicleScheduling.minimum_break_time: Minimum break time as timedelta
            - VehicleScheduling.maximum_schedule_duration: Maximum schedule duration as timedelta
            - VehicleScheduling.battery_margin: Battery safety margin as fraction (0.0-1.0)
            - VehicleScheduling.longer_break_time_trips: List of trip IDs requiring longer breaks
            - VehicleScheduling.longer_break_time_duration: Additional break time for specified trips
        """
        return {
            f"{self.__class__.__name__}.minimum_break_time": """
            Minimum break time required between trips.
            This ensures drivers have adequate rest between consecutive trips.

            Default: timedelta(seconds=0)
            Type: timedelta
            Example: timedelta(minutes=15)
            """,
            f"{self.__class__.__name__}.maximum_schedule_duration": """
            Maximum duration of a vehicle schedule.
            This limits how long a single vehicle can operate before returning to depot.

            Default: timedelta(hours=24)
            Type: timedelta
            Example: timedelta(hours=12, minutes=30)
            """,
            f"{self.__class__.__name__}.battery_margin": """
            Battery safety margin as a fraction (0.0 to 1.0).
            This reduces the effective battery capacity to ensure vehicles don't fully deplete.
            For example, 0.1 means 10% of battery capacity is reserved as safety margin.

            Default: 0.1 (10%)
            Type: float
            Example: 0.15 (15% margin)
            """,
            f"{self.__class__.__name__}.longer_break_time_trips": """
            List of trip IDs that require a longer break time after completion.
            This can be used to specify trips that require additional rest or preparation time.

            Default: [] (empty list)
            Type: List[int]
            Example: [123, 456, 789]
            """,
            f"{self.__class__.__name__}.longer_break_time_duration": """
            Additional break time duration for trips specified in longer_break_time_trips.
            This duration is added on top of the minimum_break_time.

            Default: timedelta(minutes=5)
            Type: timedelta
            Example: timedelta(minutes=10)
            """,
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Generate optimal vehicle schedules for all vehicle types.

        This method:
        1. Validates that exactly one scenario exists
        2. Generates energy consumption data for all trips
        3. For each vehicle type with rotations:
           - Creates a graph of possible trip connections
           - Solves the vehicle scheduling optimization
           - Writes the resulting rotation plan back to the database

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters:
            - VehicleScheduling.minimum_break_time (optional): Minimum break time as timedelta
            - VehicleScheduling.maximum_schedule_duration (optional): Maximum schedule duration as timedelta
            - VehicleScheduling.battery_margin (optional): Battery safety margin (0.0-1.0)

        Returns:
        --------
        None
            This modifier modifies the database in place by updating rotation plans
        """
        # Get parameters
        min_break_key = f"{self.__class__.__name__}.minimum_break_time"
        max_duration_key = f"{self.__class__.__name__}.maximum_schedule_duration"
        battery_margin_key = f"{self.__class__.__name__}.battery_margin"
        longer_break_trips_key = f"{self.__class__.__name__}.longer_break_time_trips"
        longer_break_duration_key = f"{self.__class__.__name__}.longer_break_time_duration"

        minimum_break_time = params.get(min_break_key, self._get_default_minimum_break_time())
        maximum_schedule_duration = params.get(
            max_duration_key, self._get_default_maximum_schedule_duration()
        )
        battery_margin = params.get(battery_margin_key, self._get_default_battery_margin())
        longer_break_time_trips = params.get(
            longer_break_trips_key, self._get_default_longer_break_time_trips()
        )
        longer_break_time_duration = params.get(
            longer_break_duration_key, self._get_default_longer_break_time_duration()
        )

        # Validate parameters
        if not isinstance(minimum_break_time, timedelta):
            raise ValueError(
                f"minimum_break_time must be a timedelta, got {type(minimum_break_time).__name__}"
            )
        if not isinstance(maximum_schedule_duration, timedelta):
            raise ValueError(
                f"maximum_schedule_duration must be a timedelta, got {type(maximum_schedule_duration).__name__}"
            )
        if not isinstance(battery_margin, (int, float)):
            raise ValueError(
                f"Battery margin must be a number, got {type(battery_margin).__name__}"
            )
        if not (0.0 <= battery_margin < 1.0):
            raise ValueError(f"Battery margin must be between 0.0 and 1.0, got {battery_margin}")
        if not isinstance(longer_break_time_trips, list):
            raise ValueError(
                f"longer_break_time_trips must be a list, got {type(longer_break_time_trips).__name__}"
            )
        if not all(isinstance(trip_id, int) for trip_id in longer_break_time_trips):
            raise ValueError("All elements in longer_break_time_trips must be integers (trip IDs)")
        if not isinstance(longer_break_time_duration, timedelta):
            raise ValueError(
                f"longer_break_time_duration must be a timedelta, got {type(longer_break_time_duration).__name__}"
            )

        # Make sure there is just one scenario
        scenario_q = session.query(eflips.model.Scenario)
        if scenario_q.count() != 1:
            raise ValueError(f"Expected exactly one scenario, found {scenario_q.count()}")
        scenario = scenario_q.one()

        # Log the parameters being used
        self.logger.info(
            f"Vehicle scheduling parameters: "
            f"minimum_break_time={minimum_break_time}, "
            f"maximum_schedule_duration={maximum_schedule_duration}, "
            f"battery_margin={battery_margin}, "
            f"longer_break_time_trips={longer_break_time_trips}, "
            f"longer_break_time_duration={longer_break_time_duration}"
        )

        # Create a dictionary of the energy consumption for each rotation
        consumption: Dict[int, ConsumptionResult] = generate_consumption_result(scenario)

        # Convert to the format eflips-opt expects: {trip_id: delta_soc, ...}
        # Also turn the delta_soc into a positive number
        delta_socs: Dict[int, float] = {}
        for trip_id, result in consumption.items():
            delta_socs[trip_id] = -1 * result.delta_soc_total / (1 - battery_margin)

        self.logger.info("Generated consumption results")

        # Get all vehicle types that have rotations
        all_vehicle_types = session.query(VehicleType).join(Rotation).distinct().all()
        num_vehicle_types = len(all_vehicle_types)

        if num_vehicle_types == 0:
            raise ValueError("No vehicle types found in the database")

        self.logger.info(f"Processing {num_vehicle_types} vehicle type(s)")

        # Process each vehicle type separately
        for i, vehicle_type in enumerate(all_vehicle_types):
            self.logger.info(
                f"Processing vehicle type {i+1}/{num_vehicle_types}: {vehicle_type.name_short}"
            )

            # Get all passenger trips for this vehicle type
            trips = (
                session.query(Trip)
                .join(Rotation)
                .filter(Rotation.vehicle_type_id == vehicle_type.id)
                .filter(Trip.trip_type == TripType.PASSENGER)
                .all()
            )

            # Create the graph of all possible connections between trips
            graph = create_graph(
                trips=trips,
                delta_socs=delta_socs,
                maximum_schedule_duration=maximum_schedule_duration,
                minimum_break_time=minimum_break_time,
                longer_break_time_trips=longer_break_time_trips,
                longer_break_time_duration=longer_break_time_duration,
            )
            self.logger.info(
                f"Created graph for vehicle type {vehicle_type.name_short} "
                f"with {len(graph.nodes)} nodes and {len(graph.edges)} edges"
            )

            # Solve the vehicle scheduling problem
            rotation_plan = solve(graph)
            self.logger.info(
                f"Solved vehicle scheduling for vehicle type {vehicle_type.name_short}"
            )

            # Write the rotation plan back to the database
            write_back_rotation_plan(rotation_plan, session)
            self.logger.info(
                f"Wrote back rotation plan for vehicle type {vehicle_type.name_short}"
            )

        session.flush()

        return None


class DepotAssignment(Modifier):
    """
    Optimize depot assignment for rotations in a scenario.

    This modifier uses the eflips-opt depot rotation matching optimizer to assign rotations
    to depots optimally. It considers depot capacities, vehicle type constraints, and travel
    distances when making assignments.

    The optimization process:
    1. Loads depot configuration (capacities and vehicle type constraints)
    2. Iteratively reduces depot capacity usage to find minimal feasible assignment
    3. Writes optimized depot assignments back to the database
    4. Logs comparison of before/after assignments
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_depot_usage() -> float:
        """Get the default depot usage factor."""
        return 1.0

    @staticmethod
    def _get_default_step_size() -> float:
        """Get the default step size for capacity reduction."""
        return 0.1

    @staticmethod
    def _get_default_max_iterations() -> int:
        """Get the default maximum number of optimization iterations."""
        return 5

    def document_params(self) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary describing the configurable parameters:
            - DepotAssignment.depot_config: List of depot configuration dicts
            - DepotAssignment.base_url: Base URL for routing service (ORS)
            - DepotAssignment.depot_usage: Initial depot capacity usage factor (0.0-1.0)
            - DepotAssignment.step_size: Capacity reduction step size per iteration
            - DepotAssignment.max_iterations: Maximum optimization iterations
        """
        return {
            f"{self.__class__.__name__}.depot_config": """
            A list of depot configurations. Each configuration is a dict with:
            - "depot_station": Station ID or (lon, lat) tuple
            - "capacity": Depot capacity in 12m bus equivalents
            - "vehicle_type": List of allowed vehicle type IDs
            - "name": Depot name (for new depots only)

            Required: True
            Type: List[Dict]
            Example: depots_for_bvg(db_session) from eflips.x.steps.modifiers.bvg_tools
            """,
            f"{self.__class__.__name__}.base_url": """
            Base URL for the OpenRouteService (ORS) routing API.
            Used to calculate travel distances between depots and rotation start/end points.

            Required: True
            Type: str
            Example: "http://mpm-v-ors.mpm.tu-berlin.de:8080/ors/"
            """,
            f"{self.__class__.__name__}.depot_usage": """
            Initial depot capacity usage factor (0.0 to 1.0).
            The optimizer starts with this fraction of nominal depot capacity and
            iteratively reduces it to find the minimal feasible assignment.
            For example, 1.0 means 100% of depot capacity is available initially.

            Default: 1.0 (100% capacity)
            Type: float
            Example: 0.9 (90% capacity)
            """,
            f"{self.__class__.__name__}.step_size": """
            Capacity reduction step size per iteration (0.0 to 1.0).
            After each successful optimization, depot capacities are reduced by this factor.
            Smaller step sizes find tighter capacity bounds but take more iterations.

            Default: 0.1 (reduce by 10% each iteration)
            Type: float
            Example: 0.05 (reduce by 5% each iteration)
            """,
            f"{self.__class__.__name__}.max_iterations": """
            Maximum number of optimization iterations to attempt.
            The optimizer stops after this many iterations or when a solution becomes infeasible.

            Default: 5
            Type: int
            Example: 10
            """,
        }

    def _get_depot_rotation_assignments(self, session: Session) -> Dict[int, List[Rotation]]:
        """
        Retrieve current depot assignments for all rotations.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database

        Returns:
        --------
        Dict[int, List[Rotation]]
            Dictionary mapping depot station IDs to lists of assigned Rotations
        """
        assignments = defaultdict(list)

        assert session.query(Scenario).count() == 1, "Expected exactly one scenario"

        rotations = (
            session.query(Rotation)
            .options(
                joinedload(Rotation.trips).joinedload(Trip.route).joinedload(Route.arrival_station)
            )
            .options(
                joinedload(Rotation.trips)
                .joinedload(Trip.route)
                .joinedload(Route.departure_station)
            )
            .all()
        )
        for rotation in rotations:
            if (
                rotation.trips[0].route.departure_station_id
                != rotation.trips[-1].route.arrival_station_id
            ):
                self.logger.warning(
                    f"Rotation {rotation.id} does not start and end at the same station. "
                    "Using the start station as 'the depot' for this rotation."
                )
            assignments[rotation.trips[0].route.departure_station_id].append(rotation)
        return assignments

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Optimize depot assignments for rotations in the scenario.

        This method:
        1. Validates that exactly one scenario exists
        2. Loads depot configuration via provided function
        3. Initializes the DepotRotationOptimizer
        4. Iteratively optimizes depot assignments with decreasing capacity
        5. Writes optimized assignments back to database
        6. Logs comparison of before/after assignments

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters:
            - DepotAssignment.depot_config (required): List of depot configuration dicts
            - DepotAssignment.base_url (required): ORS routing service URL
            - DepotAssignment.depot_usage (optional): Initial depot capacity usage
            - DepotAssignment.step_size (optional): Capacity reduction step per iteration
            - DepotAssignment.max_iterations (optional): Max optimization iterations

        Returns:
        --------
        None
            This modifier modifies the database in place by updating rotation depot assignments

        Raises:
        -------
        ValueError
            If required parameters are not provided or if scenario count != 1
        """

        # Get parameters
        depot_config_key = f"{self.__class__.__name__}.depot_config"
        base_url_key = f"{self.__class__.__name__}.base_url"
        depot_usage_key = f"{self.__class__.__name__}.depot_usage"
        step_size_key = f"{self.__class__.__name__}.step_size"
        max_iter_key = f"{self.__class__.__name__}.max_iterations"

        # Validate required parameters
        if depot_config_key not in params:
            raise ValueError(
                f"Required parameter '{depot_config_key}' not provided. "
                "Please specify a depot configuration."
            )
        if base_url_key in params:
            base_url = params[base_url_key]
        else:
            # Check if instead theenvironment variable OPENROUTESERVICE_BASE_URL
            base_url = os.environ.get("OPENROUTESERVICE_BASE_URL")
            if not base_url:
                raise ValueError(
                    f"Required parameter '{base_url_key}' not provided. "
                    "Also, environment variable 'OPENROUTESERVICE_BASE_URL' is not set. "
                    "Please specify the base URL for the routing service."
                )
            self.logger.debug(
                "Taking base_url from environment variable OPENROUTESERVICE_BASE_URL"
            )

        depot_config = params[depot_config_key]
        depot_usage = params.get(depot_usage_key, self._get_default_depot_usage())
        step_size = params.get(step_size_key, self._get_default_step_size())
        max_iterations = params.get(max_iter_key, self._get_default_max_iterations())

        # Validate parameters
        if not isinstance(depot_config, list):
            raise ValueError(
                f"depot_config must be a list of depot configurations, got {type(depot_config).__name__}"
            )
            for depot in depot_config:
                if not isinstance(depot, dict):
                    raise ValueError(
                        f"Each depot configuration must be a dict, got {type(depot).__name__}"
                    )
        if not isinstance(base_url, str):
            raise ValueError(f"base_url must be a string, got {type(base_url).__name__}")
        if not isinstance(depot_usage, (int, float)):
            raise ValueError(f"depot_usage must be a number, got {type(depot_usage).__name__}")
        if not (0.0 < depot_usage <= 1.0):
            raise ValueError(f"depot_usage must be between 0.0 and 1.0, got {depot_usage}")
        if not isinstance(step_size, (int, float)):
            raise ValueError(f"step_size must be a number, got {type(step_size).__name__}")
        if not (0.0 < step_size < 1.0):
            raise ValueError(f"step_size must be between 0.0 and 1.0, got {step_size}")
        if not isinstance(max_iterations, int):
            raise ValueError(
                f"max_iterations must be an integer, got {type(max_iterations).__name__}"
            )
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        # Make sure there is just one scenario
        scenario_q = session.query(eflips.model.Scenario)
        if scenario_q.count() != 1:
            raise ValueError(f"Expected exactly one scenario, found {scenario_q.count()}")
        scenario = scenario_q.one()

        self.logger.info(
            f"Depot assignment parameters: "
            f"depot_usage={depot_usage}, "
            f"step_size={step_size}, "
            f"max_iterations={max_iterations}, "
            f"base_url={base_url}"
        )

        # Get the pre-optimization depot assignments for logging
        pre_optimization_assignments = self._get_depot_rotation_assignments(session)

        self._log_assignments(pre_optimization_assignments, session)
        self.logger.info("Completed logging pre-optimization depot assignments")

        # Get depot configuration from the provided function
        self.logger.info(f"Loaded {len(depot_config)} depot configurations")

        # Set the base URL for routing service
        os.environ["BASE_URL"] = base_url

        # Initialize the Optimizer
        optimizer = DepotRotationOptimizer(session, scenario.id)
        original_capacities = [depot["capacity"] for depot in depot_config]

        # Using the optimizer iteratively to reach a lower depot capacity until the solution is not feasible
        DEPOT_USAGE = depot_usage
        STEP_SIZE = step_size
        ITER = max_iterations

        self.logger.info(f"Starting iterative optimization with {ITER} max iterations")

        while ITER > 0:
            self.logger.info(
                f"Optimization iteration {max_iterations - ITER + 1}/{max_iterations}, "
                f"depot usage: {DEPOT_USAGE:.1%}"
            )

            for depot, orig_cap in zip(depot_config, original_capacities):
                depot["capacity"] = int(orig_cap * DEPOT_USAGE)

            optimizer.get_depot_from_input(depot_config)
            optimizer.data_preparation()

            try:
                optimizer.optimize(time_report=True)
                self.logger.info(f"Optimization successful at {DEPOT_USAGE:.1%} capacity")
            except ValueError as e:
                self.logger.info(
                    f"Cannot decrease depot capacity any further at {DEPOT_USAGE:.1%}. "
                    f"Stopping optimization."
                )
                break

            DEPOT_USAGE -= STEP_SIZE
            ITER -= 1

        if ITER == 0:
            self.logger.info(
                f"Reached maximum iterations ({max_iterations}). "
                f"Final depot usage: {DEPOT_USAGE + STEP_SIZE:.1%}"
            )

        # Write optimization results back to the database
        optimizer.write_optimization_results(delete_original_data=True)

        assert optimizer.data["result"] is not None
        assert optimizer.data["result"].shape[0] == optimizer.data["rotation"].shape[0]

        self.logger.info("Wrote optimization results to database")

        # Generate post-optimization depot assignments for logging
        # We need to flush and expunge the session for geom to be converted to binary
        session.flush()
        session.expunge_all()
        post_optimization_assignments = self._get_depot_rotation_assignments(session)

        # Log the assignments before and after optimization
        self._log_assignments(post_optimization_assignments, session)
        self.logger.info("Completed logging post-optimization depot assignments")

        # Go through all depots (union of pre and post optimization keys) in alphabetical order and list the changes
        all_stations = set(pre_optimization_assignments.keys()).union(
            post_optimization_assignments.keys()
        )
        for station_id in sorted(all_stations):
            station = session.query(Station).filter(Station.id == station_id).one()
            pre_rotations = pre_optimization_assignments.get(station.id, [])
            post_rotations = post_optimization_assignments.get(station.id, [])

            pre_count = len(pre_rotations)
            post_count = len(post_rotations)

            self.logger.info(f"Depot '{station.name}' (ID {station_id}): ")
            if pre_count == post_count and set(r.id for r in pre_rotations) == set(
                r.id for r in post_rotations
            ):
                self.logger.info(f"\tNo change in assignments ({pre_count} rotations)")
            else:
                self.logger.info(f"\tChanged from {pre_count} to {post_count} rotations")

        self.logger.info("Depot assignment optimization completed successfully")

        return None

    def _log_assignments(
        self,
        pre_optimization_assignments: dict[int, list[Rotation]],
        session: sqlalchemy.orm.session.Session,
    ):
        self.logger.info(
            f"Total of {len(pre_optimization_assignments)} depots with "
            f"{sum([len(v) for v in pre_optimization_assignments.values()])}"
            f" rotations"
        )

        # Iterate over the stations (eys of the dict) in ID order
        for station_id in sorted(pre_optimization_assignments.keys()):
            rotations = pre_optimization_assignments[station_id]
            station = session.query(Station).filter(Station.id == station_id).one()
            self.logger.info(
                f"\tDepot '{station.name}' (ID {station.id}) has {len(rotations)} rotations assigned before optimization"
            )


class InsufficientChargingTimeAnalyzer(Analyzer):
    """
    Analyze whether rotations have sufficient charging time throughout the day.

    This analyzer checks if the schedule has enough charging time so that buses do not
    lose more energy driving than they ever regain from charging throughout the day.
    If rotations end with negative SOC, it indicates that the schedule needs more slack
    time for charging, and a new schedule with more break time should be created.

    Returns None if all rotations have sufficient charging time (SOC >= 0 at end of day),
    or a list of rotation IDs that end with SOC below zero.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    def document_params(self) -> Dict[str, str]:
        """
        Document the parameters of this analyzer.

        Returns:
        --------
        Dict[str, str]
            This analyzer does not use any configurable parameters.
        """
        return {
            self.__class__.__name__
            + ".charging_power_kw": """
            The charging power in kW to assume for all charging stations during the analysis. Default is 150 kW.
            """.strip()
        }

    def _get_default_charging_power_kw(self) -> float:
        """Get the default charging power in kW."""
        return 150.0

    def analyze(self, session: Session, params: Dict[str, Any]) -> List[int] | None:
        """
        Check if rotations have sufficient charging time throughout the day.

        This method:
        1. Validates that exactly one scenario exists
        2. Validates that no simulation results exist yet (fails if they do)
        3. Runs a consumption simulation to calculate energy usage
        4. Checks which rotations end with SOC below zero at the end of their last trip
        5. Returns None if all rotations are fine, or a list of problematic rotation IDs

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database
        params : Dict[str, Any]
            Pipeline parameters (none required for this analyzer)

        Returns:
        --------
        List[int] | None
            - None if all rotations have sufficient charging time (SOC >= 0)
            - List[int] of rotation IDs that end with SOC < 0, indicating insufficient
              charging time and the need for a new schedule with more slack

        Raises:
        -------
        ValueError
            If simulation results already exist in the database
        """
        # Make sure there is just one scenario
        scenario_q = session.query(Scenario)
        if scenario_q.count() != 1:
            raise ValueError(f"Expected exactly one scenario, found {scenario_q.count()}")
        scenario = scenario_q.one()

        # Check that no simulation results exist yet
        existing_events = session.query(Event).filter(Event.scenario_id == scenario.id).count()
        if existing_events > 0:
            raise ValueError(
                f"Database contains {existing_events} existing simulation results. "
                "Please clear previous simulation results before running this analyzer."
            )

        self.logger.info("Running consumption simulation to check for insufficient charging time")

        # Electrify all charging stations with the given power
        charging_power = params.get(
            f"{self.__class__.__name__}.charging_power_kw", self._get_default_charging_power_kw()
        )

        # Electrify Depots
        depot_q = (
            session.query(Station)
            .filter(Station.scenario_id == scenario.id)
            .filter(Station.charge_type == ChargeType.DEPOT)
        )
        depot_q.update(
            {
                "is_electrified": True,
                "amount_charging_places": 1000,
                "power_per_charger": charging_power,
                "power_total": 1000 * charging_power,
                "charge_type": ChargeType.DEPOT,
                "voltage_level": VoltageLevel.MV,
            }
        )

        # Electrify All termini
        station_q = (
            session.query(Station)
            .filter(Station.scenario_id == scenario.id)
            .filter(Station.charge_type != ChargeType.DEPOT)
        )
        station_q.update(
            {
                "is_electrified": True,
                "amount_charging_places": 1000,
                "power_per_charger": charging_power,
                "power_total": 1000 * charging_power,
                "charge_type": ChargeType.OPPORTUNITY,
                "voltage_level": VoltageLevel.MV,
            }
        )

        # Generate consumption results
        consumption_results = generate_consumption_result(scenario)

        # Run consumption simulation
        simple_consumption_simulation(
            scenario=scenario, initialize_vehicles=True, consumption_result=consumption_results
        )

        # Load all rotations with their trips eagerly to avoid N+1 queries
        all_rotations = (
            session.query(Rotation)
            .filter(Rotation.scenario_id == scenario.id)
            .options(joinedload(Rotation.trips))
            .all()
        )

        # For each rotation, find the last trip and check its final SOC
        rotations_with_low_soc = []

        for rotation in all_rotations:
            if not rotation.trips:
                self.logger.warning(f"Rotation {rotation.id} has no trips, skipping")
                continue

            # Get the last trip
            last_trip = rotation.trips[-1]

            # Find the last driving event for this trip
            last_driving_event = (
                session.query(Event)
                .filter(Event.trip_id == last_trip.id)
                .filter(Event.event_type == EventType.DRIVING)
                .order_by(Event.time_end.desc())
                .first()
            )

            if last_driving_event is None:
                self.logger.warning(
                    f"Rotation {rotation.id}, last trip {last_trip.id} has no driving events, skipping"
                )
                continue

            # Check if the final SOC is below zero
            if last_driving_event.soc_end < 0:
                rotations_with_low_soc.append(rotation.id)
                self.logger.debug(
                    f"Rotation {rotation.id} ends with SOC {last_driving_event.soc_end:.2%} < 0"
                )

        if rotations_with_low_soc:
            self.logger.warning(
                f"Found {len(rotations_with_low_soc)} rotation(s) with insufficient charging time "
                f"(ending with SOC < 0): {rotations_with_low_soc}"
            )
            return rotations_with_low_soc
        else:
            self.logger.info(
                "All rotations have sufficient charging time (SOC >= 0 at end of day)"
            )
            return None
