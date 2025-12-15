"""
Vehicle scheduling modifiers for rotation optimization.

This module contains modifiers that perform vehicle scheduling operations,
such as creating optimal rotation plans for different vehicle types.
"""

import logging
import os
import typing
from collections import defaultdict, OrderedDict
from datetime import timedelta
from typing import Any, Dict, List, Tuple, Counter, Set

import eflips.model
import pandas as pd
import sqlalchemy.orm.session
from eflips.depot.api import (  # type: ignore[import-untyped]
    generate_consumption_result,
    simple_consumption_simulation,
    ConsumptionResult,
    group_rotations_by_start_end_stop,
)
from eflips.model import ChargeType
from eflips.model import (
    VehicleType,
    Trip,
    TripType,
    Rotation,
    Scenario,
    Route,
    Station,
    Event,
    EventType,
    VoltageLevel,
)
from eflips.opt.depot_rotation_matching import DepotRotationOptimizer
from eflips.opt.scheduling import create_graph, solve, write_back_rotation_plan
from sqlalchemy import func, not_
from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from eflips.x.framework import Modifier, Analyzer


class IntegratedScheduling(Modifier):
    """
    Generate feasible vehicle schedules using an integrated approach. If just using `VehicleScheduling` in opportunity
    charge mode, the resulting schedules may not be feasible. This happens when the energy consumption of the trips
    is so high that even with opportunity charging at every terminal, the vehicle cannot complete the schedule without
    running out of battery. And since we usually have a clear order (Vehicle Scheduling, Depot Assignment, Terminus
    placement), we cannot go back and change the vehicle scheduling after depot assignment and terminus placement. This
    modifier does exactly that: It runs Vehicle Scheduling depot assignment and then terminus placement in a loop until
    a feasible schedule is found. It adds longer breaks to trips that need them until the schedule is feasible.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary describing the configurable parameters:
            - IntegratedScheduling.max_iterations: Maximum number of iterations to attempt
        """
        return {
            f"{cls.__name__}.max_iterations": """
            Maximum number of iterations to attempt for integrated scheduling.
            The modifier will try to create feasible schedules by adjusting break times
            and re-running vehicle scheduling, depot assignment, and terminus placement.

            Note: The heuristic is designed in a way that it should always find a feasible solution in 2 iterations.
            If you need more, this may hint at a deper design issue in the scenario or this could be a bug. Please 
            contact the developers.

            Default: 2
            Type: int
            Example: 10
            """.strip(),
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        This method runs the vehicle scheduling in a loop. It does
        1) Vehicle Scheduling
        2) Depot Assignment
        3) Electrifying *all* trips with opportunity charging at termini
        and then verifies if the resulting schedule is feasible using a consumption simulation.
        If not, it adds longer breaks to trips that need them and repeats the process until a feasible schedule is
        found.

        The database we return will be in the "just after vehicle scheduling" state, as if only VehicleScheduling
        had been run. Therefore, we use nested sessions to run depot assignment and terminus placement without
        modifying the outer session.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters:
            - IntegratedScheduling.max_iterations (optional): Maximum number of iterations to attempt

        Returns:
        --------
        None
            This modifier modifies the database in place by updating rotation plans
        """
        max_iterations_key = f"{self.__class__.__name__}.max_iterations"
        max_iterations = params.get(max_iterations_key, 2)

        if not isinstance(max_iterations, int):
            raise ValueError(
                f"max_iterations must be an integer, got {type(max_iterations).__name__}"
            )
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")
        if max_iterations > 2:
            self.logger.warning(
                "The code is designed to find a feasible solution in 2 iterations."
            )

        self.logger.info(f"Starting integrated scheduling with max_iterations={max_iterations}")

        if (
            "VehicleScheduling.charge_type" not in params
            or params["VehicleScheduling.charge_type"] == ChargeType.DEPOT
        ):
            raise ValueError(
                "IntegratedScheduling only makes sense when VehicleScheduling is run in OPPORTUNITY charge type. "
                "If you are in DEPOT mode, just ise VehicleScheduling alone."
            )

        iteration = 0
        ids_of_trips_to_add_longer_breaks: Set[int] = set()
        while True:
            iteration += 1
            if iteration > max_iterations:
                raise ValueError(
                    f"Reached maximum number of iterations ({max_iterations}) without finding a feasible schedule."
                )
            self.logger.info(f"Integrated scheduling iteration {iteration}")

            params[f"{VehicleScheduling.__name__}.longer_break_time_trips"] = list(
                ids_of_trips_to_add_longer_breaks
            )

            vehicle_scheduling = VehicleScheduling()
            vehicle_scheduling.modify(session, params)

            # Now, we begin a nested session in order to run depot assignment and terminus placement without
            # modifying the outer session
            savepoint = session.begin_nested()
            try:
                depot_assignment = DepotAssignment()
                depot_assignment.modify(session, params)

                insufficient_charging_analyzer = InsufficientChargingTimeAnalyzer()
                insufficient_charging_analyzer_result = insufficient_charging_analyzer.analyze(
                    session, params
                )
                if insufficient_charging_analyzer_result is None:
                    self.logger.info(
                        "Found feasible schedule with sufficient charging time for all rotations."
                    )
                    break
                else:
                    self.logger.info(
                        f"Schedule is not feasible, {len(insufficient_charging_analyzer_result)} rotations "
                        "have insufficient charging time. Adding longer breaks and retrying."
                    )
                    # Now, we will need to identify which trips are in the problematic rotations and which of these
                    # should have longer breaks added.
                    new_trips_to_add_longer_breaks = self.find_trips_to_add_longer_breaks(
                        insufficient_charging_analyzer_result, session, params
                    )
                    ids_of_trips_to_add_longer_breaks.update(new_trips_to_add_longer_breaks)
                    self.logger.info(
                        f"Trying scheudling again, adding longer breaks to {len(new_trips_to_add_longer_breaks)} trips."
                    )
                    # Rollback the nested session to discard changes
            finally:
                savepoint.rollback()

    def find_trips_to_add_longer_breaks(
        self,
        rotation_ids: List[int],
        session: sqlalchemy.orm.session.Session,
        params: Dict[str, Any],
    ) -> set[int]:
        """
        Identify trips within the given rotations that should have longer breaks added. We have conflicting goals of
        1) Minizing the number of trips that need longer breaks added and
        2) Putting the longer breaks at useful locations (i.e., where the vehicle can charge).

        The heuristic approach does the following:
        1) It looks (globally) at the palces where *all* vehicles spend the longest time anyway
        2) It looks (for each rotation) at the termini, taking the ones the vehicle visits on at least 33% of its trips (or the top five, if there are no such termini)
        3) from the possible termini from 2) it selects the one that is closest to the global top locations from 1)

        Parameters:
        -----------
        rotation_ids : List[int]
            List of rotation IDs that have insufficient charging time
        nested_session : Session
            SQLAlchemy session connected to the database
        params : Dict[str, Any]
            Pipeline parameters

        Returns:
        --------
        set[int]
            Set of trip IDs that should have longer breaks added
        """
        # 1) Create the global list of top locations where vehicles spend the most time
        total_length_of_stay_per_station: Dict[int, timedelta] = defaultdict(
            timedelta
        )  # station_id -> total length of stay
        all_rotations = (
            session.query(Rotation)
            .options(joinedload(Rotation.trips).joinedload(Trip.route))
            .all()
        )
        for rotation in all_rotations:
            # We are looking at the break time form an "after trip" perspective, so we skip the first trip
            for i, trip in enumerate(rotation.trips[:-1]):
                arrival_station_id = trip.route.arrival_station_id
                cur_trip_end = trip.arrival_time
                next_trip_start = rotation.trips[i + 1].departure_time
                length_of_stay = next_trip_start - cur_trip_end
                total_length_of_stay_per_station[arrival_station_id] += length_of_stay

        # Sort stations by total length of stay descending
        total_length_of_stay_per_station = OrderedDict(
            sorted(
                total_length_of_stay_per_station.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # 2) For each rotation with insufficient charging time, find candidate termini for longer breaks
        trip_ids_to_add_longer_breaks = set()
        for rotation_id in rotation_ids:
            rotation = (
                session.query(Rotation)
                .filter(Rotation.id == rotation_id)
                .options(joinedload(Rotation.trips))
                .one()
            )
            candidate_termini_station_ids: Dict[int, int] = defaultdict(
                int
            )  # station_id -> count of visits
            for trip in rotation.trips[:-1]:  # Skip last trip, as no break after it
                arrival_station_id = trip.route.arrival_station_id
                candidate_termini_station_ids[arrival_station_id] += 1

            # Filter termini that are visited on at least 33% of trips
            total_trips = len(rotation.trips)
            filtered_candidate_termini = {
                station_id: count
                for station_id, count in candidate_termini_station_ids.items()
                if count >= total_trips / 3
            }
            # If no termini pass the filter, take the top five most visited termini
            if len(filtered_candidate_termini) < 1:
                filtered_candidate_termini = dict(
                    sorted(
                        candidate_termini_station_ids.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:5]
                )
            # 3) From the filtered candidate termini, select the one closest to the global top locations
            for station_id in total_length_of_stay_per_station.keys():
                if station_id in filtered_candidate_termini:
                    # Find the trips that arrive at this station and add them to the list
                    for trip in rotation.trips:
                        if trip.route.arrival_station_id == station_id:
                            trip_ids_to_add_longer_breaks.add(trip.id)
                            self.logger.info(
                                f"Adding longer break to trip {trip.id} in rotation {rotation.id} "
                                f"at station {station_id}"
                            )
                    break  # Found the best candidate, move to next rotation

        return trip_ids_to_add_longer_breaks


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

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
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

    @staticmethod
    def _get_default_charge_type() -> ChargeType:
        """Get the default charge type."""
        return ChargeType.DEPOT

    @classmethod
    def document_params(cls) -> Dict[str, str]:
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
            f"{cls.__name__}.minimum_break_time": """
            Minimum break time required between trips.
            This ensures drivers have adequate rest between consecutive trips.

            Default: timedelta(seconds=0)
            Type: timedelta
            Example: timedelta(minutes=15)
            """.strip(),
            f"{cls.__name__}.maximum_schedule_duration": """
            Maximum duration of a vehicle schedule.
            This limits how long a single vehicle can operate before returning to depot.

            Default: timedelta(hours=24)
            Type: timedelta
            Example: timedelta(hours=12, minutes=30)
            """.strip(),
            f"{cls.__name__}.battery_margin": """
            Battery safety margin as a fraction (0.0 to 1.0).
            This reduces the effective battery capacity to ensure vehicles don't fully deplete.
            For example, 0.1 means 10% of battery capacity is reserved as safety margin.

            Default: 0.1 (10%)
            Type: float
            Example: 0.15 (15% margin)
            """.strip(),
            f"{cls.__name__}.longer_break_time_trips": """
            List of trip IDs that require a longer break time after completion.
            This can be used to specify trips that require additional rest or preparation time.

            Default: [] (empty list)
            Type: List[int]
            Example: [123, 456, 789]
            """.strip(),
            f"{cls.__name__}.longer_break_time_duration": """
            Additional break time duration for trips specified in longer_break_time_trips.
            This duration is added on top of the minimum_break_time.

            Default: timedelta(minutes=5)
            Type: timedelta
            Example: timedelta(minutes=10)
            """.strip(),
            f"{cls.__name__}.charge_type": """
            The charge type to consider for scheduling optimization. When in ChargeType.DEPOT mode, the schedule is
            created in a way that respects battery constraints, making the trip sequences (blocks) only as long as
            the battery allows. In ChargeType.OPPORTUNITY mode, the optimizer assumes that vehicles can charge
            opportunistically at termini, allowing for longer trip sequences. Therefore, ChargeType.OPPORTUNITY
            does not limit the length of blocks based on battery constraints, potentially resulting in more efficient
            schedules.
            Default: ChargeType.DEPOT
            Type: ChargeType
            Example: ChargeType.OPPORTUNITY
            """.strip(),
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
        charge_type = params.get(
            f"{self.__class__.__name__}.charge_type", self._get_default_charge_type()
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
        if not isinstance(charge_type, ChargeType):
            raise ValueError(f"charge_type must be a ChargeType, got {type(charge_type).__name__}")

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

        # If we are in DEPOT charge type, calculate consumption results, they will be fed into the optimizer
        delta_socs: Dict[int, float] | None
        if charge_type == ChargeType.DEPOT:
            self.logger.info("Calculating consumption results for DEPOT charge type")
            # Create a dictionary of the energy consumption for each rotation
            consumption: Dict[int, ConsumptionResult] = generate_consumption_result(scenario)

            # Convert to the format eflips-opt expects: {trip_id: delta_soc, ...}
            # Also turn the delta_soc into a positive number
            delta_socs = {}
            for trip_id, result in consumption.items():
                delta_socs[trip_id] = -1 * result.delta_soc_total / (1 - battery_margin)
        else:
            self.logger.info("Skipping consumption calculation for OPPORTUNITY charge type")
            delta_socs = None

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

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
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

    @classmethod
    def document_params(cls) -> Dict[str, str]:
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
            f"{cls.__name__}.depot_config": """
            A list of depot configurations. Each configuration is a dict with:
            - "depot_station": Station ID or (lon, lat) tuple
            - "capacity": Depot capacity in 12m bus equivalents
            - "vehicle_type": List of allowed vehicle type IDs
            - "name": Depot name (for new depots only)

            Required: True
            Type: List[Dict]
            Example: depots_for_bvg(db_session) from eflips.x.steps.modifiers.bvg_tools
            """,
            f"{cls.__name__}.base_url": """
            Base URL for the OpenRouteService (ORS) routing API.
            Used to calculate travel distances between depots and rotation start/end points.

            Required: True
            Type: str
            Example: "http://mpm-v-ors.mpm.tu-berlin.de:8080/ors/"
            """,
            f"{cls.__name__}.depot_usage": """
            Initial depot capacity usage factor (0.0 to 1.0).
            The optimizer starts with this fraction of nominal depot capacity and
            iteratively reduces it to find the minimal feasible assignment.
            For example, 1.0 means 100% of depot capacity is available initially.

            Default: 1.0 (100% capacity)
            Type: float
            Example: 0.9 (90% capacity)
            """,
            f"{cls.__name__}.step_size": """
            Capacity reduction step size per iteration (0.0 to 1.0).
            After each successful optimization, depot capacities are reduced by this factor.
            Smaller step sizes find tighter capacity bounds but take more iterations.

            Default: 0.1 (reduce by 10% each iteration)
            Type: float
            Example: 0.05 (reduce by 5% each iteration)
            """,
            f"{cls.__name__}.max_iterations": """
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
            - DepotAssignment.base_url (optional): ORS routing service URL. Required, but tries to load it from environment variables if unset.
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
        assert isinstance(optimizer.data["rotation"], pd.DataFrame)
        assert isinstance(optimizer.data["result"], pd.DataFrame)
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
    ) -> None:
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

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters of this analyzer.

        Returns:
        --------
        Dict[str, str]
            This analyzer does not use any configurable parameters.
        """
        return {
            cls.__name__
            + ".charging_power_kw": """
            The charging power in kW to assume for all charging stations during the analysis. Default is 150 kW.
            """.strip()
        }

    @classmethod
    def _get_default_charging_power_kw(cls) -> float:
        """Get the default charging power in kW."""
        return 450.0

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
            .filter(or_(Station.charge_type == None, Station.charge_type != ChargeType.DEPOT))
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


class StationElectrification(Modifier):
    """
    Electrify stations to ensure rotations have sufficient charging opportunities.

    This modifier iteratively adds charging infrastructure at strategic termini until all
    rotations can complete their schedules without running below 0% SOC. It uses a heuristic
    approach to select stations where rotations with low SOC spend the most time.

    The modifier incorporates the logic from the original do_station_electrification() function
    and the utility functions from util_station_electrification module.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_charging_power_kw() -> float:
        """Get the default charging power in kW."""
        return 450.0

    def _get_default_max_stations(self, session: Session) -> int:
        """Get the default maximum number of stations to electrify (25% of termini)."""
        termini_count = len(self._get_all_termini(session))
        return max(1, int(termini_count * 0.25))

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary describing the configurable parameters:
            - StationElectrification.charging_power_kw: Charging power in kW
            - StationElectrification.max_stations_to_electrify: Maximum number of stations to electrify
        """
        return {
            f"{cls.__name__}.charging_power_kw": """
            The charging power in kW for opportunity charging stations added during
            station electrification. This parameter determines how quickly vehicles can
            charge at termini. Higher power means faster charging but may require more
            expensive infrastructure.

            Default: 450.0 kW
            Type: float
            Example: 300.0
            """.strip(),
            f"{cls.__name__}.max_stations_to_electrify": """
            Maximum number of terminus stations to electrify before giving up. This prevents
            the algorithm from attempting to electrify the entire network if the schedule is
            fundamentally infeasible. If this limit is reached, an exception is raised.

            Default: 25% of all termini in the network (minimum 1)
            Type: int
            Example: 50
            """.strip(),
        }

    def _get_all_termini(self, session: Session) -> List[int]:
        """
        Get all terminus station IDs (non-depot stations where routes arrive/depart).

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database

        Returns:
        --------
        List[int]
            List of station IDs that are termini (excluding depots)
        """
        # Get all stations that are arrival or departure points for routes
        # but are not depot stations

        scenario_q = session.query(Scenario)
        if scenario_q.count() != 1:
            raise ValueError(f"Expected exactly one scenario, found {scenario_q.count()}")
        scenario = scenario_q.one()

        arrival_station_ids = (
            session.query(Route.arrival_station_id)
            .join(Station, Route.arrival_station_id == Station.id)
            .filter(Station.scenario_id == scenario.id)
            .filter(or_(Station.charge_type == None, Station.charge_type != ChargeType.DEPOT))
            .distinct()
            .all()
        )

        departure_station_ids = (
            session.query(Route.departure_station_id)
            .join(Station, Route.departure_station_id == Station.id)
            .filter(Station.scenario_id == scenario.id)
            .filter(or_(Station.charge_type == None, Station.charge_type != ChargeType.DEPOT))
            .distinct()
        )

        # Combine and deduplicate
        terminus_ids = set(
            [sid for (sid,) in arrival_station_ids] + [sid for (sid,) in departure_station_ids]
        )

        return list(terminus_ids)

    @staticmethod
    def _make_depot_stations_electrified(scenario: Scenario, session: Session) -> None:
        """
        Before running SimBA for the first time, we need to make sure that the depot stations (The ones where rotations
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

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Electrify stations iteratively until all rotations are feasible.

        This method:
        1. Validates that exactly one scenario exists
        2. Sets up initial depot electrification
        3. Removes terminus charging from rotations that don't need it
        4. Iteratively adds charging stations at strategic termini until no rotations end with SOC below 0%

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters:
            - StationElectrification.charging_power_kw (optional): Charging power in kW
            - StationElectrification.max_stations_to_electrify (optional): Max stations to electrify

        Returns:
        --------
        None
            This modifier modifies the database in place by electrifying stations

        Raises:
        -------
        ValueError
            If charging power differs from InsufficientChargingTimeAnalyzer setting
            If maximum stations limit is reached without achieving feasibility
        """
        # Get parameters
        charging_power_key = f"{self.__class__.__name__}.charging_power_kw"
        max_stations_key = f"{self.__class__.__name__}.max_stations_to_electrify"

        charging_power = params.get(charging_power_key, self._get_default_charging_power_kw())
        max_stations = params.get(max_stations_key, self._get_default_max_stations(session))

        # Validate charging power
        if not isinstance(charging_power, (int, float)):
            raise ValueError(
                f"charging_power_kw must be a number, got {type(charging_power).__name__}"
            )
        if charging_power <= 0:
            raise ValueError(f"charging_power_kw must be positive, got {charging_power}")

        # Validate max_stations
        if not isinstance(max_stations, int):
            raise ValueError(
                f"max_stations_to_electrify must be an integer, got {type(max_stations).__name__}"
            )
        if max_stations <= 0:
            raise ValueError(f"max_stations_to_electrify must be positive, got {max_stations}")

        # Check for power mismatch with InsufficientChargingTimeAnalyzer
        analyzer_power_key = f"{InsufficientChargingTimeAnalyzer.__name__}.charging_power_kw"
        analyzer_power = params.get(
            analyzer_power_key, InsufficientChargingTimeAnalyzer._get_default_charging_power_kw()
        )
        our_power_key = f"{self.__class__.__name__}.charging_power_kw"
        our_power = params.get(our_power_key, self._get_default_charging_power_kw())

        if analyzer_power != our_power:
            raise ValueError(
                f"Charging power mismatch: InsufficientChargingTimeAnalyzer uses "
                f"{analyzer_power} kW, but StationElectrification uses {our_power} kW. "
                f"Please ensure both use the same charging power."
            )

        # Make sure there is just one scenario
        scenario_q = session.query(Scenario)
        if scenario_q.count() != 1:
            raise ValueError(f"Expected exactly one scenario, found {scenario_q.count()}")
        scenario = scenario_q.one()

        termini_count = len(self._get_all_termini(session))
        self.logger.info(
            f"Starting station electrification with charging power {charging_power} kW, "
            f"max stations to electrify: {max_stations} (total termini: {termini_count})"
        )

        # Make the depots electrified
        self._make_depot_stations_electrified(scenario, session)

        # Remove terminus charging from rotations that don't need it
        self._remove_terminus_charging_from_okay_rotations(scenario, session)

        # Track number of stations electrified
        stations_electrified = 0

        # Iteratively add charging stations until all rotations are feasible or limit reached
        while True:
            # Check if we've exceeded the limit
            if stations_electrified >= max_stations:
                num_rotations_failing = self._number_of_rotations_below_zero(scenario, session)
                raise ValueError(
                    f"Station electrification failed: electrified {stations_electrified} stations "
                    f"(limit: {max_stations}) but {num_rotations_failing} rotation(s) still have "
                    f"SOC below 0%.\n\n"
                    f"Possible solutions:\n"
                    f"1. Increase the 'max_stations_to_electrify' parameter (currently {max_stations})\n"
                    f"2. Check if the schedule is feasible - have you run InsufficientChargingTimeAnalyzer?\n"
                    f"3. Consider using IntegratedScheduling which adds charging breaks to the schedule\n"
                    f"4. Review network design - the scenario may require more charging opportunities or "
                    f"higher charging power (currently {charging_power} kW)\n"
                    f"5. Check vehicle battery capacity and consumption models for unrealistic values"
                )

            savepoint = session.begin_nested()
            # Run the consumption model to assess current SOC levels
            try:
                consumption_results = generate_consumption_result(scenario)
                simple_consumption_simulation(
                    scenario, initialize_vehicles=True, consumption_result=consumption_results
                )

                if self.logger.isEnabledFor(logging.INFO):
                    min_soc_end_query = (
                        session.query(func.min(Event.soc_end))
                        .filter(Event.scenario_id == scenario.id)
                        .limit(1)
                    )
                    min_soc_end = min_soc_end_query.scalar()
                    count_of_electrified_termini = (
                        session.query(Station)
                        .filter(Station.scenario_id == scenario.id, Station.is_electrified == True)
                        .count()
                    )
                    self.logger.info(
                        f"Minimum SOC at end of day: {min_soc_end}, Electrified termini: {count_of_electrified_termini}"
                    )

                    # Log the current state
                    number_of_electrified_termini = (
                        session.query(Station)
                        .filter(Station.scenario_id == scenario.id)
                        .filter(Station.is_electrified == True)
                        .count()
                    )
                    self.logger.info(
                        f"Number of rotations with SOC below 0%: {self._number_of_rotations_below_zero(scenario, session)}, "
                        f"Number of electrified termini: {number_of_electrified_termini}, "
                        f"Stations electrified in this run: {stations_electrified}/{max_stations}"
                    )

                if self._number_of_rotations_below_zero(scenario, session) == 0:
                    # All rotations are feasible now
                    break

                electrified_station_id = self._identify_charging_station_to_add(scenario, session)

                if electrified_station_id is None:
                    num_rotations_failing = self._number_of_rotations_below_zero(scenario, session)
                    raise ValueError(
                        f"Station electrification failed: cannot add more charging stations "
                        f"(all candidate stations have zero score), but {num_rotations_failing} "
                        f"rotation(s) still have SOC below 0%.\n\n"
                        f"This indicates a fundamental issue with the schedule or network design:\n"
                        f"1. Have you run InsufficientChargingTimeAnalyzer before scheduling?\n"
                        f"2. Consider using IntegratedScheduling instead of plain VehicleScheduling\n"
                        f"3. Some rotations may not have any suitable terminus for charging\n"
                        f"4. Check vehicle battery capacity and consumption models"
                    )

            finally:
                savepoint.rollback()  # Remove simulation results to keep DB clean
                session.expire_all()  # Detach all objects to avoid stale data

            # Actually electrify the selected station
            station_to_electrify = (
                session.query(Station).filter(Station.id == electrified_station_id).one()
            )
            station_to_electrify.is_electrified = True
            station_to_electrify.amount_charging_places = 100
            station_to_electrify.power_per_charger = charging_power
            station_to_electrify.power_total = (
                station_to_electrify.amount_charging_places
                * station_to_electrify.power_per_charger
            )
            station_to_electrify.charge_type = ChargeType.OPPORTUNITY
            station_to_electrify.voltage_level = VoltageLevel.MV

            self.logger.info(
                f"Added charging station {session.query(Station).filter(Station.id == electrified_station_id).one().name} "
                f"({electrified_station_id}) to scenario"
            )
            stations_electrified += 1

        session.flush()
        self.logger.info(
            f"Station electrification completed successfully after electrifying {stations_electrified} stations"
        )

        return None

    def _remove_terminus_charging_from_okay_rotations(
        self,
        scenario: Scenario,
        session: Session,
    ) -> None:
        """
        Run the consumption model and remove terminus charging from rotations that don't need it.

        Parameters:
        -----------
        scenario : Scenario
            The scenario to process
        session : Session
            An open database session
        """

        self.logger.info("Removing terminus charging from rotations that don't need it")

        savepoint = session.begin_nested()
        try:
            # Run the consumption model
            consumption_results = generate_consumption_result(scenario)

            # `create_consumption_results` may have detached our scenario object from the session
            session.add(scenario)
            session.flush()

            simple_consumption_simulation(
                initialize_vehicles=True,
                scenario=scenario,
                consumption_result=consumption_results,
            )

            # Get the rotations with low SoC
            low_soc_rot_q = (
                session.query(Rotation.id)
                .join(Trip)
                .join(Event)
                .filter(Rotation.scenario_id == scenario.id)
                .filter(Event.event_type == EventType.DRIVING)
                .filter(Event.soc_end < 0)
                .distinct()
            )
            high_soc_rot_q = (
                session.query(Rotation)
                .filter(Rotation.scenario_id == scenario.id)
                .filter(not_(Rotation.id.in_(low_soc_rot_q)))
            )

            self.logger.info(
                f"{low_soc_rot_q.count()} rotations with low SoC, {high_soc_rot_q.count()} with high SoC, "
                f"{session.query(Rotation).filter(Rotation.scenario_id == scenario.id).count()} total rotations"
            )

            # Put the IDs of all high SoC rotations into a list to have them available after we're rolling back the
            # simulation results
            high_soc_rot_ids = [rotation.id for rotation in high_soc_rot_q]
        finally:
            savepoint.rollback()  # Remove simulation results to keep DB clean
            session.expire_all()  # Detach all objects to avoid stale data

        # For the rotations with high SoC, remove the ability to charge at the terminus
        session.query(Rotation).filter(Rotation.id.in_(high_soc_rot_ids)).update(
            {"allow_opportunity_charging": False}
        )
        session.flush()

    def _number_of_rotations_below_zero(self, scenario: Scenario, session: Session) -> int:
        """
        Count the number of rotations with SOC below 0%.

        Parameters:
        -----------
        scenario : Scenario
            The scenario to check
        session : Session
            The database session

        Returns:
        --------
        int
            The number of rotations with SOC below 0%
        """
        rotations_q = (
            session.query(Rotation)
            .join(Trip)
            .join(Event)
            .filter(Rotation.scenario_id == scenario.id)
            .filter(Event.event_type == EventType.DRIVING)
            .filter(Event.soc_end < 0)
            .distinct()
        )
        return rotations_q.count()

    def _identify_charging_station_to_add(
        self,
        scenario: Scenario,
        session: Session,
    ) -> int | None:
        """
        Identify a charging station to add based on heuristic.

        The heuristic selects the station where rotations with negative SoC spend the most time.
        If the selected station is already electrified, the next best station is selected.

        Parameters:
        -----------
        scenario : Scenario
            The scenario to add the charging station to
        session : Session
            An open database session
        power : float
            The power of the charging station in kW

        Returns:
        --------
        int | None
            The ID of the charging station that was added, or None if no station could be added
        """
        self.logger.setLevel(logging.INFO)  # TODO remove after debugging

        # Identify all rotations with SOC < 0
        rotations_with_low_soc = (
            session.query(Rotation)
            .join(Trip)
            .join(Event)
            .filter(Event.soc_end < 0)
            .filter(Event.event_type == EventType.DRIVING)
            .filter(Event.scenario == scenario)
            .options(sqlalchemy.orm.joinedload(Rotation.trips).joinedload(Trip.route))
            .distinct()
            .all()
        )

        # For these rotations, find all arrival stations except the last one (depot)
        # Sum up the time spent at each station
        total_break_time_by_station: typing.Counter[int] = Counter()
        for rotation in rotations_with_low_soc:
            for i in range(len(rotation.trips) - 1):
                trip = rotation.trips[i]
                total_break_time_by_station[trip.route.arrival_station_id] += int(
                    (rotation.trips[i + 1].departure_time - trip.arrival_time).total_seconds()
                )

        # If all stations have a score of 0, we can't add any more stations
        if all(v == 0 for v in total_break_time_by_station.values()):
            return None

        # Select the station with the highest score that isn't already electrified
        for most_popular_station_id, _ in total_break_time_by_station.most_common():
            station: Station = (
                session.query(Station).filter(Station.id == most_popular_station_id).one()
            )
            if station.is_electrified:
                self.logger.warning(
                    f"Station {station.name} is already electrified. Choosing the next best station."
                )
                continue

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"Station {most_popular_station_id} ({station.name}) was selected as the station "
                    "where the most time is spent."
                )

            return most_popular_station_id

        return None
