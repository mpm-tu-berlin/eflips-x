"""
Vehicle scheduling modifiers for rotation optimization.

This module contains modifiers that perform vehicle scheduling operations,
such as creating optimal rotation plans for different vehicle types.
"""

import logging
from datetime import timedelta
from typing import Any, Dict, List

import eflips.model
from eflips.depot.api import generate_consumption_result, ConsumptionResult
from eflips.model import VehicleType, Trip, TripType, Rotation
from eflips.opt.scheduling import create_graph, solve, write_back_rotation_plan
from sqlalchemy.orm import Session

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
