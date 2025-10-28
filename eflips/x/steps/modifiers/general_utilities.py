"""
General utility modifiers for data cleanup and maintenance.

This module contains modifiers that perform general data cleanup operations,
such as removing unused routes, lines, and stations from a scenario.
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

import eflips.model
from eflips.model import Route, Line, Station, Scenario, Temperatures
from sqlalchemy.orm import Session

from eflips.x.framework import Modifier


class RemoveUnusedData(Modifier):
    """
    Remove unused data from a scenario database.

    This modifier performs cleanup operations to remove database entries that are no longer
    referenced or used. This is useful after other modifiers have removed rotations or trips,
    leaving orphaned database entries.

    The modifier performs the following cleanup operations in order:
    1. Removes all routes that have no trips
    2. Removes all lines that have no routes
    3. Removes all stations that are not part of any route
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    def document_params(self) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        This modifier has no configurable parameters.

        Returns:
        --------
        Dict[str, str]
            Empty dictionary as this modifier takes no parameters
        """
        return {}

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Remove unused routes, lines, and stations from the database.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters (not used by this modifier)

        Returns:
        --------
        None
            This modifier modifies the database in place and doesn't return a specific result
        """
        # Make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        # Clean up the data
        # Remove all routes that have no trips
        all_routes = session.query(Route).all()
        routes_removed = 0
        for route in all_routes:
            if len(route.trips) == 0:
                self.logger.debug(f"Removing route {route.name}")
                for assoc_route_station in route.assoc_route_stations:
                    session.delete(assoc_route_station)
                session.delete(route)
                routes_removed += 1

        self.logger.info(f"Removed {routes_removed} unused routes")

        # Remove all lines that have no routes
        all_lines = session.query(Line).all()
        lines_removed = 0
        for line in all_lines:
            if len(line.routes) == 0:
                self.logger.debug(f"Removing line {line.name}")
                session.delete(line)
                lines_removed += 1

        self.logger.info(f"Removed {lines_removed} unused lines")

        # Remove all stations that are not part of a route
        all_stations = session.query(Station).all()
        stations_removed = 0
        for station in all_stations:
            if (
                len(station.assoc_route_stations) == 0
                and len(station.routes_departing) == 0
                and len(station.routes_arriving) == 0
            ):
                self.logger.debug(f"Removing station {station.name}")
                session.delete(station)
                stations_removed += 1

        self.logger.info(f"Removed {stations_removed} unused stations")

        # Log the number of remaining objects
        remaining_routes = session.query(Route).count()
        remaining_lines = session.query(Line).count()
        remaining_stations = session.query(Station).count()

        self.logger.info(
            f"After cleanup: {remaining_routes} routes, {remaining_lines} lines, "
            f"{remaining_stations} stations remain"
        )

        session.flush()

        return None


class AddTemperatures(Modifier):
    """
    Add constant temperature data to all scenarios in the database.

    This modifier creates a Temperatures object for each scenario with a constant
    temperature value across the entire possible time range. This is useful for consumption
    simulations that require temperature data.

    The temperature is applied uniformly from datetime.min to datetime.max in UTC.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_temperature() -> float:
        """Get the default temperature value in Celsius."""
        return -12.0

    def document_params(self) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary describing the configurable parameter:
            - AddTemperatures.temperature_celsius: Temperature value in degrees Celsius
        """
        return {
            f"{self.__class__.__name__}.temperature_celsius": """
            Temperature value in degrees Celsius to apply to all scenarios.
            This will be used as a constant temperature throughout all time.

            Default: -12.0 °C
            Type: float
            Example: -12.0
            """,
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Add constant temperature data to all scenarios.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters:
            - AddTemperatures.temperature_celsius (optional): Temperature in °C (default: -12.0)

        Returns:
        --------
        None
            This modifier modifies the database in place by adding Temperatures objects
        """
        # Get parameters
        temp_key = f"{self.__class__.__name__}.temperature_celsius"
        temperature_celsius = params.get(temp_key, self._get_default_temperature())

        # Emit warning if using default
        if temp_key not in params:
            warnings.warn(
                f"Using default temperature: {temperature_celsius}°C. "
                f"Set '{temp_key}' in params to specify a different temperature.",
                UserWarning,
            )

        # Validate parameter
        if not isinstance(temperature_celsius, (int, float)):
            raise ValueError(
                f"Temperature must be a number, got {type(temperature_celsius).__name__}"
            )

        # Make sure there is exactly one scenario
        scenarios = session.query(Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        # Create the temperature data using datetime.min and datetime.max in UTC
        tz_utc = ZoneInfo("UTC")
        datetimes = [
            datetime.min.replace(tzinfo=tz_utc),
            datetime.max.replace(tzinfo=tz_utc),
        ]
        temps = [float(temperature_celsius), float(temperature_celsius)]

        # Add temperature data to each scenario
        for scenario in scenarios:
            scenario_temperatures = Temperatures(
                scenario_id=scenario.id,
                name=f"{temperature_celsius} °C",
                use_only_time=False,
                datetimes=datetimes,
                data=temps,
            )
            session.add(scenario_temperatures)
            self.logger.info(
                f"Added temperature data ({temperature_celsius}°C) to scenario '{scenario.name}'"
            )

        session.flush()

        return None
