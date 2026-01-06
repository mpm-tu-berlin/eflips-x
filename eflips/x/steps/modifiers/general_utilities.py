"""
General utility modifiers for data cleanup and maintenance.

This module contains modifiers that perform general data cleanup operations,
such as removing unused routes, lines, and stations from a scenario.
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import eflips.model
from eflips.model import (
    Route,
    Line,
    Station,
    Scenario,
    Temperatures,
    Rotation,
    VehicleType,
    ConsumptionLut,
    VehicleClass,
)
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

    def __init__(self, code_version: str = "v1.0.1", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
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

        # Remove all rotaions that have no trips
        all_rotations = session.query(Rotation).all()
        rotations_removed = 0
        for rotation in all_rotations:
            if len(rotation.trips) == 0:
                self.logger.debug(f"Removing rotation {rotation.name}")
                session.delete(rotation)
                rotations_removed += 1

        self.logger.info(f"Removed {rotations_removed} unused rotations")

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

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_temperature() -> float:
        """Get the default temperature value in Celsius."""
        return -12.0

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary describing the configurable parameter:
            - AddTemperatures.temperature_celsius: Temperature value in degrees Celsius
        """
        return {
            f"{cls.__name__}.temperature_celsius": """
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


class CalculateConsumptionScaling(Modifier):
    """
    Calculate and apply consumption scaling factors based on empirical BVG data.

    This modifier runs trip-level consumption simulations across different temperature profiles
    (12 monthly averages + 2 extreme temperatures) and compares the modeled consumption to
    real-world BVG data. It then scales the consumption lookup tables for specified vehicle
    types to match empirical observations.

    The scaling process:
    1. Simulates consumption for all trips using 14 temperature profiles
    2. Aggregates to quarterly means per vehicle type
    3. Compares to real BVG quarterly consumption data
    4. Calculates scaling factors (real / model)
    5. Applies mean scaling factor to specified vehicle type LUTs
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_monthly_temperatures() -> Dict[str, float]:
        """Get default Berlin monthly average temperatures in Celsius."""
        return {
            "January": 0.0,
            "February": 1.0,
            "March": 5.0,
            "April": 9.0,
            "May": 14.0,
            "June": 17.0,
            "July": 19.0,
            "August": 18.0,
            "September": 14.0,
            "October": 9.0,
            "November": 4.0,
            "December": 1.0,
            "Hottest": 29.6,
            "Coldest": -12.0,
        }

    @staticmethod
    def _get_default_real_quarterly_consumption() -> List[float]:
        """Get real BVG quarterly consumption data in kWh/km."""
        return [
            1.6580115,  # Q1: Jan, Feb, Mar
            1.3629038,  # Q2: Apr, May, Jun
            1.3028950,  # Q3: Jul, Aug, Sep
            1.5893908,  # Q4: Oct, Nov, Dec
        ]

    @staticmethod
    def _get_default_vehicle_types_to_scale() -> List[str]:
        """Get default list of vehicle types to scale."""
        return ["EN", "DD"]

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary describing the configurable parameters
        """
        return {
            f"{cls.__name__}.vehicle_types_to_scale": """
            List of vehicle type short names to apply scaling to.
            Default: ["EN", "DD"]
            Type: List[str]
            Example: ["EN", "DD"]
            """,
            f"{cls.__name__}.real_quarterly_consumption": """
            Real-world quarterly consumption data in kWh/km for comparison.
            Four values for Q1, Q2, Q3, Q4.
            Default: BVG empirical data [1.658, 1.363, 1.303, 1.589]
            Type: List[float]
            Example: [1.658, 1.363, 1.303, 1.589]
            """,
            f"{cls.__name__}.monthly_temperatures": """
            Temperature profiles for each month and extreme days.
            Default: Berlin climate averages
            Type: Dict[str, float]
            """,
        }

    def _calculate_trip_consumption(
        self,
        trip: "eflips.model.Trip",
        temperature: float,
        consumption_lut: "ConsumptionLut",
    ) -> float:
        """
        Calculate consumption for a single trip at a given temperature.

        Parameters:
        -----------
        trip : Trip
            The trip to calculate consumption for
        temperature : float
            Ambient temperature in Celsius
        consumption_lut : ConsumptionLut
            The consumption lookup table to use

        Returns:
        --------
        float
            Consumption in kWh/km
        """
        import numpy as np
        from scipy import interpolate

        # Calculate trip parameters
        total_distance = trip.route.distance / 1000.0  # km
        if total_distance == 0:
            return 0.0

        total_duration = (trip.arrival_time - trip.departure_time).total_seconds() / 3600  # hours
        if total_duration == 0:
            return 0.0

        average_speed = total_distance / total_duration  # km/h

        # Calculate level of loading (assuming average passenger count)
        passenger_mass = 68  # kg
        passenger_count = 17.6  # German-wide average
        payload_mass = passenger_mass * passenger_count
        full_payload = (
            trip.rotation.vehicle_type.allowed_mass - trip.rotation.vehicle_type.empty_mass
        )
        level_of_loading = payload_mass / full_payload if full_payload > 0 else 0.5

        # Extract consumption LUT data
        if not consumption_lut.data_points or not consumption_lut.values:
            self.logger.warning(f"Empty consumption LUT for trip {trip.id}")
            return 0.0

        # Build the 4D interpolator
        incline_scale = sorted(set(x[0] for x in consumption_lut.data_points))
        temperature_scale = sorted(set(x[1] for x in consumption_lut.data_points))
        loading_scale = sorted(set(x[2] for x in consumption_lut.data_points))
        speed_scale = sorted(set(x[3] for x in consumption_lut.data_points))

        # Create 4D array
        consumption_array = np.full(
            (len(incline_scale), len(temperature_scale), len(loading_scale), len(speed_scale)),
            np.nan,
        )

        # Fill array with values
        for i, (incline, temp, loading, speed) in enumerate(consumption_lut.data_points):
            idx = (
                incline_scale.index(incline),
                temperature_scale.index(temp),
                loading_scale.index(loading),
                speed_scale.index(speed),
            )
            consumption_array[idx] = consumption_lut.values[i]

        # Create interpolator
        try:
            interpolator = interpolate.RegularGridInterpolator(
                (incline_scale, temperature_scale, loading_scale, speed_scale),
                consumption_array,
                bounds_error=False,
                fill_value=None,
                method="linear",
            )

            # Interpolate for this trip
            incline = 0.0  # Assume flat terrain
            consumption_per_km = interpolator(
                [incline, temperature, level_of_loading, average_speed]
            )[0]

            return float(consumption_per_km) if not np.isnan(consumption_per_km) else 0.0
        except Exception as e:
            self.logger.warning(f"Interpolation failed for trip {trip.id}: {e}")
            return 0.0

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Calculate consumption scaling factors and apply them to vehicle type LUTs.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters

        Returns:
        --------
        None
            This modifier modifies the database in place
        """
        import numpy as np
        from collections import defaultdict
        import sqlalchemy.orm

        # Get parameters
        vehicle_types_to_scale = params.get(
            f"{self.__class__.__name__}.vehicle_types_to_scale",
            self._get_default_vehicle_types_to_scale(),
        )
        real_quarterly_consumption = params.get(
            f"{self.__class__.__name__}.real_quarterly_consumption",
            self._get_default_real_quarterly_consumption(),
        )
        monthly_temperatures = params.get(
            f"{self.__class__.__name__}.monthly_temperatures",
            self._get_default_monthly_temperatures(),
        )

        # Validate parameters
        if len(real_quarterly_consumption) != 4:
            raise ValueError(
                "real_quarterly_consumption must have exactly 4 values (Q1, Q2, Q3, Q4)"
            )

        # Make sure there is exactly one scenario
        scenarios = session.query(Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")
        scenario = scenarios[0]

        self.logger.info("Starting consumption scaling calculation...")

        # Get all vehicle types and their consumption LUTs
        vehicle_type_luts = {}
        for vt in session.query(VehicleType).filter_by(scenario_id=scenario.id).all():
            if len(vt.vehicle_classes) == 0:
                self.logger.warning(
                    f"Vehicle type {vt.name_short} has no vehicle classes, skipping"
                )
                continue

            lut = None
            for vc in vt.vehicle_classes:
                if vc.consumption_lut is not None:
                    lut = vc.consumption_lut
                    break

            if lut is None:
                self.logger.warning(
                    f"Vehicle type {vt.name_short} has no consumption LUT, skipping"
                )
                continue

            vehicle_type_luts[vt.name_short] = (vt, lut)

        if not vehicle_type_luts:
            self.logger.warning("No vehicle types with consumption LUTs found, skipping scaling")
            return None

        # Query all trips
        from eflips.model import Trip

        all_trips = (
            session.query(Trip)
            .filter(Trip.scenario_id == scenario.id)
            .options(sqlalchemy.orm.joinedload(Trip.rotation).joinedload(Rotation.vehicle_type))
            .options(sqlalchemy.orm.joinedload(Trip.route))
            .all()
        )

        if not all_trips:
            self.logger.warning("No trips found in scenario, skipping scaling")
            return None

        self.logger.info(
            f"Calculating consumption for {len(all_trips)} trips across {len(monthly_temperatures)} temperature profiles..."
        )

        # Calculate consumption for each temperature profile
        # Structure: {month_name: {vehicle_type: [consumptions]}}
        consumption_by_month_and_type = defaultdict(lambda: defaultdict(list))

        for month_name, temperature in monthly_temperatures.items():
            self.logger.info(f"Processing {month_name} ({temperature}°C)...")

            for trip in all_trips:
                vt_name = trip.rotation.vehicle_type.name_short
                if vt_name not in vehicle_type_luts:
                    continue

                _, lut = vehicle_type_luts[vt_name]
                consumption_per_km = self._calculate_trip_consumption(trip, temperature, lut)

                if consumption_per_km > 0:
                    consumption_by_month_and_type[month_name][vt_name].append(consumption_per_km)

        # Calculate mean consumption per vehicle type per month
        mean_consumption = {}
        for month_name in monthly_temperatures.keys():
            mean_consumption[month_name] = {}
            for vt_name in vehicle_type_luts.keys():
                consumptions = consumption_by_month_and_type[month_name][vt_name]
                if consumptions:
                    mean_consumption[month_name][vt_name] = np.mean(consumptions)
                else:
                    mean_consumption[month_name][vt_name] = 0.0

        # Group monthly data into quarterly data (only for months, not extreme temps)
        quarters = [
            ["January", "February", "March"],
            ["April", "May", "June"],
            ["July", "August", "September"],
            ["October", "November", "December"],
        ]

        # Calculate model quarterly consumption (using EN as reference)
        model_quarterly_consumption = []
        for quarter_months in quarters:
            quarter_consumptions = [
                mean_consumption[month]["EN"]
                for month in quarter_months
                if month in mean_consumption and "EN" in mean_consumption[month]
            ]
            if quarter_consumptions:
                model_quarterly_consumption.append(np.mean(quarter_consumptions))
            else:
                model_quarterly_consumption.append(0.0)

        # Calculate scaling factors
        if len(model_quarterly_consumption) != 4:
            self.logger.error("Could not calculate quarterly consumption, skipping scaling")
            return None

        scaling_factors = np.array(real_quarterly_consumption) / np.array(
            model_quarterly_consumption
        )
        mean_scaling_factor = float(np.mean(scaling_factors))

        self.logger.info(f"Quarterly scaling factors: {scaling_factors}")
        self.logger.info(f"Mean scaling factor: {mean_scaling_factor:.4f}")
        self.logger.info(f"Standard deviation: {np.std(scaling_factors):.4f}")

        # Apply scaling to specified vehicle types
        for vt_name in vehicle_types_to_scale:
            if vt_name not in vehicle_type_luts:
                self.logger.warning(f"Vehicle type {vt_name} not found in database, skipping")
                continue

            vt, _ = vehicle_type_luts[vt_name]

            # Find and scale the consumption LUT
            for vc in vt.vehicle_classes:
                if vc.consumption_lut is not None:
                    lut = vc.consumption_lut
                    scaled_values = [v * mean_scaling_factor for v in lut.values]
                    lut.values = scaled_values
                    self.logger.info(
                        f"Scaled consumption LUT for vehicle type {vt_name} by factor {mean_scaling_factor:.4f}"
                    )

        session.flush()
        self.logger.info("Consumption scaling complete")

        return None


class RemoveConsumptionLuts(Modifier):
    """
    Remove consumption lookup tables for diesel reference scenarios.

    This modifier deletes all ConsumptionLut and VehicleClass objects and sets
    a minimal constant consumption value on all VehicleType objects. This is useful
    for creating diesel baseline scenarios where consumption modeling is not needed.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_minimal_consumption() -> float:
        """Get the default minimal consumption value in kWh/km."""
        return 0.001

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary describing the configurable parameter
        """
        return {
            f"{cls.__name__}.minimal_consumption": """
            Minimal consumption value to set on all vehicle types after removing LUTs.
            This should be a very small positive number.
            Default: 0.001 kWh/km
            Type: float
            Example: 0.001
            """,
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Remove all consumption LUTs and set minimal consumption on vehicle types.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters

        Returns:
        --------
        None
            This modifier modifies the database in place
        """
        # Get parameter
        minimal_consumption = params.get(
            f"{self.__class__.__name__}.minimal_consumption",
            self._get_default_minimal_consumption(),
        )

        # Validate parameter
        if minimal_consumption <= 0:
            raise ValueError(f"minimal_consumption must be positive, got {minimal_consumption}")

        # Delete all ConsumptionLut objects
        lut_count = session.query(ConsumptionLut).count()
        session.query(ConsumptionLut).delete()
        self.logger.info(f"Deleted {lut_count} consumption LUTs")

        # Delete all VehicleClass objects
        vc_count = session.query(VehicleClass).count()
        session.query(VehicleClass).delete()
        self.logger.info(f"Deleted {vc_count} vehicle classes")

        # Set minimal consumption on all VehicleType objects
        vt_count = 0
        for vehicle_type in session.query(VehicleType).all():
            vehicle_type.consumption = minimal_consumption
            vt_count += 1

        self.logger.info(
            f"Set minimal consumption ({minimal_consumption} kWh/km) on {vt_count} vehicle types"
        )

        session.flush()

        return None
