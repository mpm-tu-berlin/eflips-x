"""
General utility modifiers for data cleanup and maintenance.

This module contains modifiers that perform general data cleanup operations,
such as removing unused routes, lines, and stations from a scenario.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
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
    EnergySource,
    BatteryType,
    ChargingPointType,
)

from eflips.impact.utils import complete_fleet  # type: ignore[import-untyped]
from eflips.impact.tco import init_tco_params  # type: ignore[import-untyped]
from eflips.impact.lca import init_lca_params  # type: ignore[import-untyped]

from sqlalchemy.orm import Session

from eflips.x.framework import Modifier
from eflips.x.steps.modifiers.consumption_luts import load_measured_speed_temperature_lut


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
        from scipy import interpolate  # type: ignore[import-untyped]

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
        consumption_by_month_and_type: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

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
        mean_consumption: Dict[str, Dict[str, float]] = {}
        for month_name in monthly_temperatures.keys():
            mean_consumption[month_name] = {}
            for vt_name in vehicle_type_luts.keys():
                consumptions = consumption_by_month_and_type[month_name][vt_name]
                if consumptions:
                    mean_consumption[month_name][vt_name] = float(np.mean(consumptions))
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
                model_quarterly_consumption.append(float(np.mean(quarter_consumptions)))
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


class CalibrateConsumptionLut(Modifier):
    """
    Generate a Ji2022 consumption look-up table per vehicle type and calibrate it
    against a measured speed × temperature table.

    This implements the "calibration target" use of the regression model. The
    synthetic four-dimensional table (incline × temperature × level of loading ×
    speed) produced by ``eflips.model.ConsumptionLut.from_vehicle_type`` covers the
    whole parameter space, but its absolute level is only as good as the regression
    it is derived from. A measured table (layout: see
    ``data/input/consumption_lut_gn.xlsx``) resolves only speed and temperature, at
    zero incline and a mean passenger load. The modifier

    1. builds the synthetic table from the vehicle type's empty and allowed mass,
    2. evaluates it at every measured (speed, temperature) point that lies inside
       the synthetic grid, at incline 0 and level of loading 0.5,
    3. computes the ratio measured / synthetic at each of these points,
    4. interpolates the ratio over (temperature, speed) — linearly inside the
       measured region, nearest neighbour outside it — and
    5. multiplies the flat-ground part of every entry by the ratio at its own
       temperature and speed, for all loads, and adds the incline offset of the
       synthetic table back unchanged.

    The slope term of the synthetic table is purely additive (potential energy per
    kilometre), and the measurements contain no incline information, so the incline
    offset ``c(i, T, l, v) - c(0, T, l, v)`` is deliberately not scaled. The calibrated
    table therefore reproduces the measured values on flat ground at mean load,
    extends them to other loads with the shape of the regression model, and keeps
    the physics-based slope term as generated.

    Any existing consumption LUT and vehicle class on the affected vehicle types is
    replaced, and ``VehicleType.consumption`` is set to None (a vehicle type may
    carry either a constant consumption or a LUT, not both).
    """

    def __init__(
        self,
        measured_lut_path: Union[str, Path],
        code_version: str = "v1.0.0",
        **kwargs: Any,
    ):
        self.measured_lut_path = Path(measured_lut_path)
        super().__init__(
            additional_files=[self.measured_lut_path],
            code_version=code_version,
            **kwargs,
        )
        self.logger = logging.getLogger(__name__)

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
            f"{cls.__name__}.vehicle_type_names": """
            List of vehicle type short names (``VehicleType.name_short``) to calibrate.
            Every listed vehicle type must have ``empty_mass`` and ``allowed_mass`` set.
            Default: None (all vehicle types in the scenario)
            Type: Optional[List[str]]
            Example: ["GN"]
            """,
        }

    @staticmethod
    def calibrate_values(
        data_points: Sequence[Sequence[float]],
        values: Sequence[float],
        measured: Sequence[Tuple[float, float, float]],
        incline: float = 0.0,
        level_of_loading: float = 0.5,
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Scale a synthetic 4D consumption table so that it matches measured points.

        Parameters:
        -----------
        data_points : Sequence[Sequence[float]]
            LUT coordinates in the order (incline, t_amb, level_of_loading, mean_speed_kmh).
        values : Sequence[float]
            Synthetic consumption values (kWh/km), one per coordinate.
        measured : Sequence[Tuple[float, float, float]]
            Measured points as ``(mean_speed_kmh, t_amb, consumption_kwh_per_km)``.
        incline, level_of_loading : float
            Where in the synthetic table the measured points are assumed to lie.

        Returns:
        --------
        Tuple[List[float], Dict[str, float]]
            The calibrated values (same order as ``values``) and summary statistics of
            the scaling factors (``n_points``, ``ratio_min``, ``ratio_mean``, ``ratio_max``).
            Only the flat-ground (incline == ``incline``) part of each value is scaled;
            the incline offset relative to that slice is added back unchanged.
        """
        import numpy as np
        from scipy import interpolate
        from scipy.spatial import QhullError  # type: ignore[import-untyped]

        points = np.asarray(data_points, dtype=float)
        synthetic = np.asarray(values, dtype=float)
        if points.ndim != 2 or points.shape[1] != 4 or len(synthetic) != len(points):
            raise ValueError("data_points must be N×4 and values must have length N")

        # Rebuild the regular grid of the synthetic table.
        scales = [np.unique(points[:, i]) for i in range(4)]
        grid = np.full([len(scale) for scale in scales], np.nan)
        index = tuple(np.searchsorted(scales[i], points[:, i]) for i in range(4))
        grid[index] = synthetic
        if np.isnan(grid).any():
            raise ValueError("The synthetic consumption LUT is not a complete regular grid")
        model = interpolate.RegularGridInterpolator(
            tuple(scales), grid, method="linear", bounds_error=False, fill_value=np.nan
        )

        # Evaluate the synthetic table at the measured points and form the ratios.
        m = np.asarray(measured, dtype=float)
        if m.ndim != 2 or m.shape[1] != 3 or len(m) == 0:
            raise ValueError("measured must be a non-empty list of (speed, t_amb, value)")
        query = np.column_stack(
            [
                np.full(len(m), incline),
                m[:, 1],
                np.full(len(m), level_of_loading),
                m[:, 0],
            ]
        )
        # Points outside the synthetic grid come back as NaN and are dropped: the
        # Ji2022 model is strongly non-linear in speed, so extrapolating it below the
        # grid's lowest speed would produce meaningless ratios.
        model_at_measured = np.asarray(model(query), dtype=float)
        usable = np.isfinite(model_at_measured) & (model_at_measured > 0) & (m[:, 2] > 0)
        if not usable.any():
            raise ValueError("No measured point could be matched to the synthetic LUT")
        ratio = m[usable, 2] / model_at_measured[usable]
        xy = np.column_stack([m[usable, 1], m[usable, 0]])  # (t_amb, speed)

        # Interpolate the ratio over (t_amb, speed): linear inside the measured
        # region, nearest neighbour outside of it (and if the points are degenerate).
        targets = np.column_stack([points[:, 1], points[:, 3]])
        try:
            linear = interpolate.LinearNDInterpolator(xy, ratio)
            factors = np.asarray(linear(targets), dtype=float)
        except (QhullError, ValueError):
            factors = np.full(len(targets), np.nan)
        outside = np.isnan(factors)
        if outside.any():
            nearest = interpolate.NearestNDInterpolator(xy, ratio)
            factors[outside] = np.asarray(nearest(targets[outside]), dtype=float)

        # Split every entry into its flat-ground part and its incline offset. The
        # slope term of the synthetic table is additive, so the offset is exactly the
        # slope contribution. Scale only the flat part; the measurements say nothing
        # about inclines.
        flat_index = int(np.argmin(np.abs(scales[0] - incline)))
        flat_values = grid[flat_index][index[1:]]
        slope_offset = synthetic - flat_values
        scaled = flat_values * factors + slope_offset
        stats = {
            "n_points": float(usable.sum()),
            "ratio_min": float(ratio.min()),
            "ratio_mean": float(ratio.mean()),
            "ratio_max": float(ratio.max()),
        }
        return [float(v) for v in scaled], stats

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Attach a calibrated Ji2022 consumption LUT to the selected vehicle types.

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
        vehicle_type_names: Optional[List[str]] = params.get(
            f"{self.__class__.__name__}.vehicle_type_names", None
        )

        scenarios = session.query(Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")
        scenario = scenarios[0]

        measured = load_measured_speed_temperature_lut(self.measured_lut_path)
        if not measured:
            raise ValueError(
                f"Measured consumption LUT {self.measured_lut_path} contains no values"
            )

        query = session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id)
        if vehicle_type_names is not None:
            query = query.filter(VehicleType.name_short.in_(vehicle_type_names))
        vehicle_types = query.all()
        if vehicle_type_names is not None:
            missing = set(vehicle_type_names) - {vt.name_short for vt in vehicle_types}
            if missing:
                raise ValueError(f"Vehicle types not found in scenario: {sorted(missing)}")
        if not vehicle_types:
            raise ValueError("No vehicle types found to calibrate")

        for vt in vehicle_types:
            if vt.empty_mass is None or vt.allowed_mass is None:
                raise ValueError(
                    f"Vehicle type {vt.name_short} needs empty_mass and allowed_mass "
                    "to generate a consumption LUT"
                )

            # Drop any previous LUT / vehicle class so the model's "consumption xor LUT"
            # rule holds afterwards.
            for vehicle_class in list(vt.vehicle_classes):
                if len(vehicle_class.vehicle_types) > 1:
                    vehicle_class.vehicle_types.remove(vt)
                    continue
                if vehicle_class.consumption_lut is not None:
                    session.delete(vehicle_class.consumption_lut)
                session.delete(vehicle_class)
            vt.consumption = None  # type: ignore[assignment]
            session.flush()

            vehicle_class = VehicleClass(
                scenario_id=vt.scenario_id,
                name=f"Consumption LUT for {vt.name_short}",
                vehicle_types=[vt],
            )
            session.add(vehicle_class)
            session.flush()

            lut = ConsumptionLut.from_vehicle_type(vt, vehicle_class)
            scaled_values, stats = self.calibrate_values(lut.data_points, lut.values, measured)
            lut.values = scaled_values
            lut.name = f"Ji2022 calibrated to {self.measured_lut_path.name} for {vt.name}"
            session.add(lut)

            self.logger.info(
                f"Calibrated consumption LUT for vehicle type {vt.name_short} against "
                f"{int(stats['n_points'])} measured points from {self.measured_lut_path.name}: "
                f"scaling factor min {stats['ratio_min']:.3f}, mean {stats['ratio_mean']:.3f}, "
                f"max {stats['ratio_max']:.3f}"
            )

        session.flush()
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


ALLOW_INCOMPLETE_PARAMS_DOC = """
Bool. When False (the default), the step verifies after configuration that every
VehicleType, BatteryType and ChargingPointType in the scenario actually carries the
parameters it needs, and raises ValueError listing those that do not. Set to True to
downgrade that to a warning -- useful when the scenario deliberately contains vehicle
types the JSON does not describe.
Default: False
Type: bool
"""


def _assert_vehicle_types_covered(
    step: Modifier,
    session: Session,
    scenario: Scenario,
    declared: Set[str],
    json_name: str,
    section: str,
    allow_incomplete: bool,
    battery_electric_only: bool = False,
) -> None:
    """
    Verify that every vehicle type in the scenario is described by the parameter JSON.

    ``init_tco_params`` and ``init_lca_params`` warn-and-skip when a JSON entry has no
    matching database row, and ``init_lca_params`` skips the *entire* scenario when a
    battery-electric vehicle type is absent from the overrides. Crucially, a skipped
    row is **not** left empty: ``tco_parameters`` and ``lca_parameters`` carry
    server-side defaults in eflips-model (placeholder values such as
    ``useful_life: 14``, ``procurement_cost: null``, ``average_consumption_kwh_per_km:
    1.5`` and a full set of emission factors). A vehicle type the JSON forgot is
    therefore costed at those placeholders and produces a plausible-looking result
    rather than an obviously broken one -- so checking for empty columns would not
    catch it. Coverage of the JSON against the scenario is checked instead.

    The reverse direction -- JSON entries with no matching vehicle type -- is only
    logged, because one parameter file is deliberately shared across scenarios that
    contain different subsets of the fleet (the BVG flow reuses a single ``tco.json``
    for the electric and diesel scenarios).

    Args:
        step: The modifier performing the check; supplies the bypass name and logger.
        session: SQLAlchemy session connected to the eflips-model database.
        scenario: The scenario being configured.
        declared: ``name_short`` values the JSON provides parameters for.
        json_name: File name of the JSON, for the error message.
        section: The JSON section the values came from, for the error message.
        allow_incomplete: When True, log a warning instead of raising.
        battery_electric_only: Only require coverage of battery-electric vehicle types.

    Raises:
        ValueError: If a vehicle type is not covered and *allow_incomplete* is False.
    """
    query = session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id)
    if battery_electric_only:
        query = query.filter(VehicleType.energy_source == EnergySource.BATTERY_ELECTRIC)
    vehicle_types = query.all()

    present = {vt.name_short for vt in vehicle_types if vt.name_short is not None}

    unused = declared - present
    if unused:
        step.logger.info(
            "%s: '%s' section '%s' has entries with no matching vehicle type in this "
            "scenario: %s. Ignoring them (a shared parameter file usually covers several "
            "scenarios).",
            step.__class__.__name__,
            json_name,
            section,
            sorted(unused),
        )

    uncovered = present - declared
    if not uncovered:
        return

    described = []
    for vehicle_type in vehicle_types:
        if vehicle_type.name_short not in uncovered:
            continue
        rotation_count = (
            session.query(Rotation)
            .filter(
                Rotation.scenario_id == scenario.id,
                Rotation.vehicle_type_id == vehicle_type.id,
            )
            .count()
        )
        in_use = f"used by {rotation_count} rotation(s)" if rotation_count else "unused"
        described.append(f"'{vehicle_type.name_short}' ({in_use})")

    bypass = f"{step.__class__.__name__}.allow_incomplete_parameters"
    message = (
        f"{step.__class__.__name__}: {json_name} section '{section}' does not cover "
        f"{len(uncovered)} vehicle type(s) present in the scenario: "
        f"{'; '.join(sorted(described))}. eflips-impact skips them, which leaves "
        "eflips-model's placeholder parameter defaults in place rather than an empty "
        "value -- the result would look plausible but would not reflect these vehicle "
        f"types. Add them to {json_name}, or set '{bypass}' to True to accept the "
        "defaults."
    )
    if allow_incomplete:
        step.logger.warning(message)
        return
    raise ValueError(message)


class CompleteFleet(Modifier):
    """
    Complete the fleet topology in the database based on a JSON file.

    The fleet JSON defines the fleet topology: which BatteryType / ChargingPointType
    rows exist and how they map to vehicle types and charging locations. Applied via
    :func:`eflips.impact.utils.complete_fleet` with ``delete_existing_data=True``,
    so any pre-existing topology rows are rebuilt to match the JSON (and re-written by
    the installed eflips-model, avoiding stale encodings).

    The JSON path is a constructor argument rather than a pipeline parameter so that
    it can be registered as an ``additional_files`` entry: the framework hashes those
    files into the cache key, so editing the JSON in place re-runs the step. Passing
    the path through ``params`` would only hash the *path*, and edits to the file
    would silently serve cached results.
    """

    def __init__(
        self,
        fleet_json: Union[str, Path],
        code_version: str = "v2.0.0",
        **kwargs: Any,
    ):
        """
        Args:
            fleet_json: Path to the eflips-impact fleet topology JSON (``battery_types``
                + ``charging_point_types``). Content-hashed into the cache key.
            code_version: Cache-invalidation version for this step.
        """
        self.fleet_json = Path(fleet_json)
        super().__init__(additional_files=[self.fleet_json], code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """The fleet JSON path is a constructor argument, not a pipeline parameter."""
        return {
            f"{cls.__name__}.allow_incomplete_parameters": ALLOW_INCOMPLETE_PARAMS_DOC,
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Complete the fleet topology in the database based on a JSON file.

        Args:
            session: SQLAlchemy session connected to the eflips-model database.
            params: Pipeline parameters, optionally including
                ``CompleteFleet.allow_incomplete_parameters``.

        Raises:
            FileNotFoundError: If the fleet JSON does not exist.
            ValueError: If there is not exactly one scenario, or if a battery-electric
                vehicle type is left without a BatteryType and
                ``allow_incomplete_parameters`` is not set.
        """
        if not self.fleet_json.is_file():
            raise FileNotFoundError(f"Fleet topology JSON not found: {self.fleet_json}")

        scenario = session.query(Scenario).one()

        # Rebuild the fleet topology from fleet.json (delete + recreate) so the
        # BatteryType / ChargingPointType rows match the JSON and are re-written
        # by the installed eflips-model.
        complete_fleet(
            scenario=scenario,
            json_path=self.fleet_json,
            delete_existing_data=True,
        )
        session.flush()

        # complete_fleet warns and returns without mutating on any validation
        # failure, which would leave the downstream configurators silently writing
        # nothing. Catch that here rather than three steps later.
        unassigned = [
            vt.name_short
            for vt in session.query(VehicleType)
            .filter(
                VehicleType.scenario_id == scenario.id,
                VehicleType.energy_source == EnergySource.BATTERY_ELECTRIC,
            )
            .all()
            if vt.battery_type_id is None
        ]
        if not unassigned:
            return

        bypass = f"{self.__class__.__name__}.allow_incomplete_parameters"
        message = (
            f"{self.__class__.__name__}: no BatteryType was assigned to battery-electric "
            f"vehicle type(s) {sorted(unassigned)} after applying "
            f"'{self.fleet_json.name}'. eflips-impact's complete_fleet warns and returns "
            "without mutating the database when its pre-flight validation fails, so the "
            "TCO/LCA configurators would then have nothing to write onto. Check that the "
            f"fleet JSON lists every vehicle type, or set '{bypass}' to True to proceed."
        )
        if params.get(bypass, False):
            self.logger.warning(message)
            return
        raise ValueError(message)


class TCOConfigurator(Modifier):
    """
    Modifier that writes the eflips-impact TCO parameters into the database, so that
    a downstream :class:`TCOAnalyzer` can compute the TCO.

    The TCO JSON defines the financial parameters (scenario, vehicle types, battery
    types, charging point types, charging infrastructure). Applied via
    :func:`eflips.impact.tco.init_tco_params`.

    This depends on the fleet topology (BatteryType / ChargingPointType rows) already
    being present, so run :class:`CompleteFleet` before it.

    The JSON path is a constructor argument rather than a pipeline parameter so that
    it can be registered as an ``additional_files`` entry and content-hashed into the
    cache key; see :class:`CompleteFleet`.

    Because it writes to the database, this is a Modifier: the changes are
    committed and chained into the next pipeline database.
    """

    def __init__(
        self,
        tco_json: Union[str, Path],
        code_version: str = "v2.0.0",
        **kwargs: Any,
    ):
        """
        Args:
            tco_json: Path to the eflips-impact TCO parameter JSON (scenario,
                vehicle_types, battery_types, charging_point_types,
                charging_infrastructure). Content-hashed into the cache key.
            code_version: Cache-invalidation version for this step.
        """
        self.tco_json = Path(tco_json)
        super().__init__(additional_files=[self.tco_json], code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.allow_incomplete_parameters": ALLOW_INCOMPLETE_PARAMS_DOC,
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Write the TCO parameters into the database.

        Args:
            session: SQLAlchemy session connected to the eflips-model database.
            params: Pipeline parameters, optionally including
                ``TCOConfigurator.allow_incomplete_parameters``.

        Raises:
            FileNotFoundError: If the TCO JSON does not exist.
            ValueError: If any entity is left without ``tco_parameters`` and
                ``allow_incomplete_parameters`` is not set.
        """
        if not self.tco_json.is_file():
            raise FileNotFoundError(f"TCO parameter JSON not found: {self.tco_json}")

        scenario = session.query(Scenario).one()

        # Write tco_parameters onto scenario / vehicle types / battery types /
        # charging point types / stations.
        init_tco_params(scenario=scenario, json_path=self.tco_json)
        session.flush()

        allow_incomplete = bool(
            params.get(f"{self.__class__.__name__}.allow_incomplete_parameters", False)
        )
        payload = json.loads(self.tco_json.read_text(encoding="utf-8"))
        _assert_vehicle_types_covered(
            step=self,
            session=session,
            scenario=scenario,
            declared={entry["name_short"] for entry in payload.get("vehicle_types", [])},
            json_name=self.tco_json.name,
            section="vehicle_types",
            allow_incomplete=allow_incomplete,
        )
        _assert_vehicle_types_covered(
            step=self,
            session=session,
            scenario=scenario,
            declared={entry["vehicle_name_short"] for entry in payload.get("battery_types", [])},
            json_name=self.tco_json.name,
            section="battery_types",
            allow_incomplete=allow_incomplete,
            battery_electric_only=True,
        )


class LCAConfigurator(Modifier):
    """
    Modifier that writes the eflips-impact LCA parameters into the database, so
    that a downstream :class:`LCAAnalyzer` can compute the life-cycle assessment.

    Two JSON files drive it:

    - the LCA JSON is an openLCA emission-factor export defining the impact
      vectors for the materials and processes used in the assessment.
    - the LCA overrides JSON defines the per-scenario overrides (per-vehicle-type
      parameters and charging-point-type infrastructure parameters).

    Both are applied via :func:`eflips.impact.lca.init_lca_params`, which writes
    ``VehicleTypeLCAParams``, ``BatteryTypeLCAParams`` and
    ``ChargingPointTypeLCAParams`` onto the corresponding entities.

    This depends on the fleet topology (BatteryType / ChargingPointType rows)
    already being present, so run :class:`CompleteFleet` before it.

    The JSON paths are constructor arguments rather than pipeline parameters so that
    they can be registered as ``additional_files`` entries and content-hashed into
    the cache key; see :class:`CompleteFleet`.

    Because it writes to the database, this is a Modifier: the changes are
    committed and chained into the next pipeline database.
    """

    def __init__(
        self,
        lca_json: Union[str, Path],
        lca_overrides_json: Union[str, Path],
        code_version: str = "v2.0.0",
        **kwargs: Any,
    ):
        """
        Args:
            lca_json: Path to the openLCA emission-factor JSON. Content-hashed into
                the cache key.
            lca_overrides_json: Path to the per-scenario LCA overrides JSON
                (``vehicle_type_overrides`` + charging-point infrastructure params).
                Content-hashed into the cache key.
            code_version: Cache-invalidation version for this step.
        """
        self.lca_json = Path(lca_json)
        self.lca_overrides_json = Path(lca_overrides_json)
        super().__init__(
            additional_files=[self.lca_json, self.lca_overrides_json],
            code_version=code_version,
            **kwargs,
        )
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.allow_incomplete_parameters": ALLOW_INCOMPLETE_PARAMS_DOC,
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Write the LCA parameters into the database.

        Args:
            session: SQLAlchemy session connected to the eflips-model database.
            params: Pipeline parameters, optionally including
                ``LCAConfigurator.allow_incomplete_parameters``.

        Raises:
            FileNotFoundError: If either JSON does not exist.
            ValueError: If any entity is left without ``lca_parameters`` and
                ``allow_incomplete_parameters`` is not set.
        """
        for json_path in (self.lca_json, self.lca_overrides_json):
            if not json_path.is_file():
                raise FileNotFoundError(f"LCA parameter JSON not found: {json_path}")

        scenario = session.query(Scenario).one()

        # Write lca_parameters onto vehicle types / battery types / charging point
        # types.
        init_lca_params(
            scenario=scenario,
            lca_json_path=self.lca_json,
            overrides_json_path=self.lca_overrides_json,
        )
        session.flush()

        # init_lca_params refuses to write *anything* when a battery-electric vehicle
        # type is missing from the overrides, so incomplete coverage silently leaves
        # the whole scenario on eflips-model's placeholder defaults.
        payload = json.loads(self.lca_overrides_json.read_text(encoding="utf-8"))
        _assert_vehicle_types_covered(
            step=self,
            session=session,
            scenario=scenario,
            declared={entry["name_short"] for entry in payload.get("vehicle_type_overrides", [])},
            json_name=self.lca_overrides_json.name,
            section="vehicle_type_overrides",
            allow_incomplete=bool(
                params.get(f"{self.__class__.__name__}.allow_incomplete_parameters", False)
            ),
            battery_electric_only=True,
        )


class CreateDieselVehicleTypes(Modifier):
    """
    Create a diesel counterpart for every electric vehicle type in the scenario.

    Ported from the bus-type-creation half of
    :class:`eflips.x.transition_plan.multi_stage_simulation.CreateHybridFleet`: a
    diesel :class:`~eflips.model.VehicleType` is created for every
    ``EnergySource.BATTERY_ELECTRIC`` vehicle type, using the
    ``"Diesel {name}"`` / ``"Diesel {name_short}"`` naming convention. Diesel
    vehicle types get near-zero consumption and ``energy_source=EnergySource.DIESEL``.

    Unlike ``CreateHybridFleet`` this step does **not** reassign any blocks
    (rotations); it only creates the vehicle-type records. Block reassignment is
    left to :class:`VehicleTypeBlockAssignment`, which recovers the diesel
    counterparts from the database via the same ``name_short`` naming convention --
    so no mapping has to be passed between the two steps.

    Idempotent: a diesel vehicle type whose ``name_short`` already exists in the
    scenario is reused, never duplicated.
    """

    DIESEL_PREFIX = "Diesel "
    DIESEL_CONSUMPTION = 0.0001  # Near-zero consumption for diesel simulation (kWh/km)

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """This modifier has no configurable parameters."""
        return {}

    def _create_diesel_vehicle_type(
        self, session: Session, electric_type: VehicleType, scenario: Scenario
    ) -> VehicleType:
        """Create a diesel version of an electric vehicle type."""
        diesel_type = VehicleType(
            scenario=scenario,
            name=f"{self.DIESEL_PREFIX}{electric_type.name}",
            name_short=f"{self.DIESEL_PREFIX}{electric_type.name_short}",
            battery_capacity=electric_type.battery_capacity,
            charging_curve=electric_type.charging_curve,
            opportunity_charging_capable=electric_type.opportunity_charging_capable,
            consumption=self.DIESEL_CONSUMPTION,
            battery_capacity_reserve=electric_type.battery_capacity_reserve,
            minimum_charging_power=electric_type.minimum_charging_power,
            charging_efficiency=electric_type.charging_efficiency,
            energy_source=EnergySource.DIESEL,
            empty_mass=electric_type.empty_mass,
            allowed_mass=electric_type.allowed_mass,
            # Dimensions are required by DepotGenerator's optimal-layout mode, and a
            # diesel counterpart occupies the same footprint as the electric vehicle
            # it stands in for.
            length=electric_type.length,
            width=electric_type.width,
            height=electric_type.height,
        )
        session.add(diesel_type)
        return diesel_type

    def _electric_vehicle_types(self, session: Session, scenario: Scenario) -> List[VehicleType]:
        """Return the scenario's battery-electric vehicle types.

        ``VehicleType.energy_source`` is NOT NULL and defaults to
        ``BATTERY_ELECTRIC``, so databases that never set it explicitly are already
        covered by this query -- no fallback for unset energy sources is needed.
        """
        return (
            session.query(VehicleType)
            .filter(
                VehicleType.scenario_id == scenario.id,
                VehicleType.energy_source == EnergySource.BATTERY_ELECTRIC,
            )
            .all()
        )

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Create a diesel counterpart for every electric vehicle type.

        Args:
            session: SQLAlchemy session connected to the eflips-model database.
            params: Pipeline parameters (unused by this step).
        """
        scenario = session.query(Scenario).one()
        electric_types = self._electric_vehicle_types(session, scenario)

        created = 0
        for electric_type in electric_types:
            diesel_short = f"{self.DIESEL_PREFIX}{electric_type.name_short}"
            existing = (
                session.query(VehicleType)
                .filter(
                    VehicleType.scenario_id == scenario.id,
                    VehicleType.name_short == diesel_short,
                )
                .one_or_none()
            )
            if existing is None:
                self._create_diesel_vehicle_type(session, electric_type, scenario)
                created += 1

        session.flush()
        self.logger.info(
            "CreateDieselVehicleTypes: %d electric vehicle type(s); created %d diesel "
            "counterpart(s) (others already existed).",
            len(electric_types),
            created,
        )


class VehicleTypeBlockAssignment(Modifier):
    """
    Reassign a set of blocks (rotations) to their diesel vehicle-type counterparts.

    The diesel vehicle types must already exist in the database (created by
    :class:`CreateDieselVehicleTypes`). Their correspondence to the electric vehicle
    types is recovered directly from the database via the ``"Diesel {name_short}"``
    naming convention -- no mapping is passed between steps.

    For each target rotation, the rotation's current (electric) vehicle type is
    looked up by ``name_short`` and the rotation is reassigned to the matching
    ``"Diesel {name_short}"`` vehicle type. Rotations already pointing at a diesel
    vehicle type are skipped; a rotation whose electric vehicle type has no diesel
    counterpart raises :class:`ValueError`.
    """

    DIESEL_PREFIX = CreateDieselVehicleTypes.DIESEL_PREFIX

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
            Dictionary describing the configurable parameter.
        """
        return {
            f"{cls.__name__}.block_ids": """
            Optional list of Rotation (block) ids to reassign to their diesel
            vehicle-type counterpart. When omitted or None, ALL rotations in the
            scenario are reassigned (the diesel-reference scenario). An explicit
            list reassigns only those rotations; an empty list is a no-op.
            Default: None (reassign all rotations)
            Type: Optional[List[int]]
            """,
        }

    def _diesel_types_by_source_short(
        self, session: Session, scenario: Scenario
    ) -> Dict[str, VehicleType]:
        """Map each electric VT's ``name_short`` to its diesel counterpart.

        Recovers the mapping created by :class:`CreateDieselVehicleTypes` from the
        database: every vehicle type whose ``name_short`` starts with the
        ``"Diesel "`` prefix is keyed by the stripped (electric) ``name_short``.
        """
        diesel_types: Dict[str, VehicleType] = {}
        for vt in session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id).all():
            if vt.name_short and vt.name_short.startswith(self.DIESEL_PREFIX):
                source_short = vt.name_short[len(self.DIESEL_PREFIX) :]
                diesel_types[source_short] = vt
        return diesel_types

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Reassign the requested rotations to their diesel vehicle types.

        Args:
            session: SQLAlchemy session connected to the eflips-model database.
            params: Pipeline parameters including the optional
                ``VehicleTypeBlockAssignment.block_ids`` list.

        Raises:
            ValueError: If no diesel vehicle types are present (run
                :class:`CreateDieselVehicleTypes` first), or if a target rotation's
                vehicle type has no diesel counterpart.
        """
        scenario = session.query(Scenario).one()

        diesel_types = self._diesel_types_by_source_short(session, scenario)
        if not diesel_types:
            raise ValueError(
                "No diesel vehicle types found. Run CreateDieselVehicleTypes before "
                "VehicleTypeBlockAssignment."
            )

        block_ids = params.get(f"{self.__class__.__name__}.block_ids", None)
        query = session.query(Rotation).filter(Rotation.scenario_id == scenario.id)
        if block_ids is None:
            rotations = query.all()
            self.logger.info("Reassigning all %d rotation(s) to diesel.", len(rotations))
        else:
            rotations = query.filter(Rotation.id.in_(block_ids)).all()
            self.logger.info(
                "Reassigning %d of %d requested rotation(s) to diesel.",
                len(rotations),
                len(block_ids),
            )

        reassigned = 0
        for rotation in rotations:
            current = rotation.vehicle_type
            if current.name_short and current.name_short.startswith(self.DIESEL_PREFIX):
                continue  # already diesel
            if current.name_short not in diesel_types:
                raise ValueError(
                    f"No diesel counterpart for vehicle type '{current.name_short}'. "
                    f"Available: {sorted(diesel_types.keys())}"
                )
            rotation.vehicle_type = diesel_types[current.name_short]
            reassigned += 1

        session.flush()
        self.logger.info("Reassigned %d rotation(s) to diesel vehicle types.", reassigned)
