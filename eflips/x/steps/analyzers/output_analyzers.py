"""
Analyzers for output evaluation using eflips-eval and eflips-tco.

These analyzers wrap the eflips.eval.output module's prepare/visualize functions
and the eflips-tco TCO calculator to make them usable within the eflips-x pipeline
framework.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import matplotlib.animation as animation
import pandas as pd
import plotly.graph_objs as go  # type: ignore
import sqlalchemy
from eflips.eval.output import prepare as eval_output_prepare
from eflips.eval.output import visualize as eval_output_visualize
from eflips.eval.output.prepare import depot_layout
from eflips.model import EnergySource, Scenario, VehicleType
from eflips.tco.data_queries import init_tco_parameters  # type: ignore[import-untyped]
from eflips.tco.tco_calculator import TCOCalculator  # type: ignore[import-untyped]
from eflips.tco.tco_parameter_config import (  # type: ignore[import-untyped]
    BatteryTypeTCOParameter,
    ChargingInfrastructureTCOParameter,
    ChargingPointTypeTCOParameter,
    ScenarioTCOParameter,
    VehicleTypeTCOParameter,
)
from sqlalchemy.orm import Session

from eflips.x.framework import Analyzer

logger = logging.getLogger(__name__)


class DepartureArrivalSocAnalyzer(Analyzer):
    """
    This Analyzer creates a dataframe with the SoC at departure and arrival for each trip.
    The columns are
    - event_id: the associated event id
    - rotation_id: the associated rotation id (for the trip the vehicle is going on or returning from)
    - rotation_name: the name of the rotation
    - vehicle_type_id: the vehicle type id
    - vehicle_type_name: the name of the vehicle type
    - vehicle_id: the vehicle id
    - vehicle_name: the name of the vehicle
    - time: the time at which this SoC was recorded (for departure, this is the departure time from the depot, for arrival, this is the arrival time at the depot)
    - soc: the state of charge at the given time
    - event_type: the type of event, either "Departure" or "Arrival"
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {}

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        This Analyzer creates a dataframe with the SoC at departure and arrival for each trip.
        The columns are
        - event_id: the associated event id
        - rotation_id: the associated rotation id (for the trip the vehicle is going on or returning from)
        - rotation_name: the name of the rotation
        - vehicle_type_id: the vehicle type id
        - vehicle_type_name: the name of the vehicle type
        - vehicle_id: the vehicle id
        - vehicle_name: the name of the vehicle
        - time: the time at which this SoC was recorded (for departure, this is the departure time from the depot, for arrival, this is the arrival time at the depot)
        - soc: the state of charge at the given time
        - event_type: the type of event, either "Departure" or "Arrival"

        Args:
            session: SQLAlchemy session connected to the eflips-model database
            params: Analysis parameters (not used)

        Returns:
            DataFrame with SoC information at depot departure/arrival
        """
        # Auto-detect scenario_id
        scenario = session.query(Scenario).one()
        scenario_id = scenario.id

        # Call eflips-eval prepare function
        result = eval_output_prepare.departure_arrival_soc(scenario_id, session)

        return result

    @staticmethod
    def visualize(prepared_data: pd.DataFrame) -> go.Figure:
        """
        Visualize departure and arrival SoC using plotly scatter plot.

        Args:
            prepared_data: Result from analyze() method

        Returns:
            Plotly figure object
        """
        return eval_output_visualize.departure_arrival_soc(prepared_data)


class DepotEventAnalyzer(Analyzer):
    """
    This function creates a dataframe with all the events at the depot for a given scenario.
    The columns are
    - time_start: the start time of the event in datetime format
    - time_end: the end time of the event in datetime format
    - vehicle_id: the unique vehicle identifier which could be used for querying the vehicle in the database
    - event_type: the type of event specified in the eflips model. See :class:`eflips.model.EventType` for more information
    - area_id: the unique area identifier which could be used for querying the area in the database
    - trip_id: the unique trip identifier which could be used for querying the trip in the database
    - station_id: the unique station identifier which could be used for querying the station in the database
    - location: the location of the event. This could be "depot", "trip" or "station"
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.vehicle_ids": f"""
Optional parameter to filter vehicles. Can be:
- A single vehicle ID (int)
- A list of vehicle IDs (List[int])
- None to include all vehicles (default)

Example: `params["{cls.__name__}.vehicle_ids"] = [1, 2, 3]`
            """.strip()
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        This function creates a dataframe with all the events at the depot for a given scenario.
        The columns are
        - time_start: the start time of the event in datetime format
        - time_end: the end time of the event in datetime format
        - vehicle_id: the unique vehicle identifier which could be used for querying the vehicle in the database
        - event_type: the type of event specified in the eflips model. See :class:`eflips.model.EventType` for more information
        - area_id: the unique area identifier which could be used for querying the area in the database
        - trip_id: the unique trip identifier which could be used for querying the trip in the database
        - station_id: the unique station identifier which could be used for querying the station in the database
        - location: the location of the event. This could be "depot", "trip" or "station"

        Returns:
            DataFrame with depot events
        """
        # Auto-detect scenario_id
        scenario = session.query(Scenario).one()
        scenario_id = scenario.id

        # Extract parameters
        vehicle_ids = params.get(f"{self.__class__.__name__}.vehicle_ids", None)

        # Call eflips-eval prepare function
        result = eval_output_prepare.depot_event(scenario_id, session, vehicle_ids)

        return result

    @staticmethod
    def visualize(
        prepared_data: pd.DataFrame,
        color_scheme: str = "event_type",
        timezone: ZoneInfo = ZoneInfo("Europe/Berlin"),
    ) -> go.Figure:
        """
        Visualize all events as a Gantt chart using plotly.

        Args:
            prepared_data: Result from analyze() method
            color_scheme: Color scheme to use ("event_type", "soc", "location", "area_type")
            timezone: Timezone for display (default: Europe/Berlin)

        Returns:
            Plotly figure object
        """
        return eval_output_visualize.depot_event(prepared_data, color_scheme, timezone)


class PowerAndOccupancyAnalyzer(Analyzer):
    """
    This function creates a dataframe containing a timeseries of the power and occupancy of the given area(s).
    The columns are:
    - time: the time at which the data was recorded
    - power: the summed power consumption of the area(s) at the given time
    - occupancy_charging: the summed occupancy (actively charing vehicles) of the area(s) at the given time
    - occupancy_total: the summed occupancy of the area(s) at the given time, including all ev
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.area_id": f"""
**Required** parameter specifying the area ID(s) to analyze. Can be:
- A single area ID (int)
- A list of area IDs (Iterable[int])

Example: `params["{cls.__name__}.area_id"] = [1, 2]`
            """.strip(),
            f"{cls.__name__}.temporal_resolution": f"""
Temporal resolution of the timeseries in seconds. Default is 60 seconds.

Example: `params["{cls.__name__}.temporal_resolution"] = 120`
            """.strip(),
            f"{cls.__name__}.station_id": f"""
Optional station ID(s) for opportunity charging events. Can be:
- A single station ID (int)
- A list of station IDs (Iterable[int])
- None to exclude stations (default)

Example: `params["{cls.__name__}.station_id"] = [1, 2]`
            """.strip(),
            f"{cls.__name__}.sim_start_time": f"""
Optional start time to filter the timeseries. If set, no data before this time is included.

Example: `params["{cls.__name__}.sim_start_time"] = datetime(...)`
            """.strip(),
            f"{cls.__name__}.sim_end_time": f"""
Optional end time to filter the timeseries. If set, no data after this time is included.

Example: `params["{cls.__name__}.sim_end_time"] = datetime(...)`
            """.strip(),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Analyze the database and return power and occupancy timeseries.

        Args:
            session: SQLAlchemy session connected to the eflips-model database
            params: Analysis parameters (must include either area_id or station_id, but not both)

        Returns:
            DataFrame with power and occupancy timeseries

        Raises:
            ValueError: If both area_id and station_id are set, or neither is set
        """

        # Extract parameters
        area_id = params.get(f"{self.__class__.__name__}.area_id")
        station_id = params.get(f"{self.__class__.__name__}.station_id", None)

        # XOR validation: exactly one must be set
        area_id_set = area_id is not None and (not isinstance(area_id, list) or len(area_id) > 0)
        station_id_set = station_id is not None

        if area_id_set and station_id_set:
            raise ValueError("Cannot set both area_id and station_id")
        if not area_id_set and not station_id_set:
            raise ValueError("Must set either area_id or station_id")

        # If only station_id is set, use empty list for area_id
        if station_id_set and not area_id_set:
            area_id = []

        # Extract optional parameters
        temporal_resolution = params.get(f"{self.__class__.__name__}.temporal_resolution", 60)
        sim_start_time = params.get(f"{self.__class__.__name__}.sim_start_time", None)
        sim_end_time = params.get(f"{self.__class__.__name__}.sim_end_time", None)

        # Call eflips-eval prepare function
        result = eval_output_prepare.power_and_occupancy(
            area_id,  # type: ignore [arg-type]
            session,
            temporal_resolution,
            station_id,
            sim_start_time,
            sim_end_time,
        )

        return result

    @staticmethod
    def visualize(
        prepared_data: pd.DataFrame, timezone: ZoneInfo = ZoneInfo("Europe/Berlin")
    ) -> go.Figure:
        """
        Visualize power and occupancy using plotly.

        Args:
            prepared_data: Result from analyze() method
            timezone: Timezone for display (default: Europe/Berlin)

        Returns:
            Plotly figure object with power and occupancy plots
        """
        return eval_output_visualize.power_and_occupancy(prepared_data, timezone)


class SpecificEnergyConsumptionAnalyzer(Analyzer):
    """
    Creates a dataframe of all the trip energy consumptions and distances for the given scenario.
    The dataframe contains the following columns:
    - trip_id: the unique identifier of the trip
    - route_id: the unique identifier of the route
    - route_name: the name of the route
    - distance: the distance of the route in km
    - energy_consumption: the energy consumption of the trip in kWh
    - vehicle_type_id: the unique identifier of the vehicle type
    - vehicle_type_name: the name of the vehicle type
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {}

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Creates a dataframe of all the trip energy consumptions and distances for the given scenario.
        The dataframe contains the following columns:
        - trip_id: the unique identifier of the trip
        - route_id: the unique identifier of the route
        - route_name: the name of the route
        - distance: the distance of the route in km
        - energy_consumption: the energy consumption of the trip in kWh
        - vehicle_type_id: the unique identifier of the vehicle type
        - vehicle_type_name: the name of the vehicle type

        Args:
            db: Path to the database file
            params: Analysis parameters (not used)

        Returns:
            DataFrame with trip energy consumption information
        """
        # Auto-detect scenario_id
        scenario = session.query(Scenario).one()
        scenario_id = scenario.id

        # Call eflips-eval prepare function
        result = eval_output_prepare.specific_energy_consumption(scenario_id, session)

        return result

    @staticmethod
    def visualize(prepared_data: pd.DataFrame) -> go.Figure:
        """
        Visualize specific energy consumption as a histogram.

        Args:
            prepared_data: Result from analyze() method

        Returns:
            Plotly figure object with histogram
        """
        return eval_output_visualize.specific_energy_consumption(prepared_data)


class VehicleSocAnalyzer(Analyzer):
    """
    This function takes in a vehicle id and returns a description what happened to the vehicle over time.
    The dataframe contains the following columns:
    - time: the time at which the SoC was recorded
    - soc: the state of charge at the given time

    Additionally, a dictionary for the different kinds of events is returned. For each kind of event, a list of Tuples
    with a description of the event, the start time and the end time is returned.

    The kinds of events are:
    - "rotation": A list of rotation names and the time the rotation started and ended
    - "charging": A list of the location of the charging and the time the charging started and ended
    - "trip": A list of the route name and the time the trip started and ended
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.vehicle_id": f"""
**Required** parameter specifying the vehicle ID to analyze.

Example: `params["{cls.__name__}.vehicle_id"] = 1`
            """.strip(),
            f"{cls.__name__}.timezone": f"""
Optional timezone for the visualization. Default is Europe/Berlin.

Example: `params["{cls.__name__}.timezone"] = ZoneInfo("UTC")`
            """.strip(),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, datetime, datetime]]]]:
        """
        This function takes in a vehicle id and returns a description what happened to the vehicle over time.
        The dataframe contains the following columns:
        - time: the time at which the SoC was recorded
        - soc: the state of charge at the given time

        Additionally, a dictionary for the different kinds of events is returned. For each kind of event, a list of Tuples
        with a description of the event, the start time and the end time is returned.

        The kinds of events are:
        - "rotation": A list of rotation names and the time the rotation started and ended
        - "charging": A list of the location of the charging and the time the charging started and ended
        - "trip": A list of the route name and the time the trip started and ended

        Args:
            session: SQLAlchemy session connected to the eflips-model database
            params: Analysis parameters (must include vehicle_id)

        Returns:
            Tuple of (DataFrame with SoC timeseries, Dict with event descriptions)

        Raises:
            ValueError: If vehicle_id parameter is not provided
        """

        # Extract required parameter
        vehicle_id = params.get(f"{self.__class__.__name__}.vehicle_id")
        if vehicle_id is None:
            raise ValueError(
                f"Required parameter '{self.__class__.__name__}.vehicle_id' not provided"
            )

        # Extract optional parameters
        timezone = params.get(f"{self.__class__.__name__}.timezone", ZoneInfo("Europe/Berlin"))

        # Call eflips-eval prepare function
        result = eval_output_prepare.vehicle_soc(vehicle_id, session, timezone)

        return result

    @staticmethod
    def visualize(
        prepared_data: pd.DataFrame,
        descriptions: Dict[str, List[Tuple[str, datetime, datetime]]],
        timezone: ZoneInfo = ZoneInfo("Europe/Berlin"),
    ) -> go.Figure:
        """
        Visualize vehicle SoC over time with event annotations.

        Args:
            prepared_data: Result DataFrame from analyze() method
            descriptions: Event descriptions Dict from analyze() method
            timezone: Timezone for display (default: Europe/Berlin)

        Returns:
            Plotly figure object
        """
        return eval_output_visualize.vehicle_soc(prepared_data, descriptions, timezone)


class DepotActivityAnalyzer(Analyzer):
    """
    Analyzer for depot activity over time.

    Returns a dictionary of the occupancy of each slot in the depot
    during a specified time range.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.depot_id": f"""
**Required** parameter specifying the depot ID to analyze.

Example: `params["{cls.__name__}.depot_id"] = 1`
            """.strip(),
            f"{cls.__name__}.animation_start": f"""
**Required** parameter specifying the start time of the animation range.

Example: `params["{cls.__name__}.animation_start"] = datetime(...)`
            """.strip(),
            f"{cls.__name__}.animation_end": f"""
**Required** parameter specifying the end time of the animation range.

Example: `params["{cls.__name__}.animation_end"] = datetime(...)`
            """.strip(),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Analyze the database and return depot activity data.

        Args:
            db: Path to the database file
            params: Analysis parameters (must include depot_id, animation_start, animation_end)

        Returns:
            Dictionary of depot slot occupancy

        Raises:
            ValueError: If required parameters are not provided
        """

        # Extract required parameters
        depot_id = params.get(f"{self.__class__.__name__}.depot_id")
        if depot_id is None:
            raise ValueError(
                f"Required parameter '{self.__class__.__name__}.depot_id' not provided"
            )

        animation_start = params.get(f"{self.__class__.__name__}.animation_start")
        if animation_start is None:
            raise ValueError(
                f"Required parameter '{self.__class__.__name__}.animation_start' not provided"
            )

        animation_end = params.get(f"{self.__class__.__name__}.animation_end")
        if animation_end is None:
            raise ValueError(
                f"Required parameter '{self.__class__.__name__}.animation_end' not provided"
            )

        animation_range = (animation_start, animation_end)

        # Call eflips-eval prepare function
        result = eval_output_prepare.depot_activity(depot_id, session, animation_range)

        return result

    @staticmethod
    def visualize(
        area_occupancy: Dict[Tuple[int, int], List[Tuple[int, int]]],
        animation_range: Tuple[datetime, datetime],
        depot_id: int,
        session: Session,
        time_resolution: int = 120,
    ) -> animation.FuncAnimation:
        """
        Visualize depot activity as an animation using matplotlib.

        Args:
            area_blocks: Depot layout (from DepotLayoutAnalyzer)
            area_occupancy: Result from analyze() method
            animation_range: Tuple of (start_time, end_time)
            time_resolution: Time interval between frames in seconds (default: 120)

        Returns:
            Matplotlib animation object
        """
        area_blocks = depot_layout(depot_id, session)
        return eval_output_visualize.depot_activity_animation(
            area_blocks, area_occupancy, animation_range, time_resolution
        )


class InteractiveMapAnalyzer(Analyzer):
    """
    Analyzer for creating interactive map visualization.

    Prepares data for an interactive folium map showing depots, routes,
    charging stations, and termini. Supports multiple scenarios.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.scenario_ids": f"""
Optional parameter specifying the scenario ID(s) to include on the map. Can be:
- A single scenario ID (int)
- A list of scenario IDs (List[int])
- None to auto-detect the scenario (default)

Example: `params["{cls.__name__}.scenario_ids"] = [1, 2]`
            """.strip(),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare all data needed for the interactive map visualization.

        This function extracts and organizes data for creating an interactive folium map
        showing depots, routes, charging stations, and termini. Supports multiple scenarios.


        :parameter session: SQLAlchemy session connected to the eflips-model database
        :parameter params: Analysis parameters (scenario_ids is optional)

        :return: Dictionary with all map data organized by scenario
        """
        # Extract optional parameter
        scenario_ids = params.get(f"{self.__class__.__name__}.scenario_ids")

        # Auto-detect scenario_id if not provided
        if scenario_ids is None:
            scenario = session.query(Scenario).one()
            scenario_ids = scenario.id

        # Call eflips-eval prepare function
        result = eval_output_prepare.interactive_map_data(scenario_ids, session)

        return result

    @staticmethod
    def visualize(
        prepared_data: Dict[str, Any],
        station_plot_dir: str | None = None,
        depot_plot_dir: str | None = None,
    ) -> "folium.Map":  # type: ignore
        """
        Create an interactive folium map from prepared data.

        This function creates a map with depots, routes, and termini.
        Supports multiple scenarios with toggleable layers.

        Args:
            prepared_data: Output from analyze() method
            station_plot_dir: Optional path to directory containing station plots (named station_{id}.html)
            depot_plot_dir: Optional path to directory containing depot plots (named depot_{id}.html)

        Returns:
            folium.Map object ready to be saved

        Example usage:
            >>> prepared = analyzer.analyze(session, params)
            >>> m = InteractiveMapAnalyzer.visualize(prepared, depot_plot_dir="plots/depots")
            >>> m.save("map.html")
        """
        return eval_output_visualize.interactive_map(
            prepared_data, station_plot_dir, depot_plot_dir
        )


class TCOAnalyzer(Analyzer):
    """
    Analyzer for calculating Total Cost of Ownership (TCO) using the eflips-tco package.

    Computes TCO broken down by cost category (vehicle, battery, infrastructure,
    energy, maintenance, staff, other) normalized to EUR/km. Uses constant energy
    consumption mode, where the per-vehicle-type energy consumption factor is an
    explicit parameter rather than being derived from simulation events.

    The energy_consumption_factor parameter specifies per-vehicle-type average energy
    consumption (kWh/km) for cost estimation. This is typically lower than simulated
    worst-case consumption, reflecting average-day operations rather than the
    planning-conservative values used in simulation.
    """

    COST_CATEGORIES = [
        "VEHICLE",
        "BATTERY",
        "INFRASTRUCTURE",
        "ENERGY",
        "MAINTENANCE",
        "STAFF",
        "OTHER",
    ]
    CATEGORY_NAMES = {
        "VEHICLE": "Vehicle",
        "BATTERY": "Battery",
        "INFRASTRUCTURE": "Infrastructure",
        "ENERGY": "Energy",
        "MAINTENANCE": "Maintenance",
        "STAFF": "Staff",
        "OTHER": "Other",
    }

    def __init__(self, code_version: str = "v1.0.2", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.scenario_name": (
                "Display name for this scenario in output. "
                "Falls back to Scenario.name from the database."
            ),
            f"{cls.__name__}.vehicle_type_tco_params": (
                "Dict mapping vehicle type name_short to TCO parameters. "
                'Each value: {"useful_life": int, "procurement_cost": float, '
                '"cost_escalation": float}. Required.'
            ),
            f"{cls.__name__}.battery_type_tco_params": (
                "Dict mapping vehicle type name_short to battery TCO parameters. "
                'Each value: {"name": str, "procurement_cost": float (EUR per kWh), '
                '"useful_life": int, "cost_escalation": float, '
                '"specific_mass": float (kWh/kg, required for new battery types), '
                '"chemistry": str (required for new battery types)}. Required.'
            ),
            f"{cls.__name__}.charging_point_type_params": (
                "List of dicts for charging point types. Each: "
                '{"type": "depot"|"opportunity", "name": str, "procurement_cost": float, '
                '"useful_life": int, "cost_escalation": float}. Required.'
            ),
            f"{cls.__name__}.charging_infrastructure_params": (
                "List of dicts for charging infrastructure. Each: "
                '{"type": "depot"|"station", "name": str, "procurement_cost": float, '
                '"useful_life": int, "cost_escalation": float}. Required.'
            ),
            f"{cls.__name__}.energy_consumption_factor": (
                "Dict mapping vehicle type name_short to average energy consumption "
                "in kWh/km. This should be lower than the simulated worst-case "
                "consumption, reflecting average-day operations for cost estimation. "
                "Required."
            ),
            f"{cls.__name__}.project_duration": "Project duration in years. Default: 20",
            f"{cls.__name__}.interest_rate": "Interest rate. Default: 0.04",
            f"{cls.__name__}.inflation_rate": "Inflation/discount rate. Default: 0.02",
            f"{cls.__name__}.staff_cost": "Cost per driver hour in EUR. Default: 25.0",
            f"{cls.__name__}.fuel_cost": "Electricity cost per kWh in EUR. Default: 0.1794",
            f"{cls.__name__}.diesel_fuel_cost": "Diesel cost per litre in EUR. Default: 1.50",
            f"{cls.__name__}.maint_cost": "Electric vehicle maintenance cost per km in EUR. Default: 0.35",
            f"{cls.__name__}.diesel_maint_cost": "Diesel vehicle maintenance cost per km in EUR. Default: 0.45",
            f"{cls.__name__}.maint_infr_cost": (
                "Infrastructure maintenance cost per year per charging slot in EUR. "
                "Default: 1000"
            ),
            f"{cls.__name__}.taxes": ("Tax cost per vehicle per year in EUR. Default: 278"),
            f"{cls.__name__}.insurance": (
                "Insurance cost per vehicle per year in EUR. Default: 9693"
            ),
            f"{cls.__name__}.pef_general": "General price escalation factor. Default: 0.02",
            f"{cls.__name__}.pef_wages": "Wage price escalation factor. Default: 0.025",
            f"{cls.__name__}.pef_fuel": (
                "Fuel/electricity price escalation factor. Default: 0.038"
            ),
            f"{cls.__name__}.pef_insurance": ("Insurance price escalation factor. Default: 0.02"),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calculate TCO for the scenario in the database.

        Initializes TCO parameters in the (temporary) database, then runs the
        TCO calculator with constant energy consumption mode.

        Args:
            session: SQLAlchemy session connected to the eflips-model database
            params: Pipeline parameters including TCO configuration

        Returns:
            Single-row DataFrame with columns: scenario_name + one column per
            cost category (VEHICLE, BATTERY, INFRASTRUCTURE, ENERGY, MAINTENANCE,
            STAFF, OTHER). Values are in EUR/km.
        """
        scenario = session.query(Scenario).one()

        scenario_name: str = params.get(f"{self.__class__.__name__}.scenario_name", "")
        if not scenario_name:
            scenario_name = scenario.name or "Unknown"

        # Get required parameters
        vt_tco_params: Dict[str, Dict[str, Any]] = params.get(
            f"{self.__class__.__name__}.vehicle_type_tco_params", {}
        )
        bt_tco_params: Dict[str, Dict[str, Any]] = params.get(
            f"{self.__class__.__name__}.battery_type_tco_params", {}
        )
        cp_type_params: List[Dict[str, Any]] = params.get(
            f"{self.__class__.__name__}.charging_point_type_params", []
        )
        infra_params: List[Dict[str, Any]] = params.get(
            f"{self.__class__.__name__}.charging_infrastructure_params", []
        )
        energy_factors: Dict[str, float] = params.get(
            f"{self.__class__.__name__}.energy_consumption_factor", {}
        )

        if not all([vt_tco_params, bt_tco_params, cp_type_params, infra_params, energy_factors]):
            raise ValueError(
                f"{self.__class__.__name__} requires vehicle_type_tco_params, "
                "battery_type_tco_params, charging_point_type_params, "
                "charging_infrastructure_params, and energy_consumption_factor"
            )

        # Resolve vehicle type name_shorts to determine energy source
        vehicle_types = (
            session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id).all()
        )
        vt_by_name = {vt.name_short: vt for vt in vehicle_types}

        # Build VehicleTypeTCOParameter list (includes energy consumption per vehicle type)
        vehicle_type_params: List[VehicleTypeTCOParameter] = []
        for name_short, tco_info in vt_tco_params.items():
            vt = vt_by_name.get(name_short)
            if vt is None:
                logger.warning(f"Vehicle type '{name_short}' not found in database, skipping")
                continue
            energy_factor = energy_factors.get(name_short, 0.0)
            if vt.energy_source == EnergySource.DIESEL:
                vt_param = VehicleTypeTCOParameter(
                    name_short=name_short,
                    useful_life=tco_info["useful_life"],
                    procurement_cost=tco_info["procurement_cost"],
                    cost_escalation=tco_info["cost_escalation"],
                    average_diesel_consumption=energy_factor,
                )
            else:
                vt_param = VehicleTypeTCOParameter(
                    name_short=name_short,
                    useful_life=tco_info["useful_life"],
                    procurement_cost=tco_info["procurement_cost"],
                    cost_escalation=tco_info["cost_escalation"],
                    average_electricity_consumption=energy_factor,
                )
            vehicle_type_params.append(vt_param)

        # Build BatteryTypeTCOParameter list
        battery_type_params: List[BatteryTypeTCOParameter] = []
        for name_short, bt_info in bt_tco_params.items():
            battery_type_params.append(
                BatteryTypeTCOParameter(
                    vehicle_name_short=name_short,
                    procurement_cost=bt_info["procurement_cost"],
                    useful_life=bt_info["useful_life"],
                    cost_escalation=bt_info["cost_escalation"],
                    specific_mass=bt_info.get("specific_mass"),
                    chemistry=bt_info.get("chemistry"),
                )
            )

        # Build ChargingPointTypeTCOParameter list
        charging_point_type_params: List[ChargingPointTypeTCOParameter] = [
            ChargingPointTypeTCOParameter(
                type=cp["type"],
                name=cp.get("name"),
                procurement_cost=cp["procurement_cost"],
                useful_life=cp["useful_life"],
                cost_escalation=cp["cost_escalation"],
            )
            for cp in cp_type_params
        ]

        # Build ChargingInfrastructureTCOParameter list
        charging_infra_params: List[ChargingInfrastructureTCOParameter] = [
            ChargingInfrastructureTCOParameter(
                type=infra["type"],
                procurement_cost=infra["procurement_cost"],
                useful_life=infra["useful_life"],
                cost_escalation=infra["cost_escalation"],
            )
            for infra in infra_params
        ]

        # Build ScenarioTCOParameter with defaults
        prefix = f"{self.__class__.__name__}"
        electricity_cost: float = params.get(f"{prefix}.fuel_cost", 0.1794)
        diesel_cost: float = params.get(f"{prefix}.diesel_fuel_cost", 1.50)
        electricity_maint: float = params.get(f"{prefix}.maint_cost", 0.35)
        diesel_maint: float = params.get(f"{prefix}.diesel_maint_cost", 0.45)
        pef_fuel: float = params.get(f"{prefix}.pef_fuel", 0.038)
        scenario_params = ScenarioTCOParameter(
            project_duration=params.get(f"{prefix}.project_duration", 20),
            interest_rate=params.get(f"{prefix}.interest_rate", 0.04),
            inflation_rate=params.get(f"{prefix}.inflation_rate", 0.02),
            staff_cost=params.get(f"{prefix}.staff_cost", 25.0),
            fuel_cost={"electricity": electricity_cost, "diesel": diesel_cost},
            vehicle_maint_cost={"electricity": electricity_maint, "diesel": diesel_maint},
            infra_maint_cost=params.get(f"{prefix}.maint_infr_cost", 1000),
            cost_escalation_rate={
                "general": params.get(f"{prefix}.pef_general", 0.02),
                "staff": params.get(f"{prefix}.pef_wages", 0.025),
                "electricity": pef_fuel,
                "diesel": pef_fuel,
                "insurance": params.get(f"{prefix}.pef_insurance", 0.02),
            },
            insurance=params.get(f"{prefix}.insurance", 9693),
            taxes=params.get(f"{prefix}.taxes", 278),
        )

        # Initialize TCO parameters in the (temporary) database
        init_tco_parameters(
            scenario=scenario,
            scenario_params=scenario_params,
            vehicle_type_params=vehicle_type_params,
            battery_type_params=battery_type_params,
            charging_point_type_params=charging_point_type_params,
            charging_infra_params=charging_infra_params,
        )

        # Calculate TCO
        tco_calculator = TCOCalculator(
            scenario=scenario,
            energy_consumption_mode="constant",
        )
        tco_result = tco_calculator.calculate()

        # Get results and merge CHARGING_POINT into INFRASTRUCTURE (backwards compatibility)
        result: Dict[str, Any] = dict(tco_result.tco_by_type)
        result["INFRASTRUCTURE"] = result.get("INFRASTRUCTURE", 0.0) + result.get(
            "CHARGING_POINT", 0.0
        )
        result.pop("CHARGING_POINT", None)

        # Ensure all standard categories are present
        for cat in self.COST_CATEGORIES:
            if cat not in result:
                result[cat] = 0.0

        result["scenario_name"] = scenario_name

        return pd.DataFrame([result])

    @staticmethod
    def visualize(prepared_data: pd.DataFrame) -> go.Figure:
        """
        Create a plotly stacked bar chart of TCO by cost category.

        Args:
            prepared_data: DataFrame from analyze() or merged results from
                          merge_tco_results(). Must have 'scenario_name' column
                          and cost category columns.

        Returns:
            Plotly figure object
        """
        categories = [c for c in TCOAnalyzer.COST_CATEGORIES if c in prepared_data.columns]

        fig = go.Figure()
        for cat in categories:
            fig.add_trace(
                go.Bar(
                    name=TCOAnalyzer.CATEGORY_NAMES.get(cat, cat),
                    x=prepared_data["scenario_name"],
                    y=prepared_data[cat],
                    text=[f"{v:.2f}" for v in prepared_data[cat]],
                    textposition="inside",
                )
            )

        # Add total annotations on top of each bar
        totals = prepared_data[categories].sum(axis=1)
        for scenario, total in zip(prepared_data["scenario_name"], totals):
            fig.add_annotation(
                x=scenario,
                y=total,
                text=f"<b>{total:.2f}</b>",
                showarrow=False,
                yshift=10,
                font=dict(size=12),
            )

        fig.update_layout(
            barmode="stack",
            yaxis_title="Total Cost of Ownership [EUR/km]",
            xaxis_title="",
            showlegend=True,
            legend_title="",
        )

        return fig


def merge_tco_results(results: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple single-scenario TCO result DataFrames.

    Args:
        results: List of single-row DataFrames from TCOAnalyzer.analyze()

    Returns:
        Combined DataFrame with all scenarios
    """
    return pd.concat(results, ignore_index=True)
