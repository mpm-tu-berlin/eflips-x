"""
Analyzers for output evaluation using eflips-eval and eflips-impact.

These analyzers wrap the eflips.eval.output module's prepare/visualize functions
and the eflips-impact TCO calculator to make them usable within the eflips-x
pipeline framework.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import matplotlib.animation as animation
import pandas as pd
import plotly.graph_objs as go  # type: ignore
import sqlalchemy
from eflips.eval.output import prepare as eval_output_prepare
from eflips.eval.output import visualize as eval_output_visualize
from eflips.eval.output.prepare import depot_layout
from eflips.impact.tco import calculate_tco, init_tco_params  # type: ignore[import-untyped]
from eflips.impact.utils import complete_fleet  # type: ignore[import-untyped]
from eflips.model import (
    Event,
    EventType,
    Route,
    Scenario,
    Trip,
    Vehicle,
    VehicleType,
)
from sqlalchemy.orm import Session

from eflips.x.framework import Analyzer, Modifier, ScenarioDisplayConfig

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


class TCOConfigurator(Modifier):
    """
    Modifier that writes the eflips-impact fleet topology and TCO parameters into
    the database, so that a downstream :class:`TCOAnalyzer` can compute the TCO.

    Two JSON files drive it, with paths passed via the pipeline parameters
    ``TCOConfigurator.fleet_json`` and ``TCOConfigurator.tco_json`` (set by the
    calling flow, e.g. pointing at ``data/impact/fleet.json`` and
    ``data/impact/tco.json``):

    - the fleet JSON defines the fleet topology: which BatteryType /
      ChargingPointType rows exist and how they map to vehicle types and charging
      locations. Applied via :func:`eflips.impact.utils.complete_fleet` with
      ``delete_existing_data=True``, so any pre-existing topology rows are rebuilt
      to match the JSON (and re-written by the installed eflips-model, avoiding
      stale encodings).
    - the TCO JSON defines the financial parameters (scenario, vehicle types,
      battery types, charging point types, charging infrastructure). Applied via
      :func:`eflips.impact.tco.init_tco_params`.

    Because it writes to the database, this is a Modifier: the changes are
    committed and chained into the next pipeline database.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.fleet_json": (
                "Path to the eflips-impact fleet topology JSON (battery_types + "
                "charging_point_types), applied via complete_fleet. Required."
            ),
            f"{cls.__name__}.tco_json": (
                "Path to the eflips-impact TCO parameter JSON (scenario, vehicle_types, "
                "battery_types, charging_point_types, charging_infrastructure), applied "
                "via init_tco_params. Required."
            ),
        }

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """
        Write the fleet topology and TCO parameters into the database.

        Args:
            session: SQLAlchemy session connected to the eflips-model database.
            params: Pipeline parameters including ``TCOConfigurator.fleet_json``
                and ``TCOConfigurator.tco_json``.
        """
        scenario = session.query(Scenario).one()

        fleet_json_param = params.get(f"{self.__class__.__name__}.fleet_json")
        tco_json_param = params.get(f"{self.__class__.__name__}.tco_json")
        if not fleet_json_param or not tco_json_param:
            raise ValueError(
                f"{self.__class__.__name__} requires the "
                f"'{self.__class__.__name__}.fleet_json' and "
                f"'{self.__class__.__name__}.tco_json' parameters (paths to the "
                "eflips-impact fleet and TCO parameter JSON files)."
            )

        # Rebuild the fleet topology from fleet.json (delete + recreate) so the
        # BatteryType / ChargingPointType rows match the JSON and are re-written
        # by the installed eflips-model.
        complete_fleet(
            scenario=scenario,
            json_path=Path(fleet_json_param),
            delete_existing_data=True,
        )

        # Write tco_parameters onto scenario / vehicle types / battery types /
        # charging point types / stations.
        init_tco_params(scenario=scenario, json_path=Path(tco_json_param))


class TCOAnalyzer(Analyzer):
    """
    Analyzer that calculates Total Cost of Ownership (TCO) using eflips-impact.

    Computes TCO broken down by cost category (vehicle, battery, infrastructure,
    energy, maintenance, staff, other) normalized to EUR/revenue-km, using the
    constant energy-consumption mode.

    This analyzer only reads the database: the fleet topology and
    ``tco_parameters`` it depends on must already be present, written by
    :class:`TCOConfigurator`. Run ``TCOConfigurator`` before this analyzer.
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

    def __init__(self, code_version: str = "v3.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.scenario_name": (
                "Display name for this scenario in output. "
                "Falls back to Scenario.name from the database."
            ),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calculate TCO for the scenario in the database.

        Runs the eflips-impact TCO calculator (constant energy-consumption mode)
        against the already-configured database. The fleet topology and
        ``tco_parameters`` must have been written by :class:`TCOConfigurator`
        beforehand.

        Args:
            session: SQLAlchemy session connected to the eflips-model database
            params: Pipeline parameters including ``TCOAnalyzer.scenario_name``

        Returns:
            Single-row DataFrame with columns: scenario_name + one column per
            cost category (VEHICLE, BATTERY, INFRASTRUCTURE, ENERGY, MAINTENANCE,
            STAFF, OTHER). Values are in EUR/revenue-km.
        """
        scenario = session.query(Scenario).one()

        scenario_name: str = params.get(f"{self.__class__.__name__}.scenario_name", "")
        if not scenario_name:
            scenario_name = scenario.name or "Unknown"

        # Calculate TCO (per revenue-km, by cost category).
        tco_result = calculate_tco(scenario=scenario)
        per_revenue_km = tco_result.tco_by_type_per_revenue_km

        result: Dict[str, Any] = {cat.name: cost for cat, cost in per_revenue_km.items()}

        # Ensure all standard categories are present
        for cat_name in self.COST_CATEGORIES:
            result.setdefault(cat_name, 0.0)

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


# ============================================================================
# Energy Consumption and Battery-Electric Range
# ============================================================================


class EnergyConsumptionByVehicleTypeAnalyzer(Analyzer):
    """
    Compute average energy consumption and battery-electric range per vehicle type and scenario.

    For each driving event in the simulation results, calculates energy consumed from
    ``(soc_start - soc_end) * battery_capacity``, then aggregates per vehicle type to give
    average consumption in kWh/km and a derived battery-electric range.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.scenario_name": "Label for this scenario in the output table.",
        }

    def analyze(self, session: Session, params: Dict[str, Any]) -> pd.DataFrame:
        scenario_name = params.get(f"{self.__class__.__name__}.scenario_name", "Unknown")

        rows = (
            session.query(Event, VehicleType, Route)
            .join(Vehicle, Event.vehicle_id == Vehicle.id)
            .join(VehicleType, Vehicle.vehicle_type_id == VehicleType.id)
            .join(Trip, Event.trip_id == Trip.id)
            .join(Route, Trip.route_id == Route.id)
            .filter(Event.event_type == EventType.DRIVING)
            .all()
        )

        records = []
        for event, vtype, route in rows:
            energy_kwh = (event.soc_start - event.soc_end) * vtype.battery_capacity
            distance_km = route.distance / 1000
            records.append(
                {
                    "vehicle_type_id": vtype.id,
                    "vehicle_type": vtype.name,
                    "vehicle_type_short": vtype.name_short,
                    "battery_capacity_kwh": vtype.battery_capacity,
                    "battery_capacity_reserve_kwh": vtype.battery_capacity_reserve,
                    "energy_kwh": energy_kwh,
                    "distance_km": distance_km,
                }
            )

        df = pd.DataFrame(records)

        result_rows = []
        for vtype_id, group in df.groupby("vehicle_type_id"):
            total_energy = group["energy_kwh"].sum()
            total_distance = group["distance_km"].sum()
            avg_consumption = total_energy / total_distance if total_distance > 0 else float("nan")
            battery_cap = group["battery_capacity_kwh"].iloc[0]
            battery_reserve = group["battery_capacity_reserve_kwh"].iloc[0]
            usable_battery = battery_cap - battery_reserve
            range_km = usable_battery / avg_consumption if avg_consumption > 0 else float("nan")

            result_rows.append(
                {
                    "scenario_name": scenario_name,
                    "vehicle_type": group["vehicle_type"].iloc[0],
                    "vehicle_type_short": group["vehicle_type_short"].iloc[0],
                    "avg_consumption_kwh_per_km": round(avg_consumption, 2),
                    "battery_capacity_kwh": battery_cap,
                    "usable_battery_kwh": usable_battery,
                    "battery_electric_range_km": round(range_km, 0),
                }
            )

        result_rows.sort(key=lambda r: r["vehicle_type_short"])
        return pd.DataFrame(result_rows)


def merge_energy_consumption_results(
    dfs: List[pd.DataFrame],
    config: "ScenarioDisplayConfig | None" = None,
) -> pd.DataFrame:
    """
    Merge single-scenario energy consumption DataFrames into one table.

    Args:
        dfs: List of DataFrames from ``EnergyConsumptionByVehicleTypeAnalyzer.analyze()``.
        config: Optional scenario display configuration for ordering.
                Falls back to hardcoded ``["OU", "DEP", "TERM"]`` order if not provided.

    Returns:
        Combined DataFrame sorted by scenario order then vehicle type.
    """
    merged = pd.concat(dfs, ignore_index=True)

    if config is not None:
        merged["_scenario_sort"] = merged["scenario_name"].map(lambda s: config.sort_key(s))
    else:
        scenario_order = {name: i for i, name in enumerate(["OU", "DEP", "TERM"])}
        merged["_scenario_sort"] = merged["scenario_name"].map(scenario_order).fillna(99)
    merged = (
        merged.sort_values(["_scenario_sort", "vehicle_type_short"])
        .drop(columns="_scenario_sort")
        .reset_index(drop=True)
    )

    return merged
