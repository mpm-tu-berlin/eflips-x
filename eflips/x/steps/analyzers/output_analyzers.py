"""
Analyzers for output evaluation using eflips-eval.

These analyzers wrap the eflips.eval.output module's prepare/visualize functions
to make them usable within the eflips-x pipeline framework.
"""

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
from eflips.model import Scenario
from sqlalchemy.orm import Session

from eflips.x.framework import Analyzer


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

    def document_params(self) -> Dict[str, str]:
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

    def document_params(self) -> Dict[str, str]:
        return {
            "DepotEventAnalyzer.vehicle_ids": """
Optional parameter to filter vehicles. Can be:
- A single vehicle ID (int)
- A list of vehicle IDs (List[int])
- None to include all vehicles (default)

Example: `params["DepotEventAnalyzer.vehicle_ids"] = [1, 2, 3]`
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

    def document_params(self) -> Dict[str, str]:
        return {
            "PowerAndOccupancyAnalyzer.area_id": """
**Required** parameter specifying the area ID(s) to analyze. Can be:
- A single area ID (int)
- A list of area IDs (Iterable[int])

Example: `params["PowerAndOccupancyAnalyzer.area_id"] = [1, 2]`
            """.strip(),
            "PowerAndOccupancyAnalyzer.temporal_resolution": """
Temporal resolution of the timeseries in seconds. Default is 60 seconds.

Example: `params["PowerAndOccupancyAnalyzer.temporal_resolution"] = 120`
            """.strip(),
            "PowerAndOccupancyAnalyzer.station_id": """
Optional station ID(s) for opportunity charging events. Can be:
- A single station ID (int)
- A list of station IDs (Iterable[int])
- None to exclude stations (default)

Example: `params["PowerAndOccupancyAnalyzer.station_id"] = [1, 2]`
            """.strip(),
            "PowerAndOccupancyAnalyzer.sim_start_time": """
Optional start time to filter the timeseries. If set, no data before this time is included.

Example: `params["PowerAndOccupancyAnalyzer.sim_start_time"] = datetime(...)`
            """.strip(),
            "PowerAndOccupancyAnalyzer.sim_end_time": """
Optional end time to filter the timeseries. If set, no data after this time is included.

Example: `params["PowerAndOccupancyAnalyzer.sim_end_time"] = datetime(...)`
            """.strip(),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Analyze the database and return power and occupancy timeseries.

        Args:
            session: SQLAlchemy session connected to the eflips-model database
            params: Analysis parameters (must include area_id)

        Returns:
            DataFrame with power and occupancy timeseries

        Raises:
            ValueError: If area_id parameter is not provided
        """

        # Extract required parameter
        area_id = params.get(f"{self.__class__.__name__}.area_id")
        if area_id is None:
            raise ValueError(
                f"Required parameter '{self.__class__.__name__}.area_id' not provided"
            )

        # Extract optional parameters
        temporal_resolution = params.get(f"{self.__class__.__name__}.temporal_resolution", 60)
        station_id = params.get(f"{self.__class__.__name__}.station_id", None)
        sim_start_time = params.get(f"{self.__class__.__name__}.sim_start_time", None)
        sim_end_time = params.get(f"{self.__class__.__name__}.sim_end_time", None)

        # Call eflips-eval prepare function
        result = eval_output_prepare.power_and_occupancy(
            area_id,
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

    def document_params(self) -> Dict[str, str]:
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

    def document_params(self) -> Dict[str, str]:
        return {
            "VehicleSocAnalyzer.vehicle_id": """
**Required** parameter specifying the vehicle ID to analyze.

Example: `params["VehicleSocAnalyzer.vehicle_id"] = 1`
            """.strip(),
            "VehicleSocAnalyzer.timezone": """
Optional timezone for the visualization. Default is Europe/Berlin.

Example: `params["VehicleSocAnalyzer.timezone"] = ZoneInfo("UTC")`
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

    def document_params(self) -> Dict[str, str]:
        return {
            "DepotActivityAnalyzer.depot_id": """
**Required** parameter specifying the depot ID to analyze.

Example: `params["DepotActivityAnalyzer.depot_id"] = 1`
            """.strip(),
            "DepotActivityAnalyzer.animation_start": """
**Required** parameter specifying the start time of the animation range.

Example: `params["DepotActivityAnalyzer.animation_start"] = datetime(...)`
            """.strip(),
            "DepotActivityAnalyzer.animation_end": """
**Required** parameter specifying the end time of the animation range.

Example: `params["DepotActivityAnalyzer.animation_end"] = datetime(...)`
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
