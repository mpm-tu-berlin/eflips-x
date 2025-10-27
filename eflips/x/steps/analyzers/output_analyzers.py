"""
Analyzers for output evaluation using eflips-eval.

These analyzers wrap the eflips.eval.output module's prepare/visualize functions
to make them usable within the eflips-x pipeline framework.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import eflips.model
import matplotlib.animation as animation
import pandas as pd
import plotly.graph_objs as go  # type: ignore
from eflips.eval.output import prepare as eval_output_prepare
from eflips.eval.output import visualize as eval_output_visualize
from eflips.model import Area, Scenario
from matplotlib.figure import Figure
from sqlalchemy.orm import Session

from eflips.x.framework import Analyzer


class DepartureArrivalSocAnalyzer(Analyzer):
    """
    Analyzer for vehicle State of Charge (SoC) at depot departure and arrival.

    Creates a dataframe with the SoC at departure from depot and arrival at depot
    for each trip. This analyzer requires simulation results to be present.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    def document_params(self) -> Dict[str, str]:
        return {}

    def analyze(self, db: Path, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze the database and return SoC at departure/arrival.

        Args:
            db: Path to the database file
            params: Analysis parameters (not used)

        Returns:
            DataFrame with SoC information at depot departure/arrival
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Auto-detect scenario_id
            scenario = session.query(Scenario).one()
            scenario_id = scenario.id

            # Call eflips-eval prepare function
            result = eval_output_prepare.departure_arrival_soc(scenario_id, session)

            return result
        finally:
            session.close()
            engine.dispose()

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
    Analyzer for depot events.

    Creates a dataframe with all events at the depot for a given scenario.
    This analyzer requires simulation results to be present.
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

    def analyze(self, db: Path, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze the database and return depot events.

        Args:
            db: Path to the database file
            params: Analysis parameters

        Returns:
            DataFrame with depot events
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Auto-detect scenario_id
            scenario = session.query(Scenario).one()
            scenario_id = scenario.id

            # Extract parameters
            vehicle_ids = params.get(f"{self.__class__.__name__}.vehicle_ids", None)

            # Call eflips-eval prepare function
            result = eval_output_prepare.depot_event(scenario_id, session, vehicle_ids)

            return result
        finally:
            session.close()
            engine.dispose()

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
    Analyzer for power consumption and depot occupancy over time.

    Creates a timeseries dataframe of power consumption and occupancy
    for specified area(s) and/or station(s).
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

    def analyze(self, db: Path, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze the database and return power and occupancy timeseries.

        Args:
            db: Path to the database file
            params: Analysis parameters (must include area_id)

        Returns:
            DataFrame with power and occupancy timeseries

        Raises:
            ValueError: If area_id parameter is not provided
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Extract required parameter
            area_id = params.get(f"{self.__class__.__name__}.area_id")
            if area_id is None:
                raise ValueError(
                    f"Required parameter '{self.__class__.__name__}.area_id' not provided"
                )

            # Extract optional parameters
            temporal_resolution = params.get(
                f"{self.__class__.__name__}.temporal_resolution", 60
            )
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
        finally:
            session.close()
            engine.dispose()

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
    Analyzer for specific energy consumption.

    Creates a dataframe of all trip energy consumptions and distances
    for the given scenario.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    def document_params(self) -> Dict[str, str]:
        return {}

    def analyze(self, db: Path, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze the database and return specific energy consumption data.

        Args:
            db: Path to the database file
            params: Analysis parameters (not used)

        Returns:
            DataFrame with trip energy consumption information
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Auto-detect scenario_id
            scenario = session.query(Scenario).one()
            scenario_id = scenario.id

            # Call eflips-eval prepare function
            result = eval_output_prepare.specific_energy_consumption(scenario_id, session)

            return result
        finally:
            session.close()
            engine.dispose()

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
    Analyzer for vehicle State of Charge over time.

    Takes a vehicle ID and returns a description of what happened to the vehicle
    over time, including rotations, charging events, and trips.
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
        self, db: Path, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, datetime, datetime]]]]:
        """
        Analyze the database and return vehicle SoC timeseries and event descriptions.

        Args:
            db: Path to the database file
            params: Analysis parameters (must include vehicle_id)

        Returns:
            Tuple of (DataFrame with SoC timeseries, Dict with event descriptions)

        Raises:
            ValueError: If vehicle_id parameter is not provided
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Extract required parameter
            vehicle_id = params.get(f"{self.__class__.__name__}.vehicle_id")
            if vehicle_id is None:
                raise ValueError(
                    f"Required parameter '{self.__class__.__name__}.vehicle_id' not provided"
                )

            # Extract optional parameters
            timezone = params.get(
                f"{self.__class__.__name__}.timezone", ZoneInfo("Europe/Berlin")
            )

            # Call eflips-eval prepare function
            result = eval_output_prepare.vehicle_soc(vehicle_id, session, timezone)

            return result
        finally:
            session.close()
            engine.dispose()

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


class DepotLayoutAnalyzer(Analyzer):
    """
    Analyzer for depot layout.

    Returns a list of Area objects representing all areas in the depot,
    organized into blocks.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    def document_params(self) -> Dict[str, str]:
        return {
            "DepotLayoutAnalyzer.depot_id": """
**Required** parameter specifying the depot ID to analyze.

Example: `params["DepotLayoutAnalyzer.depot_id"] = 1`
            """.strip()
        }

    def analyze(self, db: Path, params: Dict[str, Any]) -> List[List[Area]]:
        """
        Analyze the database and return depot layout information.

        Args:
            db: Path to the database file
            params: Analysis parameters (must include depot_id)

        Returns:
            List of lists of Area objects representing depot layout

        Raises:
            ValueError: If depot_id parameter is not provided
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Extract required parameter
            depot_id = params.get(f"{self.__class__.__name__}.depot_id")
            if depot_id is None:
                raise ValueError(
                    f"Required parameter '{self.__class__.__name__}.depot_id' not provided"
                )

            # Call eflips-eval prepare function
            result = eval_output_prepare.depot_layout(depot_id, session)

            return result
        finally:
            session.close()
            engine.dispose()

    @staticmethod
    def visualize(area_blocks: List[List[Area]]) -> Tuple[Dict, Figure]:
        """
        Visualize depot layout using matplotlib.

        Args:
            area_blocks: Result from analyze() method

        Returns:
            Tuple of (area dictionary, matplotlib figure)
        """
        return eval_output_visualize.depot_layout(area_blocks)


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
        self, db: Path, params: Dict[str, Any]
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
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
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
        finally:
            session.close()
            engine.dispose()

    @staticmethod
    def visualize(
        area_blocks: List[List[Area]],
        area_occupancy: Dict[Tuple[int, int], List[Tuple[int, int]]],
        animation_range: Tuple[datetime, datetime],
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
        return eval_output_visualize.depot_activity_animation(
            area_blocks, area_occupancy, animation_range, time_resolution
        )
