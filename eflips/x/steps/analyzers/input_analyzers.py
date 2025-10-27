"""
Analyzers for input evaluation using eflips-eval.

These analyzers wrap the eflips.eval.input module's prepare/visualize functions
to make them usable within the eflips-x pipeline framework.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

import dash_cytoscape as cyto  # type: ignore
import eflips.model
import folium  # type: ignore
import pandas as pd
import plotly.graph_objs as go  # type: ignore
from eflips.eval.input import prepare as eval_input_prepare
from eflips.eval.input import visualize as eval_input_visualize
from eflips.model import Scenario
from sqlalchemy.orm import Session

from eflips.x.framework import Analyzer


class RotationInfoAnalyzer(Analyzer):
    """
    Analyzer for rotation information.

    Provides information about rotations in a scenario including:
    - Rotation ID and name
    - Vehicle type information
    - Total distance
    - Start and end times
    - Line information
    - Start and end stations

    This information is available before simulation has been run.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    def document_params(self) -> Dict[str, str]:
        return {
            "RotationInfoAnalyzer.rotation_ids": """
Optional parameter to filter rotations. Can be:
- A single rotation ID (int)
- A list of rotation IDs (List[int])
- None to include all rotations (default)

Example: `params["RotationInfoAnalyzer.rotation_ids"] = [1, 2, 3]`
            """.strip()
        }

    def analyze(self, db: Path, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze the database and return rotation information.

        Args:
            db: Path to the database file
            params: Analysis parameters

        Returns:
            DataFrame with rotation information
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Auto-detect scenario_id
            scenario = session.query(Scenario).one()
            scenario_id = scenario.id

            # Extract parameters
            rotation_ids = params.get(f"{self.__class__.__name__}.rotation_ids", None)

            # Call eflips-eval prepare function
            result = eval_input_prepare.rotation_info(scenario_id, session, rotation_ids)

            return result
        finally:
            session.close()
            engine.dispose()

    @staticmethod
    def visualize(
        prepared_data: pd.DataFrame, timezone: ZoneInfo = ZoneInfo("Europe/Berlin")
    ) -> go.Figure:
        """
        Visualize rotation information as a timeline using plotly.

        Args:
            prepared_data: Result from analyze() method
            timezone: Timezone for display (default: Europe/Berlin)

        Returns:
            Plotly figure object
        """
        return eval_input_visualize.rotation_info(prepared_data, timezone)


class GeographicTripPlotAnalyzer(Analyzer):
    """
    Analyzer for geographic trip visualization.

    Creates a dataframe that can be used to visualize the geographic
    distribution of rotations with one row for each trip.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    def document_params(self) -> Dict[str, str]:
        return {
            "GeographicTripPlotAnalyzer.rotation_ids": """
Optional parameter to filter rotations. Can be:
- A single rotation ID (int)
- A list of rotation IDs (List[int])
- None to include all rotations (default)

Example: `params["GeographicTripPlotAnalyzer.rotation_ids"] = [1, 2, 3]`
            """.strip()
        }

    def analyze(self, db: Path, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze the database and return geographic trip data.

        Args:
            db: Path to the database file
            params: Analysis parameters

        Returns:
            DataFrame with geographic trip information
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Auto-detect scenario_id
            scenario = session.query(Scenario).one()
            scenario_id = scenario.id

            # Extract parameters
            rotation_ids = params.get(f"{self.__class__.__name__}.rotation_ids", None)

            # Call eflips-eval prepare function
            result = eval_input_prepare.geographic_trip_plot(scenario_id, session, rotation_ids)

            return result
        finally:
            session.close()
            engine.dispose()

    @staticmethod
    def visualize(prepared_data: pd.DataFrame) -> folium.Map:
        """
        Visualize trips on a map using folium.

        Args:
            prepared_data: Result from analyze() method

        Returns:
            Folium map object
        """
        return eval_input_visualize.geographic_trip_plot(prepared_data)


class SingleRotationInfoAnalyzer(Analyzer):
    """
    Analyzer for detailed single rotation information.

    Provides information about all trips in a single rotation including:
    - Trip ID and type
    - Line and route names
    - Distance
    - Departure and arrival times
    - Station information
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    def document_params(self) -> Dict[str, str]:
        return {
            "SingleRotationInfoAnalyzer.rotation_id": """
**Required** parameter specifying the rotation ID to analyze.

Example: `params["SingleRotationInfoAnalyzer.rotation_id"] = 1`
            """.strip()
        }

    def analyze(self, db: Path, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze the database and return detailed information for a single rotation.

        Args:
            db: Path to the database file
            params: Analysis parameters (must include rotation_id)

        Returns:
            DataFrame with trip information for the rotation

        Raises:
            ValueError: If rotation_id parameter is not provided
        """
        db_url = f"sqlite:///{db.absolute().as_posix()}"
        engine = eflips.model.create_engine(db_url)
        session = Session(engine)

        try:
            # Extract required parameter
            rotation_id = params.get(f"{self.__class__.__name__}.rotation_id")
            if rotation_id is None:
                raise ValueError(
                    f"Required parameter '{self.__class__.__name__}.rotation_id' not provided"
                )

            # Call eflips-eval prepare function
            result = eval_input_prepare.single_rotation_info(rotation_id, session)

            return result
        finally:
            session.close()
            engine.dispose()

    @staticmethod
    def visualize(prepared_data: pd.DataFrame) -> cyto.Cytoscape:
        """
        Visualize a single rotation as a network graph using Dash Cytoscape.

        Nodes represent stops and edges represent trips between stops.

        Args:
            prepared_data: Result from analyze() method

        Returns:
            Dash Cytoscape object (can be added to a Dash layout)
        """
        return eval_input_visualize.single_rotation_info(prepared_data)
