"""
Analyzers for input evaluation using eflips-eval.

These analyzers wrap the eflips.eval.input module's prepare/visualize functions
to make them usable within the eflips-x pipeline framework.
"""

import json
from pathlib import Path
from typing import Any, Dict
from zoneinfo import ZoneInfo

import dash_cytoscape as cyto  # type: ignore
import folium  # type: ignore
import pandas as pd
import plotly.graph_objs as go  # type: ignore
import sqlalchemy
from eflips.eval.input import prepare as eval_input_prepare
from eflips.eval.input import visualize as eval_input_visualize
from eflips.model import Scenario
from sqlalchemy.orm import Session

from eflips.x.framework import Analyzer


class RotationInfoAnalyzer(Analyzer):
    """
    This Analyzer provides information about the rotations in a scenario. This information can be provided even before
    the simulation has been run. It creates a dataframe with the following columns:

    - rotation_id: the id of the rotation
    - rotation_name: the name of the rotation
    - vehicle_type_id: the id of the vehicle type
    - vehicle_type_name: the name of the vehicle type
    - total_distance: the total distance of the rotation
    - time_start: the departure of the first trip
    - time_end: the arrival of the last trip
    - line_name: the name of the line, which is the first part of the rotation name. Used for sorting
    - line_is_unified: True if the rotation only contains one line
    - start_station: the name of the departure station
    - end_station: the name of the arrival station

    This information is available before simulation has been run.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.rotation_ids": f"""
Optional parameter to filter rotations. Can be:
- A single rotation ID (int)
- A list of rotation IDs (List[int])
- None to include all rotations (default)

Example: `params["{cls.__name__}.rotation_ids"] = [1, 2, 3]`
            """.strip()
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        This function provides information about the rotations in a scenario. This information can be provided even before
        the simulation has been run. It creates a dataframe with the following columns:

        - rotation_id: the id of the rotation
        - rotation_name: the name of the rotation
        - vehicle_type_id: the id of the vehicle type
        - vehicle_type_name: the name of the vehicle type
        - total_distance: the total distance of the rotation
        - time_start: the departure of the first trip
        - time_end: the arrival of the last trip
        - line_name: the name of the line, which is the first part of the rotation name. Used for sorting
        - line_is_unified: True if the rotation only contains one line
        - start_station: the name of the departure station
        - end_station: the name of the arrival station

        :return: a pandas DataFrame
        """

        # Auto-detect scenario_id
        scenario = session.query(Scenario).one()
        scenario_id = scenario.id

        # Extract parameters
        rotation_ids = params.get(f"{self.__class__.__name__}.rotation_ids", None)

        # Call eflips-eval prepare function
        result = eval_input_prepare.rotation_info(scenario_id, session, rotation_ids)

        return result

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
    This Analyzer creates a dataframe that can be used to visualize the geographic distribution of rotations. It creates
    a dataframe with one row for each trip and the following columns:

    - rotation_id: the id of the rotation
    - rotation_name: the name of the rotation
    - vehicle_type_id: the id of the vehicle type
    - vehicle_type_name: the name of the vehicle type
    - originating_depot_id: the id of the originating depot
    - originating_depot_name: the name of the originating depot
    - distance: the distance of the route
    - coordinates: An array of (lat, lon) tuples with the coordinates of the route - the shape if set, otherwise the stops
    - line_name: the name of the line, which is the first part of the rotation name. Used for sorting
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.rotation_ids": f"""
Optional parameter to filter rotations. Can be:
- A single rotation ID (int)
- A list of rotation IDs (List[int])
- None to include all rotations (default)

Example: `params[f"{cls.__name__}.rotation_ids"] = [1, 2, 3]`
            """.strip()
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        This function creates a dataframe that can be used to visualize the geographic distribution of rotations. It creates
        a dataframe with one row for each trip and the following columns:

        - rotation_id: the id of the rotation
        - rotation_name: the name of the rotation
        - vehicle_type_id: the id of the vehicle type
        - vehicle_type_name: the name of the vehicle type
        - originating_depot_id: the id of the originating depot
        - originating_depot_name: the name of the originating depot
        - distance: the distance of the route
        - coordinates: An array of (lat, lon) tuples with the coordinates of the route - the shape if set, otherwise the stops
        - line_name: the name of the line, which is the first part of the rotation name. Used for sorting

        :return: a pandas DataFrame
        """

        # Auto-detect scenario_id
        scenario = session.query(Scenario).one()
        scenario_id = scenario.id

        # Extract parameters
        rotation_ids = params.get(f"{self.__class__.__name__}.rotation_ids", None)

        # Call eflips-eval prepare function
        result = eval_input_prepare.geographic_trip_plot(scenario_id, session, rotation_ids)

        return result

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
    This Analyzer provides information over the trips in a single rotation and returns a pandas DataFrame with the
    following columns:

    - trip_id: the id of the trip
    - trip_type: the type of the trip
    - line_name: the name of the line
    - route_name: the name of the route
    - distance: the distance of the route
    - departure_time: the departure time of the trip
    - arrival_time: the arrival time of the trip
    - departure_station_name: the name of the departure station
    - departure_station_id: the id of the departure station
    - arrival_station_name: the name of the arrival station
    - arrival_station_id: the id of the arrival station
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.rotation_id": f"""
**Required** parameter specifying the rotation ID to analyze.

Example: `params["{cls.__name__}.rotation_id"] = 1`
            """.strip()
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        This Analyzer provides information over the trips in a single rotation and returns a pandas DataFrame with the
        following columns:

        - trip_id: the id of the trip
        - trip_type: the type of the trip
        - line_name: the name of the line
        - route_name: the name of the route
        - distance: the distance of the route
        - departure_time: the departure time of the trip
        - arrival_time: the arrival time of the trip
        - departure_station_name: the name of the departure station
        - departure_station_id: the id of the departure station
        - arrival_station_name: the name of the arrival station
        - arrival_station_id: the id of the arrival station

        Args:
            db: Path to the database file
            params: Analysis parameters (must include rotation_id)

        Returns:
            DataFrame with trip information for the rotation

        Raises:
            ValueError: If rotation_id parameter is not provided
        """

        # Extract required parameter
        rotation_id = params.get(f"{self.__class__.__name__}.rotation_id")
        if rotation_id is None:
            raise ValueError(
                f"Required parameter '{self.__class__.__name__}.rotation_id' not provided"
            )

        # Call eflips-eval prepare function
        result = eval_input_prepare.single_rotation_info(rotation_id, session)

        return result

    def export_cytoscape_html(
        self, cytoscape: cyto.Cytoscape, filename: str | Path, layout: str = "cose"
    ) -> None:
        """Export Cytoscape elements to a self-contained HTML file."""

        elements = cytoscape.elements

        html_template = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Cytoscape Graph</title>
        <script src="https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
        <style>
            #cy {{
                width: 100%;
                height: 100vh;
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <div id="cy"></div>
        <script>
            var cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: {json.dumps(elements)},
                layout: {{ name: '{layout}' }},
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'label': 'data(label)',
                            'background-color': '#0074D9',
                            'color': '#fff',
                            'text-valign': 'center',
                            'text-halign': 'center'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 2,
                            'line-color': '#ccc',
                            'target-arrow-color': '#ccc',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier'
                        }}
                    }}
                ]
            }});
        </script>
    </body>
    </html>"""

        with open(filename, "w") as f:
            f.write(html_template)
        self.logger.info(f"Exported to {filename}")

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
