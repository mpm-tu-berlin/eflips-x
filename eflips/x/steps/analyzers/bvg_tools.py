"""
BVG-specific analyzers for eflips-x pipeline.

This module contains analyzers tailored for BVG (Berlin public transport) data analysis.
"""

import warnings
from typing import Any, Dict, List, Tuple

import cartopy.crs as ccrs  # type: ignore[import-untyped]
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from eflips.model import Rotation, Route, Station, TripType, VehicleType, Trip
from geoalchemy2.shape import to_shape
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from sqlalchemy.orm import Session

from eflips.x.framework import Analyzer

# ============================================================================
# Plot Configuration Constants
# ============================================================================

# Plot dimensions in points (for LaTeX integration)
PLOT_WIDTH_PT = 490.0
PLOT_HEIGHT_PT = 375.0

# Convert points to inches for matplotlib (1 pt = 1/72 inch)
PLOT_WIDTH_INCH = PLOT_WIDTH_PT / 72.0
PLOT_HEIGHT_INCH = PLOT_HEIGHT_PT / 72.0

# LaTeX-compatible matplotlib configuration
LATEX_RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.0,
}

# Vehicle type name translations
VEHICLE_NAME_TRANSLATION = {
    "EN": "Single Decker",
    "GN": "Articulated Bus",
    "DD": "Double Decker",
}

# Standard vehicle type ordering
VEHICLE_TYPE_ORDER = ["Single Decker", "Articulated Bus", "Double Decker"]

# BVG depot short names
BVG_DEPOT_SHORT_NAMES = ["BTRB", "BFI", "BHLI", "BF S", "BF C", "BF M", "BF MDA"]


# ============================================================================
# Helper Functions
# ============================================================================


def configure_latex_plotting() -> None:
    """Configure matplotlib for LaTeX-compatible plotting."""
    matplotlib.rcParams.update(LATEX_RC_PARAMS)


def clean_depot_name(depot_name: str) -> str:
    """
    Clean depot station name by removing common prefixes.

    Args:
        depot_name: Original depot station name

    Returns:
        Cleaned depot name
    """
    cleaned = depot_name.removeprefix("Betriebshof ")
    cleaned = cleaned.removeprefix("Abstellfläche ")
    return cleaned


# ============================================================================
# Analyzers
# ============================================================================


class VehicleTypeDepotPlotAnalyzer(Analyzer):
    """
    Analyzer for creating stacked bar charts showing vehicle kilometers by depot and vehicle type.

    This analyzer creates a publication-quality plot with LaTeX rendering showing the distribution
    of revenue mileage across different vehicle types and depots. The plot is saved as both PDF
    and PNG formats.
    """

    def __init__(self, code_version: str = "v1.0.5", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.scenario_id": """
Optional scenario ID to analyze. If not provided, assumes single scenario in database.
Default: None (auto-detect)
            """.strip(),
            f"{cls.__name__}.depot_short_names": f"""
List of depot short names to include in the analysis.
Default: {BVG_DEPOT_SHORT_NAMES}
            """.strip(),
            f"{cls.__name__}.weeks_per_year": """
Number of weeks to scale the results to annual values.
Default: 52 weeks
            """.strip(),
        }

    def analyze(self, session: Session, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze vehicle kilometers by depot and vehicle type, and generate plots.

        Args:
            session: SQLAlchemy session connected to the eflips-model database
            params: Analysis parameters

        Returns:
            - dataframe: DataFrame with vehicle/depot statistics
        """
        # Extract parameters
        scenario_id = params.get(f"{self.__class__.__name__}.scenario_id", 1)
        depot_short_names = params.get(
            f"{self.__class__.__name__}.depot_short_names", BVG_DEPOT_SHORT_NAMES
        )
        weeks_per_year = params.get(f"{self.__class__.__name__}.weeks_per_year", 52)

        # Collect data for each vehicle type and depot combination
        vehicle_type_data: List[Dict[str, Any]] = []

        vehicle_types = (
            session.query(VehicleType).filter(VehicleType.scenario_id == scenario_id).all()
        )

        for vehicle_type in vehicle_types:
            for depot_short_name in depot_short_names:
                # Get depot station
                depot_station = (
                    session.query(Station).filter(Station.name_short == depot_short_name).first()
                )

                if not depot_station:
                    warnings.warn(
                        f"Depot station with short name '{depot_short_name}' not found for vehicle type '{vehicle_type.name_short}'. Skipping."
                    )
                    continue

                depot_station_name = clean_depot_name(depot_station.name)

                # Query rotations originating from this depot with this vehicle type
                rotations = (
                    session.query(Rotation)
                    .join(Trip)
                    .join(Route)
                    .join(Station, Route.departure_station_id == Station.id)
                    .join(VehicleType)
                    .filter(
                        Rotation.scenario_id == scenario_id,
                        VehicleType.id == vehicle_type.id,
                        Station.name_short == depot_short_name,
                    )
                    .all()
                )

                # Count trips
                trips = sum([len(rotation.trips) for rotation in rotations])

                # Calculate total passenger trip distance
                total_distance = (
                    sum(
                        [
                            sum(
                                [
                                    (
                                        trip.route.distance
                                        if trip.trip_type == TripType.PASSENGER
                                        else 0
                                    )
                                    for trip in rotation.trips
                                ]
                            )
                            for rotation in rotations
                        ]
                    )
                    / 1000  # Convert to km
                )

                vehicle_type_data.append(
                    {
                        "Fahrzeugtyp": vehicle_type.name_short,
                        "depot": depot_station_name,
                        "Umläufe": len(rotations),
                        "trips": trips,
                        "Fahrzeugkilometer": total_distance,
                    }
                )

        # Create DataFrame
        df = pd.DataFrame(vehicle_type_data)

        # Translate vehicle type names
        df["Fahrzeugtyp"] = df["Fahrzeugtyp"].apply(lambda x: VEHICLE_NAME_TRANSLATION.get(x, x))

        # Scale to annual values and convert to millions of km
        df["Fahrzeugkilometer"] *= weeks_per_year / 1_000_000

        return df

    @staticmethod
    def visualize(
        prepared_data: pd.DataFrame,
    ) -> Figure:
        """
        Create and save stacked bar chart visualization.

        Args:
            prepared_data: Result DataFrame from analyze() method
            output_dir: Directory where plots will be saved (if None, uses current directory)
            output_basename: Base name for output files without extension

        Returns:
            Tuple of (pdf_path, png_path) for the saved plot files
        """
        # Configure matplotlib for LaTeX
        configure_latex_plotting()

        # Create figure with specified dimensions
        fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH))

        # Pivot data for stacked bar chart
        df_pivot = prepared_data.pivot(
            index="depot", columns="Fahrzeugtyp", values="Fahrzeugkilometer"
        )

        # Reorder columns to match standard vehicle type ordering
        available_types = [vt for vt in VEHICLE_TYPE_ORDER if vt in df_pivot.columns]
        df_pivot = df_pivot[available_types]

        # Use seaborn Set2 color palette
        palette = sns.color_palette("Set2")

        # Create stacked bar chart
        df_pivot.plot(kind="bar", stacked=True, ax=ax, color=palette)

        # Configure plot appearance
        ax.set_title("")
        ax.set_ylabel(
            r"Revenue Mileage $\left[ \frac{\mathrm{km} \times 10^6}{\mathrm{a}} \right]$"
        )
        ax.set_xlabel("")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Configure legend
        plt.legend(title="", bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper left", ncols=3)

        plt.tight_layout()

        return fig


# ============================================================================
# Route Visualization Functions
# ============================================================================


def visualize_routes_by_depot_cartopy(
    prepared_data: pd.DataFrame, session: Session, projection_type: str = "equidistant"
) -> Figure:
    """
    Create matplotlib/cartopy visualization of routes colored by depot.

    This function creates a publication-quality map showing all routes in the network,
    with each route colored according to its originating depot. The map uses a
    locally-centered equidistant projection (or UTM) to provide kilometer-based axes.
    Depot locations are shown as colored dots matching their respective route colors.

    Args:
        prepared_data: DataFrame from GeographicTripPlotAnalyzer.analyze()
                      Expected columns:
                      - coordinates: List of (lat, lon) tuples for route geometry
                      - originating_depot_name: Depot name for color grouping
        session: SQLAlchemy session for querying depot locations
        projection_type: Type of projection to use
                        - "equidistant" (default): Azimuthal equidistant centered on network
                        - "utm": Auto-discovered UTM zone based on network location

    Returns:
        matplotlib Figure with routes plotted on cartopy axes
    """
    # Configure LaTeX plotting
    configure_latex_plotting()

    # Calculate network centroid from all coordinates
    all_coords: List[Tuple[float, float]] = []
    for coords_list in prepared_data["coordinates"]:
        if coords_list:  # Handle empty coordinate lists
            all_coords.extend(coords_list)

    if not all_coords:
        raise ValueError("No coordinate data found in prepared_data")

    # Extract lats and lons for centroid calculation
    center_lat = np.mean([coord[0] for coord in all_coords])
    center_lon = np.mean([coord[1] for coord in all_coords])

    # Create projection based on type
    if projection_type == "utm":
        # Auto-discover UTM zone from centroid longitude
        utm_zone = int((center_lon + 180) / 6) + 1
        southern_hemisphere = center_lat < 0
        projection = ccrs.UTM(zone=utm_zone, southern_hemisphere=southern_hemisphere)
    else:  # equidistant (default)
        # Create azimuthal equidistant projection centered on network
        projection = ccrs.AzimuthalEquidistant(
            central_longitude=center_lon, central_latitude=center_lat
        )

    # Deduplicate routes by unique coordinate sequences
    # Convert coordinate lists to hashable tuples for grouping
    prepared_data_copy = prepared_data.copy()
    prepared_data_copy["coord_hash"] = prepared_data_copy["coordinates"].apply(
        lambda coords: hash(tuple(coords)) if coords else None
    )

    # Group by coordinate hash and depot, take first occurrence
    unique_routes = (
        prepared_data_copy.groupby(["coord_hash", "originating_depot_name"], dropna=False)
        .first()
        .reset_index()
    )

    # Create color mapping for depots
    # Separate routes with and without depot assignments
    unique_depots = sorted(
        [d for d in prepared_data["originating_depot_name"].unique() if pd.notna(d)]
    )
    palette = sns.color_palette("Set2", n_colors=len(unique_depots))
    depot_colors = dict(zip(unique_depots, palette))

    # Create figure with cartopy projection
    # Increase height to accommodate legend below
    fig, ax = plt.subplots(
        figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH), subplot_kw={"projection": projection}
    )

    # Track if we have routes without depot assignment
    has_unassigned_routes = False

    # Plot each unique route
    for idx, row in unique_routes.iterrows():
        coords = row["coordinates"]
        if not coords:  # Skip empty coordinate lists
            continue

        depot_name = row["originating_depot_name"]

        # Extract lats and lons
        lats = [coord[0] for coord in coords]
        lons = [coord[1] for coord in coords]

        # Determine color and handle unassigned routes
        if pd.notna(depot_name):
            color = depot_colors[depot_name]
        else:
            color = "grey"
            has_unassigned_routes = True

        # Transform from WGS84 to projection coordinates
        ax.plot(
            lons,
            lats,
            color=color,
            linewidth=0.8,
            transform=ccrs.PlateCarree(),  # Input is WGS84
            alpha=0.7,
            zorder=2,
        )

    # Query and plot depot locations
    for depot_name in unique_depots:
        # Query the depot station from database
        depot_station = session.query(Station).filter(Station.name.contains(depot_name)).first()

        if depot_station and depot_station.geom is not None:
            # Extract latitude and longitude from the station geometry
            # Assuming geom is a WKBElement with lat/lon
            point = to_shape(depot_station.geom)  # type: ignore[arg-type]
            depot_lat = point.y
            depot_lon = point.x

            # Plot depot location as a colored dot
            ax.scatter(
                depot_lon,
                depot_lat,
                color=depot_colors[depot_name],
                s=100,  # Marker size
                marker="o",
                edgecolors="black",
                linewidths=1.5,
                transform=ccrs.PlateCarree(),
                zorder=3,  # Place on top of routes
                alpha=0.9,
            )

    # Format axes to show kilometers
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("Distance [km]")

    # Convert axis ticks from meters to kilometers
    def m_to_km_formatter(x: float, pos: int) -> str:
        return f"{x/1000:.0f}"

    ax.xaxis.set_major_formatter(FuncFormatter(m_to_km_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(m_to_km_formatter))

    # Add gridlines with kilometer labels
    ax.gridlines(draw_labels=True, alpha=0.3, zorder=1)  # type: ignore[attr-defined]

    # Create legend
    legend_elements = [
        Patch(facecolor=depot_colors[depot], label=depot, alpha=0.7) for depot in unique_depots
    ]

    # Add grey lines to legend if they exist
    if has_unassigned_routes:
        legend_elements.append(Patch(facecolor="grey", label="No depot assigned", alpha=0.7))

    # Position legend below plot in 3 columns
    ax.legend(
        handles=legend_elements,
        title="Depot",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()

    return fig
