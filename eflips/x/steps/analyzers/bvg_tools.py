"""
BVG-specific analyzers for eflips-x pipeline.

This module contains analyzers tailored for BVG (Berlin public transport) data analysis.
"""

import warnings
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs  # type: ignore[import-untyped]
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from eflips.model import (
    Area,
    Depot,
    Event,
    EventType,
    Process,
    Rotation,
    Route,
    Scenario,
    Station,
    TripType,
    Vehicle,
    VehicleType,
    Trip,
    ChargeType,
)
from geoalchemy2.shape import to_shape
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from sqlalchemy import func
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

# Revenue service timeline plot defaults (datetime objects for xlim cropping)
from datetime import datetime as dt

REVENUE_SERVICE_PLOT_START = dt(
    2025, 6, 16, 0, 0, tzinfo=ZoneInfo("Europe/Berlin")
)  # Monday 00:00
REVENUE_SERVICE_PLOT_END = dt(
    2025, 6, 23, 0, 0, tzinfo=ZoneInfo("Europe/Berlin")
)  # Next Monday 00:00


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


class RevenueServiceTimelineAnalyzer(Analyzer):
    """
    Analyzer for creating stacked area charts showing the number of vehicles
    in revenue (PASSENGER) service at each minute of the day, grouped by vehicle type.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.timezone": """
Timezone for localizing departure/arrival times.
Default: "Europe/Berlin"
            """.strip(),
        }

    def analyze(self, session: Session, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Count vehicles in revenue service at each minute, grouped by vehicle type.

        Returns a DataFrame with a datetime index (minute resolution) and one column
        per vehicle type, containing the count of vehicles in service at that minute.
        """
        import pytz

        timezone_str = params.get(f"{self.__class__.__name__}.timezone", "Europe/Berlin")
        tz = pytz.timezone(timezone_str)

        # Query all PASSENGER trips with their vehicle type
        trips = (
            session.query(Trip)
            .join(Rotation)
            .join(VehicleType)
            .filter(Trip.trip_type == TripType.PASSENGER)
            .all()
        )

        # Build event list: +1 at departure, -1 at arrival
        events: List[Dict[str, Any]] = []
        for trip in trips:
            vt_short = trip.rotation.vehicle_type.name_short
            vt_name = VEHICLE_NAME_TRANSLATION.get(vt_short, vt_short)

            dep_time = trip.departure_time.astimezone(tz)
            arr_time = trip.arrival_time.astimezone(tz)

            events.append({"time": dep_time, "vehicle_type": vt_name, "delta": 1})
            events.append({"time": arr_time, "vehicle_type": vt_name, "delta": -1})

        if not events:
            return pd.DataFrame()

        df_events = pd.DataFrame(events)

        # Build running count per vehicle type
        vehicle_types_present = [
            vt for vt in VEHICLE_TYPE_ORDER if vt in df_events["vehicle_type"].unique()
        ]

        # Create minute-resolution time index covering full data range
        time_min = df_events["time"].min().floor("min")
        time_max = df_events["time"].max().ceil("min")
        minute_index = pd.date_range(start=time_min, end=time_max, freq="min")

        result = pd.DataFrame(index=minute_index)

        for vt in vehicle_types_present:
            vt_events_df = df_events[df_events["vehicle_type"] == vt].copy()
            vt_grouped: "pd.Series[Any]" = vt_events_df.groupby("time")["delta"].sum().sort_index()
            vt_cumsum = vt_grouped.cumsum()
            # Reindex to minute resolution with forward fill
            vt_cumsum = vt_cumsum.reindex(minute_index, method="ffill").fillna(0)
            result[vt] = vt_cumsum

        result.index.name = "time"
        return result

    @staticmethod
    def visualize(
        prepared_data: pd.DataFrame,
        xlim_start: "dt | None" = None,
        xlim_end: "dt | None" = None,
    ) -> Figure:
        """
        Create stacked area chart of vehicles in revenue service over time.

        Args:
            prepared_data: DataFrame from analyze() with datetime index and vehicle type columns
            xlim_start: Left x-axis limit (default: REVENUE_SERVICE_PLOT_START)
            xlim_end: Right x-axis limit (default: REVENUE_SERVICE_PLOT_END)

        Returns:
            matplotlib Figure
        """
        from matplotlib.dates import DateFormatter, DayLocator, date2num

        configure_latex_plotting()

        if xlim_start is None:
            xlim_start = REVENUE_SERVICE_PLOT_START
        if xlim_end is None:
            xlim_end = REVENUE_SERVICE_PLOT_END

        fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH))

        # Order columns by VEHICLE_TYPE_ORDER
        columns = [vt for vt in VEHICLE_TYPE_ORDER if vt in prepared_data.columns]
        palette = sns.color_palette("Set2", n_colors=len(columns))

        stack_data = [prepared_data[col].values for col in columns]
        ax.stackplot(
            prepared_data.index,
            *stack_data,  # type: ignore[arg-type]
            labels=columns,
            colors=palette,
        )

        ax.set_ylabel(r"Vehicles in Revenue Service")
        ax.xaxis.set_major_locator(DayLocator())  # type: ignore[no-untyped-call]
        ax.xaxis.set_major_formatter(DateFormatter("%a"))  # type: ignore[no-untyped-call]
        plt.xticks(rotation=45, ha="right")

        ax.set_xlim(date2num(xlim_start), date2num(xlim_end))  # type: ignore[no-untyped-call]

        ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="upper left",
            ncols=3,
        )

        plt.tight_layout()
        return fig


class SchedulingEfficiencyAnalyzer(Analyzer):
    """
    Analyzer for evaluating scheduling efficiency: revenue vs. empty/deadhead kilometers
    and times per rotation, aggregated by scenario.

    Returns a per-rotation DataFrame. Use ``visualize()`` for a summary bar chart and
    ``visualize_histogram()`` for a debugging histogram of passenger-window durations.
    """

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.scenario_name": (
                "Display name for this scenario in combined output. "
                "Falls back to Scenario.name from the database."
            ),
        }

    def analyze(self, session: Session, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute per-rotation scheduling efficiency statistics.

        Returns:
            DataFrame with columns: scenario_name, rotation_name, vehicle_type,
            revenue_km, empty_km, revenue_time_h, empty_time_h,
            total_duration_h, pax_window_h.
        """
        scenario_name: str = params.get(f"{self.__class__.__name__}.scenario_name", "")
        if not scenario_name:
            scenario = session.query(Scenario).first()
            scenario_name = scenario.name if scenario else "Unknown"

        rows: List[Dict[str, Any]] = []
        rotations = session.query(Rotation).all()

        for rotation in rotations:
            trips = rotation.trips
            if not trips:
                continue

            pax_trips = [t for t in trips if t.trip_type == TripType.PASSENGER]
            empty_trips = [t for t in trips if t.trip_type == TripType.EMPTY]

            revenue_km = sum(t.route.distance / 1000 for t in pax_trips)
            empty_km = sum(t.route.distance / 1000 for t in empty_trips)

            revenue_time_h = sum(
                (t.arrival_time - t.departure_time).total_seconds() / 3600 for t in pax_trips
            )
            empty_time_h = sum(
                (t.arrival_time - t.departure_time).total_seconds() / 3600 for t in empty_trips
            )

            all_deps = [t.departure_time for t in trips]
            all_arrs = [t.arrival_time for t in trips]
            total_duration_h = (max(all_arrs) - min(all_deps)).total_seconds() / 3600

            if pax_trips:
                pax_deps = [t.departure_time for t in pax_trips]
                pax_arrs = [t.arrival_time for t in pax_trips]
                pax_window_h: float = (max(pax_arrs) - min(pax_deps)).total_seconds() / 3600
            else:
                pax_window_h = 0.0

            rows.append(
                {
                    "scenario_name": scenario_name,
                    "rotation_name": rotation.name,
                    "vehicle_type": rotation.vehicle_type.name_short,
                    "revenue_km": revenue_km,
                    "empty_km": empty_km,
                    "revenue_time_h": revenue_time_h,
                    "empty_time_h": empty_time_h,
                    "total_duration_h": total_duration_h,
                    "pax_window_h": pax_window_h,
                }
            )

        return pd.DataFrame(rows)

    # Ordered scenario codes and display names used in visualize()
    SCENARIO_ORDER: List[str] = ["OU", "DEP", "TERM"]
    SCENARIO_DISPLAY_NAMES: Dict[str, str] = {
        "OU": "Original Blocks",
        "DEP": "Depot Charging Only",
        "TERM": "Small Batteries, Termini",
    }

    @staticmethod
    def visualize(df: pd.DataFrame) -> Figure:
        """
        Create a 2-panel 100%-stacked bar chart:
        (a) revenue vs. empty/deadhead km share; (b) revenue vs. non-revenue time share.

        The time panel uses ``total_duration_h - revenue_time_h`` as the non-revenue
        portion, which includes both deadhead driving and idle/break time.

        Args:
            df: DataFrame from analyze() (may be concatenated across scenarios).

        Returns:
            matplotlib Figure
        """
        configure_latex_plotting()

        grouped = (
            df.groupby("scenario_name")[
                ["revenue_km", "empty_km", "revenue_time_h", "total_duration_h"]
            ]
            .sum()
            .reset_index()
        )

        # Enforce scenario order: OU first, then DEP, TERM, then any others
        order = SchedulingEfficiencyAnalyzer.SCENARIO_ORDER
        order_map = {code: i for i, code in enumerate(order)}
        grouped["_sort"] = grouped["scenario_name"].map(order_map).fillna(len(order))
        grouped = grouped.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

        # Non-revenue time = total rotation duration minus time in PASSENGER trips
        grouped["nonrevenue_time_h"] = grouped["total_duration_h"] - grouped["revenue_time_h"]

        # Percentage columns for distance
        grouped["total_km"] = grouped["revenue_km"] + grouped["empty_km"]
        grouped["rev_km_pct"] = grouped["revenue_km"] / grouped["total_km"] * 100
        grouped["empty_km_pct"] = grouped["empty_km"] / grouped["total_km"] * 100

        # Percentage columns for time (revenue vs. everything else within rotation span)
        grouped["rev_time_pct"] = grouped["revenue_time_h"] / grouped["total_duration_h"] * 100
        grouped["nonrev_time_pct"] = (
            grouped["nonrevenue_time_h"] / grouped["total_duration_h"] * 100
        )

        # Map scenario codes to display names, with line breaks after each word
        display_names = [
            "\n".join(SchedulingEfficiencyAnalyzer.SCENARIO_DISPLAY_NAMES.get(s, s).split())
            for s in grouped["scenario_name"]
        ]

        palette = sns.color_palette("Set2")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH / 1.5))

        x = np.arange(len(display_names))
        width = 0.5

        # --- Panel (a): distance ---
        rev_pct = grouped["rev_km_pct"].values
        empty_pct = grouped["empty_km_pct"].values

        ax1.bar(x, rev_pct, width, label="Revenue", color=palette[0])
        ax1.bar(x, empty_pct, width, bottom=rev_pct, label="Empty/Deadhead", color=palette[1])

        for i in range(len(x)):
            ax1.text(
                x[i],
                rev_pct[i] / 2,
                f"{rev_pct[i]:.1f}\\%",
                ha="center",
                va="center",
                fontsize=9,
            )
            ax1.text(
                x[i],
                rev_pct[i] + empty_pct[i] / 2,
                f"{empty_pct[i]:.1f}\\%",
                ha="center",
                va="center",
                fontsize=9,
            )

        ax1.set_xticks(x)
        ax1.set_xticklabels(display_names, rotation=60, ha="right")
        ax1.set_ylabel(r"Share of Distance [\%]")
        ax1.set_ylim(0, 100)
        ax1.legend(loc="lower right")

        # --- Panel (b): time ---
        rev_t_pct = grouped["rev_time_pct"].values
        nonrev_t_pct = grouped["nonrev_time_pct"].values

        ax2.bar(x, rev_t_pct, width, label="Revenue", color=palette[0])
        ax2.bar(
            x,
            nonrev_t_pct,
            width,
            bottom=rev_t_pct,
            label="Empty/Deadhead/Idle",
            color=palette[1],
        )

        for i in range(len(x)):
            ax2.text(
                x[i],
                rev_t_pct[i] / 2,
                f"{rev_t_pct[i]:.1f}\\%",
                ha="center",
                va="center",
                fontsize=9,
            )
            ax2.text(
                x[i],
                rev_t_pct[i] + nonrev_t_pct[i] / 2,
                f"{nonrev_t_pct[i]:.1f}\\%",
                ha="center",
                va="center",
                fontsize=9,
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels(display_names, rotation=60, ha="right")
        ax2.set_ylabel(r"Share of Time [\%]")
        ax2.set_ylim(0, 100)
        ax2.legend(loc="lower right")

        plt.tight_layout()
        return fig

    @staticmethod
    def visualize_histogram(df: pd.DataFrame) -> Figure:
        """
        Debug histogram of per-rotation passenger-window duration, one panel per scenario.

        Args:
            df: DataFrame from analyze() (may be concatenated across scenarios).

        Returns:
            matplotlib Figure
        """
        scenarios = df["scenario_name"].unique()
        n_scenarios = len(scenarios)

        fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 4), squeeze=False)

        for ax, scenario in zip(axes[0], scenarios):
            data = df[df["scenario_name"] == scenario]["pax_window_h"]
            ax.hist(data, bins=30)
            ax.set_title(str(scenario))
            ax.set_xlabel("Passenger window [h]")
            ax.set_ylabel("Count")

        plt.tight_layout()
        return fig


# ============================================================================
# Power Visualization Functions
# ============================================================================


def _plot_two_power_series(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    title: str,
    xlim_start: "dt | None" = None,
    xlim_end: "dt | None" = None,
) -> Figure:
    """Line chart comparing two power timeseries (kW → MW)."""
    from matplotlib.dates import DateFormatter, DayLocator, date2num

    configure_latex_plotting()

    if xlim_start is None:
        xlim_start = REVENUE_SERVICE_PLOT_START
    if xlim_end is None:
        xlim_end = REVENUE_SERVICE_PLOT_END

    fig, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH))
    palette = sns.color_palette("Set2")

    ax.plot(df_a["time"], df_a["power"] / 1000, color=palette[0], label=label_a, alpha=0.8)
    ax.plot(df_b["time"], df_b["power"] / 1000, color=palette[1], label=label_b, alpha=0.8)

    ax.set_ylabel(r"Power [MW]")
    ax.set_title(title)
    ax.xaxis.set_major_locator(DayLocator())  # type: ignore[no-untyped-call]
    ax.xaxis.set_major_formatter(DateFormatter("%a"))  # type: ignore[no-untyped-call]
    plt.xticks(rotation=45, ha="right")

    ax.set_xlim(date2num(xlim_start), date2num(xlim_end))  # type: ignore[no-untyped-call]

    ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper left", ncols=2)
    plt.tight_layout()
    return fig


def visualize_power_comparison(
    df_none: pd.DataFrame,
    df_even: pd.DataFrame,
    depot_name: str = "BFI",
    xlim_start: "dt | None" = None,
    xlim_end: "dt | None" = None,
) -> Figure:
    """Line chart comparing power consumption between NONE and EVEN smart charging."""
    return _plot_two_power_series(
        df_none,
        df_even,
        "Without Smart Charging",
        "With Smart Charging (EVEN)",
        f"Power Consumption -- {depot_name}",
        xlim_start,
        xlim_end,
    )


def visualize_depot_and_terminus_power(
    df_depots: pd.DataFrame,
    df_termini: pd.DataFrame,
    xlim_start: "dt | None" = None,
    xlim_end: "dt | None" = None,
) -> Figure:
    """Line chart showing total depot power and total terminus power for the EVEN case."""
    return _plot_two_power_series(
        df_depots,
        df_termini,
        "Depots (total)",
        "Termini (total)",
        "Power Consumption -- Smart Charging (EVEN)",
        xlim_start,
        xlim_end,
    )


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


def _select_representative_vehicle(
    session: Session,
    day_start: dt,
    day_end: dt,
    required_event_type: EventType,
    max_soc_threshold: "float | None" = None,
) -> int:
    """
    Select a representative vehicle by event type, SoC filtering, and highest delta_soc.

    Args:
        session: SQLAlchemy session
        day_start: Window start (timezone-aware)
        day_end: Window end (timezone-aware)
        required_event_type: EventType that candidate vehicles must have
        max_soc_threshold: If set, exclude candidates whose soc_end reaches this value or above

    Returns:
        vehicle_id of the selected vehicle
    """
    # Find candidates: vehicles that have the required event type overlapping the window
    qualifying_events = (
        session.query(Event)
        .filter(
            Event.time_start < day_end,
            Event.time_end > day_start,
            Event.event_type == required_event_type,
        )
        .all()
    )
    candidate_vehicle_ids = {e.vehicle_id for e in qualifying_events}

    if not candidate_vehicle_ids:
        raise ValueError(f"No vehicles with {required_event_type.name} found in the given window.")

    # Apply max_soc_threshold filter: drop vehicles where any event hits the ceiling
    if max_soc_threshold is not None:
        filtered_ids = set()
        for vid in candidate_vehicle_ids:
            all_events = (
                session.query(Event)
                .filter(
                    Event.vehicle_id == vid,
                    Event.time_start < day_end,
                    Event.time_end > day_start,
                )
                .all()
            )
            if all(e.soc_end < max_soc_threshold for e in all_events):
                filtered_ids.add(vid)
        candidate_vehicle_ids = filtered_ids

    if not candidate_vehicle_ids:
        raise ValueError(
            f"No vehicles with {required_event_type.name} and max SoC < "
            f"{max_soc_threshold} found in the given window."
        )

    # Pick the vehicle with the highest total |delta_soc|
    best_vehicle_id = None
    best_delta_soc = -1.0
    for vid in candidate_vehicle_ids:
        events = (
            session.query(Event)
            .filter(
                Event.vehicle_id == vid,
                Event.time_start < day_end,
                Event.time_end > day_start,
            )
            .all()
        )
        total_delta = sum(abs(e.soc_end - e.soc_start) for e in events)
        if total_delta > best_delta_soc:
            best_delta_soc = total_delta
            best_vehicle_id = vid

    assert best_vehicle_id is not None
    return best_vehicle_id


def _build_vehicle_soc(
    session: Session,
    day_start: dt,
    day_end: dt,
    tz: Any,
    required_event_type: EventType,
    max_soc_threshold: "float | None" = None,
) -> Tuple[pd.DataFrame, List[Tuple[str, dt, dt]]]:
    """
    Select a representative vehicle and build its SoC timeseries + event spans.

    Uses ``eflips.eval.output.prepare.vehicle_soc`` for the SoC timeseries to avoid
    duplicating the event-to-timeseries conversion logic.

    Args:
        session: SQLAlchemy session
        day_start: Window start (timezone-aware)
        day_end: Window end (timezone-aware)
        tz: pytz timezone for display
        required_event_type: EventType that candidate vehicles must have
        max_soc_threshold: If set, exclude candidates whose soc_end reaches this value or above

    Returns:
        (soc_df, event_spans) — DataFrame with 'time'/'soc' columns, list of shading spans
    """
    from zoneinfo import ZoneInfo

    from eflips.eval.output.prepare import vehicle_soc as eval_vehicle_soc

    vehicle_id = _select_representative_vehicle(
        session, day_start, day_end, required_event_type, max_soc_threshold
    )

    # Reuse eflips.eval for SoC timeseries construction
    soc_df, _descriptions = eval_vehicle_soc(vehicle_id, session, ZoneInfo(str(tz)))

    # Build event spans from events in the display window
    # (vehicle_soc returns all events; we keep window-filtered spans for clean shading)
    events = (
        session.query(Event)
        .filter(
            Event.vehicle_id == vehicle_id,
            Event.time_start < day_end,
            Event.time_end > day_start,
        )
        .order_by(Event.time_start)
        .all()
    )

    label_map = {
        EventType.DRIVING: "Driving",
        EventType.CHARGING_DEPOT: "Depot Charging",
        EventType.CHARGING_OPPORTUNITY: "Terminus Charging",
    }
    event_spans: List[Tuple[str, dt, dt]] = []
    for event in events:
        label = label_map.get(event.event_type)
        if label is not None:
            event_spans.append(
                (label, event.time_start.astimezone(tz), event.time_end.astimezone(tz))
            )

    return soc_df, event_spans


class RepresentativeVehicleSocAnalyzer(Analyzer):
    """
    Analyzer that plots a representative vehicle's SoC over one service day (3AM–3AM).

    In "terminus" mode (default): selects the vehicle with opportunity charging and highest
    total delta_soc. In "depot" mode: selects the vehicle with depot charging, highest
    delta_soc, and whose SoC never reaches the configured threshold.
    """

    def __init__(self, code_version: str = "v1.0.1", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.day_start": (
                "Start of the 24h window. Default: 2025-06-18 03:00 (a Wednesday)."
            ),
            f"{cls.__name__}.day_end": ("End of the 24h window. Default: 2025-06-19 03:00."),
            f"{cls.__name__}.mode": (
                '"terminus" (default) — vehicle with CHARGING_OPPORTUNITY; '
                '"depot" — vehicle with CHARGING_DEPOT.'
            ),
            f"{cls.__name__}.max_soc_threshold": (
                "Depot mode only: exclude vehicles whose soc_end reaches this value. Default: 0.99."
            ),
        }

    def analyze(
        self, session: Session, params: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, List[Tuple[str, dt, dt]], dt, dt]:
        """
        Select representative vehicle and build SoC timeseries + event spans.

        Returns:
            (soc_df, event_spans, day_start, day_end)
        """
        import pytz

        day_start = params.get(f"{self.__class__.__name__}.day_start", dt(2025, 6, 18, 3, 0))
        day_end = params.get(f"{self.__class__.__name__}.day_end", dt(2025, 6, 19, 3, 0))
        mode = params.get(f"{self.__class__.__name__}.mode", "terminus")

        tz = pytz.timezone("Europe/Berlin")
        if day_start.tzinfo is None:
            day_start = tz.localize(day_start)
        if day_end.tzinfo is None:
            day_end = tz.localize(day_end)

        if mode == "depot":
            required_event_type = EventType.CHARGING_DEPOT
            max_soc_threshold: "float | None" = params.get(
                f"{self.__class__.__name__}.max_soc_threshold", 0.99
            )
        else:
            required_event_type = EventType.CHARGING_OPPORTUNITY
            max_soc_threshold = None

        soc_df, event_spans = _build_vehicle_soc(
            session, day_start, day_end, tz, required_event_type, max_soc_threshold
        )
        return soc_df, event_spans, day_start, day_end

    @staticmethod
    def visualize(
        prepared_data: pd.DataFrame,
        event_spans: List[Tuple[str, dt, dt]],
        xlim_start: "dt | None" = None,
        xlim_end: "dt | None" = None,
    ) -> Figure:
        """
        Create publication-quality SoC day plot with shaded event regions.

        Args:
            prepared_data: DataFrame with 'time' and 'soc' columns.
            event_spans: List of (label, start, end) tuples for background shading.
            xlim_start: Left x-axis limit for display window.
            xlim_end: Right x-axis limit for display window.

        Returns:
            matplotlib Figure
        """
        from matplotlib.dates import DateFormatter, HourLocator, date2num

        configure_latex_plotting()

        fig, ax = plt.subplots(figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH))

        palette = sns.color_palette("Set2")
        span_colors = {
            "Driving": palette[0],
            "Depot Charging": palette[1],
            "Terminus Charging": palette[2],
        }

        # Track which labels have been drawn (avoid duplicate legend entries)
        drawn_labels: set[str] = set()

        for label, t_start, t_end in event_spans:
            kw: Dict[str, Any] = {"alpha": 0.2, "color": span_colors[label]}
            if label not in drawn_labels:
                kw["label"] = label
                drawn_labels.add(label)
            ax.axvspan(t_start, t_end, **kw)  # type: ignore[arg-type]

        # SoC line
        ax.plot(
            prepared_data["time"],
            prepared_data["soc"] * 100,
            color="black",
            linewidth=1.2,
            label="SoC",
        )

        ax.set_ylabel(r"State of Charge [\%]")
        ax.set_ylim(0, 105)
        ax.set_title("Representative Vehicle -- Service Day")

        import pytz

        ax.xaxis.set_major_locator(HourLocator(interval=2))  # type: ignore[no-untyped-call]
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=pytz.timezone("Europe/Berlin")))  # type: ignore[no-untyped-call]
        plt.xticks(rotation=45, ha="right")

        if xlim_start is not None and xlim_end is not None:
            ax.set_xlim(date2num(xlim_start), date2num(xlim_end))  # type: ignore[no-untyped-call]

        ax.legend(
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="upper left",
            ncols=4,
        )

        plt.tight_layout()
        return fig


# ============================================================================
# Scenario Comparison Analysis
# ============================================================================


def _peak_occupancy(
    session: Session,
    area_ids: List[int] | None = None,
    station_ids: List[int] | None = None,
) -> int:
    """
    Compute peak total occupancy using ``eflips.eval.output.prepare.power_and_occupancy``.

    Exactly one of *area_ids* or *station_ids* must be non-empty.

    Returns:
        Peak value of the ``occupancy_total`` column.
    """
    from eflips.eval.output.prepare import power_and_occupancy

    if station_ids:
        df = power_and_occupancy(area_id=[], session=session, station_id=station_ids)
    elif area_ids:
        df = power_and_occupancy(area_id=area_ids, session=session)
    else:
        return 0

    if df.empty or "occupancy_total" not in df.columns:
        return 0

    return int(df["occupancy_total"].max())


class ScenarioComparisonAnalyzer(Analyzer):
    """
    Analyzer that extracts fleet and charging infrastructure summary for a single scenario.

    Call once per scenario database, then use ``merge_scenario_comparisons()`` to combine
    the results and compute additional-vehicle metrics relative to the DIESEL baseline.
    """

    # Ordered scenario codes used in merge / visualize
    SCENARIO_ORDER: List[str] = ["OU", "DEP", "TERM", "DIESEL"]

    def __init__(self, code_version: str = "v1.0.0", cache_enabled: bool = True):
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.scenario_name": (
                "Display name for this scenario (e.g. 'OU', 'DEP', 'TERM', 'DIESEL')."
            ),
        }

    def analyze(self, session: Session, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract fleet size and charging infrastructure metrics for one scenario.

        Returns a single-row DataFrame with columns:
            scenario_name, vehicles_EN, vehicles_GN, vehicles_DD, total_vehicles,
            terminus_chargers_utilized, electrified_termini, depot_chargers,
            terminus_charger_power_kw, depot_charger_power_kw
        """
        scenario_name: str = params.get(f"{self.__class__.__name__}.scenario_name", "")
        if not scenario_name:
            scenario = session.query(Scenario).first()
            scenario_name = scenario.name if scenario else "Unknown"

        # --- Vehicles per type ---
        vehicle_counts = (
            session.query(VehicleType.name_short, func.count(Vehicle.id))
            .select_from(Vehicle)
            .join(VehicleType, Vehicle.vehicle_type_id == VehicleType.id)
            .group_by(VehicleType.name_short)
            .all()
        )
        vc: Dict[str, int] = {name_short: count for name_short, count in vehicle_counts}

        vehicles_en = vc.get("EN", 0)
        vehicles_gn = vc.get("GN", 0)
        vehicles_dd = vc.get("DD", 0)
        total_vehicles = sum(vc.values())

        # --- Depot station IDs (used to distinguish depot vs. terminus) ---
        depot_station_ids = {d.station_id for d in session.query(Depot).all()}

        # --- Electrified terminus stations ---
        et_q = session.query(Station).filter(
            Station.is_electrified == True,  # noqa: E712
            Station.charge_type == ChargeType.OPPORTUNITY,
        )
        electrified_terminus_stations = et_q.all()
        electrified_termini: int = len(electrified_terminus_stations)
        electrified_terminus_ids = [s.id for s in electrified_terminus_stations]

        # --- Terminus chargers utilized (sum of per-station peak occupancy) ---
        terminus_station_peaks: List[Dict[str, Any]] = []
        for s in electrified_terminus_stations:
            peak = _peak_occupancy(session, station_ids=[s.id])
            terminus_station_peaks.append(
                {
                    "station_id": s.id,
                    "name_short": s.name_short,
                    "name": s.name,
                    "peak_occupancy_total": peak,
                }
            )
        terminus_chargers_utilized: int = sum(
            r["peak_occupancy_total"] for r in terminus_station_peaks
        )

        # --- Depot chargers (peak total occupancy in charging areas only) ---
        # Only include depot areas that have an associated Process with electric_power > 0
        charging_area_ids = [
            a.id
            for depot in session.query(Depot).all()
            for a in depot.areas
            if any(p.electric_power is not None and p.electric_power > 0 for p in a.processes)
        ]
        depot_chargers: int = sum(
            _peak_occupancy(session, area_ids=[area_id]) for area_id in charging_area_ids
        )

        # --- Terminus charger power (from Station.power_per_charger) ---
        tp_q = session.query(Station.power_per_charger).filter(
            Station.is_electrified == True,  # noqa: E712
            Station.power_per_charger.isnot(None),
        )
        if depot_station_ids:
            tp_q = tp_q.filter(~Station.id.in_(depot_station_ids))
        terminus_power_rows = tp_q.distinct().all()

        terminus_charger_power_kw: float | None
        if len(terminus_power_rows) == 1:
            terminus_charger_power_kw = float(terminus_power_rows[0][0])
        elif len(terminus_power_rows) > 1:
            terminus_charger_power_kw = float(
                max(r[0] for r in terminus_power_rows if r[0] is not None)
            )
        else:
            terminus_charger_power_kw = None

        # --- Depot charger power (from Process.electric_power linked to depot areas) ---
        depot_power_rows = (
            session.query(Process.electric_power)
            .join(Process.areas)
            .join(Depot, Area.depot_id == Depot.id)
            .filter(Process.electric_power.isnot(None), Process.electric_power > 0)
            .distinct()
            .all()
        )
        depot_charger_power_kw: float | None
        if len(depot_power_rows) == 1:
            depot_charger_power_kw = float(depot_power_rows[0][0])
        elif len(depot_power_rows) > 1:
            depot_charger_power_kw = float(max(r[0] for r in depot_power_rows))
        else:
            depot_charger_power_kw = None

        return pd.DataFrame(
            [
                {
                    "scenario_name": scenario_name,
                    "vehicles_EN": vehicles_en,
                    "vehicles_GN": vehicles_gn,
                    "vehicles_DD": vehicles_dd,
                    "total_vehicles": total_vehicles,
                    "terminus_chargers_utilized": terminus_chargers_utilized,
                    "electrified_termini": electrified_termini,
                    "depot_chargers": depot_chargers,
                    "terminus_charger_power_kw": terminus_charger_power_kw,
                    "depot_charger_power_kw": depot_charger_power_kw,
                }
            ]
        )

    @staticmethod
    def visualize(df: pd.DataFrame) -> Figure:
        """
        Render the comparison DataFrame as a publication-quality matplotlib table.

        Args:
            df: Merged DataFrame from ``merge_scenario_comparisons()``.

        Returns:
            matplotlib Figure
        """
        configure_latex_plotting()

        # Prepare display table
        display_df = df.copy()

        col_map = {
            "scenario_name": "Scenario",
            "vehicles_EN": "Single Decker",
            "vehicles_GN": "Articulated",
            "vehicles_DD": "Double Decker",
            "total_vehicles": "Total Vehicles",
            "terminus_chargers_utilized": "Terminus Chargers",
            "electrified_termini": "Electrified Termini",
            "depot_chargers": "Depot Chargers",
            "terminus_charger_power_kw": "Terminus Power [kW]",
            "depot_charger_power_kw": "Depot Power [kW]",
            "additional_vehicles": "Add. Vehicles",
            "additional_vehicles_pct": r"Add. Vehicles [\%]",
        }
        display_cols = [c for c in col_map if c in display_df.columns]
        display_df = display_df[display_cols].rename(columns=col_map)

        # Format numeric columns
        for col in display_df.columns:
            if col in ("Terminus Power [kW]", "Depot Power [kW]"):
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.0f}" if pd.notna(x) else "--"
                )
            elif col == r"Add. Vehicles [\%]":
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:+.1f}\\%" if pd.notna(x) else "--"
                )
            elif col == "Add. Vehicles":
                display_df[col] = display_df[col].apply(
                    lambda x: f"{int(x):+d}" if pd.notna(x) else "--"
                )

        fig, ax = plt.subplots(figsize=(PLOT_WIDTH_INCH, 0.4 * (len(display_df) + 1.5)))
        ax.axis("off")

        table = ax.table(
            cellText=display_df.values.tolist(),
            colLabels=display_df.columns.tolist(),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(list(range(len(display_df.columns))))

        # Style header row
        for j in range(len(display_df.columns)):
            cell = table[0, j]
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9e2f3")

        plt.tight_layout()
        return fig


def merge_scenario_comparisons(scenario_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge single-scenario comparison rows and compute additional-vehicle metrics.

    Args:
        scenario_dfs: List of single-row DataFrames from
                      ``ScenarioComparisonAnalyzer.analyze()``.

    Returns:
        Combined DataFrame with ``additional_vehicles`` and ``additional_vehicles_pct``
        columns relative to the DIESEL scenario.
    """
    merged = pd.concat(scenario_dfs, ignore_index=True)

    diesel_mask = merged["scenario_name"] == "DIESEL"
    if diesel_mask.any():
        diesel_total = int(merged.loc[diesel_mask, "total_vehicles"].iloc[0])
        merged["additional_vehicles"] = merged["total_vehicles"] - diesel_total
        merged["additional_vehicles_pct"] = (
            merged["additional_vehicles"] / diesel_total * 100
        ).round(1)
    else:
        merged["additional_vehicles"] = None
        merged["additional_vehicles_pct"] = None

    # Sort by standard order
    order = {name: i for i, name in enumerate(ScenarioComparisonAnalyzer.SCENARIO_ORDER)}
    merged["_sort"] = merged["scenario_name"].map(order).fillna(len(order))
    merged = merged.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    return merged
