"""
Visualize the vehicle scheduling connection graph for one BVG vehicle type (default: double deckers).

Builds the same trip connection graph that ``VehicleScheduling`` hands to eflips-opt, restricted to
the passenger trips of one vehicle type departing on Tuesday 2025-06-17 between 03:00 and 03:00 the next
day. Nodes are laid out on a time × station grid (x: departure time, y: departure station). Two
full-page figures are produced:

* an overview of the whole graph, with a red rectangle marking the zoomed-in region, and
* a zoom into the busiest part of the largest connected component, with trip labels.

Usage::

    PYTHONPATH=. poetry run python scripts/plot_scheduling_graph.py [--vehicle-type GN] [path/to/database.db]
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import eflips.model
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from eflips.model import Rotation, Route, Trip, TripType, VehicleType
from eflips.opt.scheduling import create_graph
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
from sqlalchemy.orm import Session, joinedload

from eflips.x.flows.bvg import save_plot_to_files_in_output_dir
from eflips.x.framework import PipelineStep
from eflips.x.steps.analyzers.bvg_tools import (
    PLOT_HEIGHT_PT,
    PLOT_WIDTH_PT,
    configure_latex_plotting,
)

DEFAULT_DB = (
    PipelineStep.find_project_root()
    / "data"
    / "cache"
    / "bvg"
    / "common"
    / "step_007_CalculateConsumptionScaling.db"
)

TZ = ZoneInfo("Europe/Berlin")
WINDOW_START = datetime(2025, 6, 17, 3, 0, tzinfo=TZ)  # Tuesday 03:00
WINDOW_END = WINDOW_START + timedelta(days=1)

FULL_PAGE = (PLOT_WIDTH_PT / 72.0, 3 * PLOT_HEIGHT_PT / 72.0)

# Zoom box selection. The hub is the station in the largest component served by the most lines.
# The box spans its row ± ZOOM_ROW_RADIUS and the time window (at most ZOOM_WINDOW long, holding at
# most ZOOM_MAX_NODES trips) with the most connections.
ZOOM_ROW_RADIUS = 1
ZOOM_WINDOW = timedelta(minutes=90)
ZOOM_MAX_NODES = 36
# Manual override: (start, end, first row, last row) or None for automatic selection
ZOOM_OVERRIDE: Optional[Tuple[datetime, datetime, int, int]] = None

ACCENT = "#1f4e79"
MUTED = "#a8a8a8"
FALLBACK = "#c0392b"
BOX = "#d62728"
BAND = "#f3f3f3"
LABEL_FONTSIZE = 5.5

Pos = Dict[int, Tuple[float, float]]


@dataclass
class ZoomBox:
    x0: float  # hours after WINDOW_START
    x1: float
    row0: int
    row1: int

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.row0 <= y <= self.row1

    @property
    def rows(self) -> range:
        return range(self.row0, self.row1 + 1)


def hours(t: datetime) -> float:
    return (t - WINDOW_START).total_seconds() / 3600


def format_hour(x: float, _: object = None) -> str:
    return f"{WINDOW_START + timedelta(hours=x):%H:%M}"


def load_trips(session: Session, vehicle_type: str) -> List[Trip]:
    trips = (
        session.query(Trip)
        .join(Rotation)
        .join(VehicleType)
        .filter(VehicleType.name_short == vehicle_type)
        .filter(Trip.trip_type == TripType.PASSENGER)
        .options(
            joinedload(Trip.route).joinedload(Route.line),
            joinedload(Trip.route).joinedload(Route.departure_station),
            joinedload(Trip.route).joinedload(Route.arrival_station),
        )
        .all()
    )
    # Filtering in Python avoids SQLite comparing timezone-aware datetime strings
    return [t for t in trips if WINDOW_START <= t.departure_time.astimezone(TZ) < WINDOW_END]


def build_graph(trips: List[Trip]) -> nx.DiGraph:
    """Build the connection graph like ``VehicleScheduling`` does and attach trip details."""
    graph = create_graph(trips, minimum_break_time=timedelta(minutes=0))
    for trip in trips:
        route = trip.route
        line = route.line.name if route.line is not None else route.name.split(" ")[0]
        graph.nodes[trip.id].update(
            line=line,
            dep_station=route.departure_station.name,
            arr_station=route.arrival_station.name,
            dep_station_id=route.departure_station_id,
            arr_station_id=route.arrival_station_id,
            dep=trip.departure_time.astimezone(TZ),
            arr=trip.arrival_time.astimezone(TZ),
        )
    return graph


def station_rows(graph: nx.DiGraph) -> Dict[int, int]:
    """
    Assign a row to every departure station, ordering them so that stations connected by many
    trips end up next to each other.
    """
    station_graph = nx.Graph()
    for _, data in graph.nodes(data=True):
        a, b = data["dep_station_id"], data["arr_station_id"]
        station_graph.add_nodes_from([a, b])
        if a != b:
            weight = station_graph.get_edge_data(a, b, {"weight": 0})["weight"]
            station_graph.add_edge(a, b, weight=weight + 1)

    order: List[int] = []
    for component in sorted(nx.connected_components(station_graph), key=len, reverse=True):
        sub = station_graph.subgraph(component)
        if len(sub) <= 2:
            order.extend(sorted(sub.nodes))
        else:
            order.extend(nx.spectral_ordering(sub, weight="weight", seed=0))

    departure_stations = {d["dep_station_id"] for _, d in graph.nodes(data=True)}
    order = [s for s in order if s in departure_stations]
    return {station: row for row, station in enumerate(order)}


def positions(graph: nx.DiGraph, rows: Dict[int, int]) -> Pos:
    return {
        n: (hours(d["dep"]), float(rows[d["dep_station_id"]])) for n, d in graph.nodes(data=True)
    }


def choose_zoom_box(graph: nx.DiGraph, pos: Pos, rows: Dict[int, int]) -> ZoomBox:
    if ZOOM_OVERRIDE is not None:
        t0, t1, r0, r1 = ZOOM_OVERRIDE
        return ZoomBox(hours(t0), hours(t1), r0, r1)

    largest = max(nx.weakly_connected_components(graph), key=len)
    departures = Counter(graph.nodes[n]["dep_station_id"] for n in largest)
    lines: Dict[int, set[str]] = {}
    for n in largest:
        lines.setdefault(graph.nodes[n]["dep_station_id"], set()).add(graph.nodes[n]["line"])
    hub = max(departures, key=lambda s: (len(lines[s]), departures[s]))
    row0, row1 = rows[hub] - ZOOM_ROW_RADIUS, rows[hub] + ZOOM_ROW_RADIUS

    window = ZOOM_WINDOW.total_seconds() / 3600
    while True:
        best: Optional[Tuple[int, ZoomBox]] = None
        for start in np.arange(0, 24 - window, 1 / 12):
            box = ZoomBox(float(start), float(start) + window, row0, row1)
            inside = {n for n in graph.nodes if box.contains(*pos[n])}
            if len(inside) > ZOOM_MAX_NODES:
                continue
            score = sum(1 for u, v in graph.edges if u in inside and v in inside)
            if best is None or score > best[0]:
                best = (score, box)
        if best is not None:
            break
        window -= 1 / 6

    # Drop rows at the edges of the box without any trips in the chosen window
    box = best[1]
    occupied = [int(y) for x, y in pos.values() if box.contains(x, y)]
    return ZoomBox(box.x0, box.x1, min(occupied), max(occupied))


def plot_overview(graph: nx.DiGraph, pos: Pos, rows: Dict[int, int], box: ZoomBox) -> Figure:
    fig, ax = plt.subplots(figsize=FULL_PAGE, layout="constrained")
    largest = max(nx.weakly_connected_components(graph), key=len)

    for in_largest, color, alpha in ((False, MUTED, 0.35), (True, ACCENT, 0.18)):
        segments = [
            (pos[u], pos[v])
            for u, v, c in graph.edges(data="color")
            if c != "red" and (u in largest) == in_largest
        ]
        ax.add_collection(
            LineCollection(segments, colors=color, linewidths=0.15, alpha=alpha, rasterized=True)
        )
    fallback = [(pos[u], pos[v]) for u, v, c in graph.edges(data="color") if c == "red"]
    ax.add_collection(
        LineCollection(fallback, colors=FALLBACK, linewidths=0.3, alpha=0.7, rasterized=True)
    )

    nodes = list(graph.nodes)
    xy = np.array([pos[n] for n in nodes])
    colors = [ACCENT if n in largest else MUTED for n in nodes]
    ax.scatter(xy[:, 0], xy[:, 1], s=0.8, c=colors, linewidths=0, rasterized=True, zorder=3)

    ax.add_patch(
        Rectangle(
            (box.x0 - 0.15, box.row0 - 0.8),
            box.x1 - box.x0 + 0.3,
            box.row1 - box.row0 + 1.6,
            fill=False,
            edgecolor=BOX,
            linewidth=1.0,
            zorder=4,
        )
    )

    ax.set_xlim(0, 24)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.xaxis.set_major_locator(MultipleLocator(3))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FuncFormatter(format_hour))
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Departure time")
    ax.set_ylabel(f"Departure station ({len(rows)} termini)")

    handles = [
        Line2D([], [], color=ACCENT, marker="o", markersize=2, linewidth=0.6),
        Line2D([], [], color=MUTED, marker="o", markersize=2, linewidth=0.6),
        Line2D([], [], color=FALLBACK, linewidth=0.8),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=BOX),
    ]
    labels = [
        f"Largest component ({len(largest)} trips)",
        "Other components",
        "Fallback connection (30--60 min)",
        "Detail view",
    ]
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    return fig


def _trip_label(data: Dict[str, Any]) -> str:
    return "\n".join(
        [
            rf"\textbf{{{data['line']}}}\quad {data['dep']:%H:%M}--{data['arr']:%H:%M}",
            rf"{data['dep_station']} $\rightarrow$",
            data["arr_station"],
        ]
    )


def _pack_lanes(
    nodes: List[int], pos: Pos, widths: Dict[int, float], gap: float
) -> Tuple[Dict[int, int], int]:
    """Greedily assign each node (sorted by x) to the first lane where its label fits."""
    lanes: Dict[int, int] = {}
    lane_end: List[float] = []
    for n in nodes:
        x = pos[n][0]
        for i, end in enumerate(lane_end):
            if x > end:
                lanes[n] = i
                lane_end[i] = x + widths[n] + gap
                break
        else:
            lanes[n] = len(lane_end)
            lane_end.append(x + widths[n] + gap)
    return lanes, max(len(lane_end), 1)


def plot_zoom(graph: nx.DiGraph, pos: Pos, station_names: Dict[int, str], box: ZoomBox) -> Figure:
    fig, ax = plt.subplots(figsize=FULL_PAGE, layout="constrained")
    inside = sorted((n for n in graph.nodes if box.contains(*pos[n])), key=lambda n: pos[n][0])

    # Decorate first, so the axes has its final size when the labels are measured
    station_labels = [station_names[r] for r in box.rows]
    ax.set_yticks(list(range(len(box.rows))), labels=station_labels, rotation=90, va="center")
    ax.set_ylim(len(box.rows) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(FuncFormatter(format_hour))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Departure time")
    handles = [
        Line2D([], [], color=ACCENT, linewidth=0.8),
        Line2D([], [], color=FALLBACK, linewidth=0.8),
        Line2D([], [], color=MUTED, linewidth=0.8),
    ]
    labels = ["Connection (0--30 min)", "Fallback (30--60 min)", "Leaving the view"]
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.canvas.draw()
    fig.set_layout_engine("none")

    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    axes_width = ax.get_window_extent(renderer).width
    label_px: Dict[int, float] = {}
    for n in inside:
        text = ax.text(0, 0, _trip_label(graph.nodes[n]), fontsize=LABEL_FONTSIZE)
        label_px[n] = text.get_window_extent(renderer).width + 4
        text.remove()

    # Choose the right x limit so that every label ends inside the axes
    x_lo = box.x0 - 0.02 * (box.x1 - box.x0)
    x_hi = box.x1 + 0.02 * (box.x1 - box.x0)
    for n in inside:
        f = label_px[n] / axes_width
        x_hi = max(x_hi, (pos[n][0] - f * x_lo) / (1 - f))
    per_px = (x_hi - x_lo) / axes_width
    widths = {n: label_px[n] * per_px for n in inside}
    ax.set_xlim(x_lo, x_hi)

    # Stack the lanes of all rows from top to bottom, with a gap between station rows
    row_gap = 1.0
    zpos: Pos = {}
    row_extent: Dict[int, Tuple[float, float]] = {}
    y = 0.0
    for row in box.rows:
        row_nodes = [n for n in inside if int(pos[n][1]) == row]
        lanes, n_lanes = _pack_lanes(row_nodes, pos, widths, gap=4 * per_px)
        for n in row_nodes:
            zpos[n] = (pos[n][0], y + lanes[n] + 0.5)
        row_extent[row] = (y, y + n_lanes)
        y += n_lanes + row_gap
    bottom = y - row_gap
    ax.set_ylim(bottom + 0.4, -0.4)
    ax.set_yticks(
        [sum(row_extent[r]) / 2 for r in box.rows],
        labels=station_labels,
        rotation=90,
        va="center",
    )

    for i, row in enumerate(box.rows):
        if i % 2 == 0:
            top, low = row_extent[row]
            ax.axhspan(top - 0.3, low + 0.3, color=BAND, linewidth=0, zorder=0)

    # Connections to trips outside the box are hinted at with short, faded stubs
    stubs = []
    for u, v in graph.edges:
        if (u in zpos) == (v in zpos):
            continue
        inner, outer = (u, v) if u in zpos else (v, u)
        ox, orow = pos[outer]
        if orow < box.row0:
            oy = -3.0
        elif orow > box.row1:
            oy = bottom + 3.0
        else:
            oy = sum(row_extent[int(orow)]) / 2
        ix, iy = zpos[inner]
        dx, dy = ox - ix, oy - iy
        t = min(1.0, 12 * per_px / max(abs(dx), 1e-9), 0.8 / max(abs(dy), 1e-9))
        stubs.append(((ix, iy), (ix + t * dx, iy + t * dy)))
    ax.add_collection(LineCollection(stubs, colors=MUTED, linewidths=0.5, zorder=1))

    for u, v, data in graph.edges(data=True):
        if u not in zpos or v not in zpos:
            continue
        ax.add_patch(
            FancyArrowPatch(
                zpos[u],
                zpos[v],
                arrowstyle="-|>,head_length=2.0,head_width=0.9",
                connectionstyle="arc3,rad=0.1",
                color=FALLBACK if data["color"] == "red" else ACCENT,
                alpha=0.4,
                linewidth=0.4,
                shrinkA=2,
                shrinkB=2,
                zorder=2,
            )
        )

    xy = np.array([zpos[n] for n in inside])
    ax.scatter(xy[:, 0], xy[:, 1], s=7, color=ACCENT, linewidths=0, zorder=3)
    for n in inside:
        x, y = zpos[n]
        ax.text(
            x + 2.5 * per_px,
            y,
            _trip_label(graph.nodes[n]),
            fontsize=LABEL_FONTSIZE,
            va="center",
            ha="left",
            linespacing=1.05,
            zorder=4,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.75),
        )

    # Keep the time axis to the box itself; the label overhang needs no ticks
    ax.spines["bottom"].set_bounds(box.x0, box.x1)
    ax.set_xticks([t for t in ax.get_xticks() if box.x0 <= t <= box.x1])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", nargs="?", type=Path, default=DEFAULT_DB)
    parser.add_argument("--vehicle-type", default="DD", help="VehicleType.name_short (DD, EN, GN)")
    args = parser.parse_args()

    engine = eflips.model.create_engine(f"sqlite:////{args.database.absolute().as_posix()}")
    with Session(engine) as session:
        graph = build_graph(load_trips(session, args.vehicle_type))
    engine.dispose()

    components = sorted(nx.weakly_connected_components(graph), key=len, reverse=True)
    print(
        f"{graph.number_of_nodes()} trips, {graph.number_of_edges()} connections, "
        f"{len(components)} components (largest: {len(components[0])} trips)"
    )

    rows = station_rows(graph)
    station_names = {
        rows[d["dep_station_id"]]: d["dep_station"] for _, d in graph.nodes(data=True)
    }
    pos = positions(graph, rows)
    box = choose_zoom_box(graph, pos, rows)
    n_inside = sum(1 for p in pos.values() if box.contains(*p))
    print(
        f"Detail view: {format_hour(box.x0)}-{format_hour(box.x1)}, "
        f"{[station_names[r] for r in box.rows]}, {n_inside} trips"
    )

    configure_latex_plotting()
    save_plot_to_files_in_output_dir(
        plot_overview(graph, pos, rows, box),
        f"scheduling_graph_{args.vehicle_type.lower()}_overview",
    )
    save_plot_to_files_in_output_dir(
        plot_zoom(graph, pos, station_names, box),
        f"scheduling_graph_{args.vehicle_type.lower()}_zoom",
    )


if __name__ == "__main__":
    main()
