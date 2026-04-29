#!/usr/bin/env python3
"""
Analyse benchmark results and produce scatterplot visualizations.

Usage:
    python -m eflips.x.flows.benchmark_analysis benchmark_results.csv
    python -m eflips.x.flows.benchmark_analysis benchmark_results.csv --output-dir data/output/benchmark_plots
"""

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]
from matplotlib.patches import Patch
from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Styling (follows eflips/x/steps/analyzers/bvg_tools.py conventions)
# ---------------------------------------------------------------------------

PLOT_HEIGHT_PT = 490.0 / 3
PLOT_WIDTH_PT = 375.0
PLOT_WIDTH_INCH = PLOT_WIDTH_PT / 72.0
PLOT_HEIGHT_INCH = PLOT_HEIGHT_PT / 72.0

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


def configure_latex_plotting() -> None:
    matplotlib.rcParams.update(LATEX_RC_PARAMS)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "benchmark_plots"

# Columns that are metadata / features, not timing phases
META_COLS = {"flow_type", "agency", "repetition", "n_trips", "n_rotations", "n_vehicles"}


# ---------------------------------------------------------------------------
# Core reusable scatterplot
# ---------------------------------------------------------------------------


def scatterplot_runtime(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    xlabel: str | None = None,
    ylabel: str = r"Runtime [s]",
    title: str | None = None,
) -> plt.Figure:
    """Create a scatterplot of *value_col* against *feature_col*, colored by agency.

    This is the single plotting function used for every scatterplot in the
    analysis.  Modify the styling here and it applies everywhere.
    """
    configure_latex_plotting()

    fig, ax = plt.subplots(
        figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH * 1.5),
        layout="constrained",
    )

    sns.scatterplot(
        data=df,
        x=feature_col,
        y=value_col,
        hue="agency",
        ax=ax,
    )
    ax.legend().remove()

    ax.set_xlabel(xlabel or feature_col)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.xticks(rotation=45, ha="right")

    return fig


# ---------------------------------------------------------------------------
# Stacked-area phase share
# ---------------------------------------------------------------------------


_GROUP_COLORMAPS = {
    "common": "Blues",
    "depot": "Oranges",
    "opportunity": "Greens",
}

# Hatches cycle within each group so adjacent bands of the same color family
# remain visually distinct. The patterns are chosen to be roughly equal in
# visual weight and to differ structurally from each neighbour.
_HATCH_CYCLE = ["", "//", "..", "xx", "\\\\", "++", "oo", "--"]


def _humanize_phase(phase: str) -> str:
    """Turn ``"common/GTFSIngester"`` into ``"GTFS Ingester"`` (no group prefix).

    Labels with 4+ words and 25+ characters get a line break at the midpoint
    so they remain readable inside a narrow legend column.
    """
    name = phase.split("/", 1)[1] if "/" in phase else phase
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    words = spaced.split()
    if len(words) >= 4 and len(spaced) >= 25:
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
    return spaced


def _drop_crashed_runs(df: pd.DataFrame, phase_cols: list[str]) -> pd.DataFrame:
    """Drop repetitions where any phase column is NaN (i.e. the run crashed)."""
    incomplete = df[phase_cols].isna().any(axis=1)
    if incomplete.any():
        dropped = df.loc[incomplete, "agency"].unique().tolist()
        print(
            f"  Dropping {incomplete.sum()} crashed run(s) from stacked-area plot "
            f"(agencies: {', '.join(dropped)})"
        )
        df = df.loc[~incomplete]
    return df


def _phase_styling(
    phase_cols: list[str],
) -> tuple[list[str], list[tuple[float, float, float, float]], list[str]]:
    """Return ``(ordered_phases, colors, hatches)`` for stack-plotting."""
    ordered_phases: list[str] = []
    for group in _GROUP_COLORMAPS:
        ordered_phases.extend(c for c in phase_cols if c.split("/", 1)[0] == group)
    ordered_phases.extend(c for c in phase_cols if c not in ordered_phases)

    colors: list[tuple[float, float, float, float]] = []
    hatches: list[str] = []
    for group, cmap_name in _GROUP_COLORMAPS.items():
        group_phases = [c for c in ordered_phases if c.split("/", 1)[0] == group]
        if not group_phases:
            continue
        cmap = matplotlib.colormaps[cmap_name]
        for i, shade in enumerate(np.linspace(0.3, 0.95, len(group_phases))):
            colors.append(cmap(shade))
            hatches.append(_HATCH_CYCLE[i % len(_HATCH_CYCLE)])
    while len(hatches) < len(ordered_phases):
        hatches.append("")
        colors.append((0.5, 0.5, 0.5, 1.0))
    return ordered_phases, colors, hatches


def _smooth_phases_lowess(
    x: np.ndarray, y_per_phase: np.ndarray, x_eval: np.ndarray, frac: float = 0.5
) -> np.ndarray:
    """LOESS-smooth each phase's series independently and clip to non-negative.

    ``y_per_phase`` is shape ``(n_agencies, n_phases)``. Returns shape
    ``(len(x_eval), n_phases)``. Negative smoothed values (which LOESS can
    produce near zero-valued phases) are clipped to 0 so the stack stays
    physically meaningful.
    """
    out = np.empty((len(x_eval), y_per_phase.shape[1]))
    for p in range(y_per_phase.shape[1]):
        smooth = lowess(
            endog=y_per_phase[:, p],
            exog=x,
            frac=frac,
            xvals=x_eval,
            return_sorted=False,
        )
        out[:, p] = smooth
    return np.clip(out, 0.0, None)


def _stacked_area_phase_plot(
    df: pd.DataFrame,
    feature_col: str,
    xlabel: str | None,
    *,
    mode: str,
    smooth: bool,
) -> plt.Figure:
    """Shared scaffolding for the percentage and absolute-runtime stack plots."""
    configure_latex_plotting()

    phase_cols = [c for c in df.columns if c not in META_COLS and c != "total_runtime_s"]
    df = _drop_crashed_runs(df, phase_cols)
    ordered_phases, colors, hatches = _phase_styling(phase_cols)

    if smooth:
        # Skip the per-agency median: feed every (agency, repetition) row to
        # LOESS so the smoother sees the full sample, including within-agency
        # variability.
        sorted_df = df.sort_values(feature_col)
        x_data = sorted_df[feature_col].to_numpy(dtype=float)
        y_data = sorted_df[ordered_phases].to_numpy()
        x_agency_ticks = np.sort(df.groupby("agency")[feature_col].median().to_numpy())
        if len(x_data) >= 4:
            x_plot = np.linspace(x_data.min(), x_data.max(), 200)
            y_plot = _smooth_phases_lowess(x_data, y_data, x_plot)
        else:
            x_plot, y_plot = x_data, y_data
    else:
        # Raw view: collapse repetitions with the median, one vertex per agency.
        medians = (
            df.groupby("agency")[phase_cols + [feature_col]].median().sort_values(feature_col)
        )
        x_plot = medians[feature_col].to_numpy(dtype=float)
        y_plot = medians[ordered_phases].to_numpy()
        x_agency_ticks = x_plot

    if mode == "share":
        row_sums = y_plot.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        y_values = y_plot / row_sums * 100.0
        ylabel = r"Share of total runtime [\%]"
        ylim: tuple[float, float] | None = (0.0, 100.0)
    elif mode == "runtime":
        y_values = y_plot
        ylabel = r"Runtime [s]"
        ylim = None
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    labels = [_humanize_phase(p) for p in ordered_phases]

    fig, (ax, legend_ax) = plt.subplots(
        2,
        1,
        figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH * 2.8),
        gridspec_kw={"height_ratios": [3.0, 1.4]},
        layout="constrained",
    )
    legend_ax.axis("off")

    polys = ax.stackplot(x_plot, y_values.T, colors=colors)
    for poly, hatch in zip(polys, hatches):
        poly.set_hatch(hatch)
        poly.set_edgecolor("0.2")
        poly.set_linewidth(0.4)

    # Mark agency x-positions at the bottom so the smoothed view doesn't
    # hide where the data lives.
    if smooth and len(x_agency_ticks) > 0:
        ax.plot(
            x_agency_ticks,
            np.zeros_like(x_agency_ticks),
            marker="|",
            linestyle="none",
            color="0.15",
            markersize=5,
            markeredgewidth=0.6,
            clip_on=False,
        )

    ax.set_xlabel(xlabel or feature_col)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(x_plot.min(), x_plot.max())

    group_handles: dict[str, list[Patch]] = {g: [] for g in _GROUP_COLORMAPS}
    for color, hatch, label, phase in zip(colors, hatches, labels, ordered_phases):
        group = phase.split("/", 1)[0]
        if group in group_handles:
            group_handles[group].append(
                Patch(facecolor=color, hatch=hatch, edgecolor="0.2", linewidth=0.4, label=label)
            )

    legend_objs: list = []
    for group in _GROUP_COLORMAPS:
        handles = group_handles.get(group, [])
        if not handles:
            continue
        leg = legend_ax.legend(
            handles=handles,
            title=group.title(),
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            title_fontproperties={"weight": "bold", "size": 9},
            fontsize=8,
            frameon=False,
            handleheight=1.4,
            handlelength=2.0,
            handletextpad=0.5,
            alignment="left",
            borderaxespad=0.0,
        )
        legend_ax.add_artist(leg)
        legend_objs.append(leg)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    widths = [leg.get_tightbbox(renderer).width for leg in legend_objs]
    n = len(legend_objs)
    total = sum(widths)
    legend_ax_top_fig = legend_ax.get_position().y1
    pad_px = 0.01 * fig_w_px
    available_px = fig_w_px - 2 * pad_px
    gap = max(0.0, (available_px - total) / (n - 1)) if n > 1 else 0.0
    cursor_px = pad_px
    for leg, w in zip(legend_objs, widths):
        leg.set_bbox_to_anchor(
            (cursor_px / fig_w_px, legend_ax_top_fig), transform=fig.transFigure
        )
        cursor_px += w + gap

    return fig


def stacked_area_phase_share(
    df: pd.DataFrame,
    feature_col: str,
    xlabel: str | None = None,
    smooth: bool = True,
) -> plt.Figure:
    """Stacked area of each phase's share (%) of total runtime, agencies on x.

    LOESS-smoothed by default so trends across agency size are visible without
    the saw-tooth artefacts of straight-line interpolation between unevenly
    spaced agencies. After smoothing, shares are renormalised so each x still
    sums to exactly 100%. Tick marks at the bottom show the actual agency
    x-positions feeding the smoother.
    """
    return _stacked_area_phase_plot(df, feature_col, xlabel, mode="share", smooth=smooth)


def stacked_area_phase_runtime(
    df: pd.DataFrame,
    feature_col: str,
    xlabel: str | None = None,
    smooth: bool = True,
) -> plt.Figure:
    """Stacked area of each phase's median runtime (seconds), agencies on x.

    Unlike the share variant this preserves absolute scale: the stack height
    grows with agency size so you can read off how much wall-clock time each
    phase contributes. LOESS-smoothed by default.
    """
    return _stacked_area_phase_plot(df, feature_col, xlabel, mode="runtime", smooth=smooth)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    """Turn a phase column name into a filesystem-safe string."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _save(fig: plt.Figure, path: Path) -> None:
    """Save *fig* as both PDF and PNG, then close it."""
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse benchmark results")
    parser.add_argument("csv", type=Path, help="Path to benchmark_results.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output plots (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    # --- Summary plots: total runtime vs. feature --------------------------
    for feature, label in [
        ("n_trips", r"Number of Trips"),
        ("n_rotations", r"Number of Rotations"),
    ]:
        fig = scatterplot_runtime(df, feature, "total_runtime_s", xlabel=label)
        _save(fig, out / f"runtime_by_{feature}")
        print(f"Saved runtime_by_{feature}.{{pdf,png}}")

    # --- Detail plots: one per phase column --------------------------------
    phase_cols = [c for c in df.columns if c not in META_COLS and c != "total_runtime_s"]

    for feature, label in [
        ("n_trips", r"Number of Trips"),
        ("n_rotations", r"Number of Rotations"),
    ]:
        detail_dir = out / "detail" / f"by_{feature}"
        detail_dir.mkdir(parents=True, exist_ok=True)

        for phase in phase_cols:
            safe_name = _sanitize(phase)
            fig = scatterplot_runtime(
                df,
                feature,
                phase,
                xlabel=label,
                title=phase,
            )
            _save(fig, detail_dir / safe_name)
        print(f"Saved detail/by_{feature}/ ({len(phase_cols)} phases)")

    # --- Stacked-area variants: phase share + absolute runtime, raw + smooth
    for feature, label in [
        ("n_trips", r"Number of Trips"),
        ("n_rotations", r"Number of Rotations"),
    ]:
        for smooth_flag, suffix in [(False, ""), (True, "_smooth")]:
            fig = stacked_area_phase_share(df, feature, xlabel=label, smooth=smooth_flag)
            _save(fig, out / f"phase_share{suffix}_by_{feature}")
            print(f"Saved phase_share{suffix}_by_{feature}.{{pdf,png}}")

            fig = stacked_area_phase_runtime(df, feature, xlabel=label, smooth=smooth_flag)
            _save(fig, out / f"phase_runtime{suffix}_by_{feature}")
            print(f"Saved phase_runtime{suffix}_by_{feature}.{{pdf,png}}")


if __name__ == "__main__":
    main()
