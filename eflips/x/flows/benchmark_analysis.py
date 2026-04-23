#!/usr/bin/env python3
"""
Analyse benchmark results and produce boxplot visualizations.

Usage:
    python -m eflips.x.flows.benchmark_analysis benchmark_results.csv
    python -m eflips.x.flows.benchmark_analysis benchmark_results.csv --output-dir data/output/benchmark_plots
"""

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore[import-untyped]

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
# Core reusable boxplot
# ---------------------------------------------------------------------------


def boxplot_runtime(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    xlabel: str | None = None,
    ylabel: str = r"Runtime [s]",
    title: str | None = None,
) -> plt.Figure:
    """Create a vertical boxplot of *value_col* grouped by *feature_col*.

    This is the single plotting function used for every boxplot in the
    analysis.  Modify the styling here and it applies everywhere.
    """
    configure_latex_plotting()
    palette = sns.color_palette("Set2")

    fig, ax = plt.subplots(
        figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH * 1.5),
        layout="constrained",
    )

    order = sorted(df[feature_col].dropna().unique())

    sns.boxplot(
        data=df,
        x=feature_col,
        y=value_col,
        order=order,
        color=palette[0],
        ax=ax,
    )

    ax.set_xlabel(xlabel or feature_col)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.xticks(rotation=45, ha="right")

    return fig


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
        fig = boxplot_runtime(df, feature, "total_runtime_s", xlabel=label)
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
            fig = boxplot_runtime(
                df,
                feature,
                phase,
                xlabel=label,
                title=phase,
            )
            _save(fig, detail_dir / safe_name)
        print(f"Saved detail/by_{feature}/ ({len(phase_cols)} phases)")


if __name__ == "__main__":
    main()
