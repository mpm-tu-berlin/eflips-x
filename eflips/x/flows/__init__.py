"""eflips-x reusable Prefect flows."""

import logging
from pathlib import Path
from typing import Sequence

from matplotlib.figure import Figure

from eflips.x.flows.analysis_flow import generate_all_plots

__all__ = ["generate_all_plots", "run_steps", "save_plot_to_files_in_output_dir"]

from eflips.x.framework import PipelineContext, PipelineStep

logger = logging.getLogger(__name__)


def run_steps(context: PipelineContext, steps: Sequence[PipelineStep]) -> None:
    """Run a sequence of pipeline steps."""
    for step in steps:
        step.execute(context=context)


def save_plot_to_files_in_output_dir(
    fig: Figure, output_dir: Path, basename: str, dpi: int = 300
) -> None:
    """Save a Matplotlib figure to ``{output_dir}/{basename}.{pdf,png}``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = output_dir / f"{basename}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        logger.info(f"Saved plot to: {path}")
