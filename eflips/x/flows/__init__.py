"""eflips-x reusable Prefect flows."""

from typing import List

from eflips.x.flows.analysis_flow import generate_all_plots

__all__ = ["generate_all_plots"]

from eflips.x.framework import PipelineContext, PipelineStep


def run_steps(context: PipelineContext, steps: List[PipelineStep]) -> None:
    """Run a sequence of pipeline steps."""
    for step in steps:
        step.execute(context=context)
