#!/usr/bin/env python3
"""
Benchmark eflips-x pipeline flows and collect per-step timing data.

Runs GTFS and/or BVG flows with a clean cache each time, extracts Prefect task-level
timing, and writes a CSV with problem-size features and per-step runtimes.

Usage:
    python -m eflips.x.flows.benchmark --mode gtfs --repetitions 10 --agency "Havel"
    python -m eflips.x.flows.benchmark --mode bvg --repetitions 1
    python -m eflips.x.flows.benchmark --mode both --repetitions 5
"""

import argparse
import logging
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd
from prefect import get_client
from prefect.client.schemas.filters import (
    FlowRunFilter,
    FlowRunFilterId,
    FlowRunFilterParentFlowRunId,
)
from prefect.client.schemas.sorting import TaskRunSort

from eflips.model import Rotation, Trip, Vehicle

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Prefect timing extraction
# ---------------------------------------------------------------------------


def _collect_descendant_flow_run_ids(root_id: UUID) -> Dict[UUID, str]:
    """Return {flow_run_id: flow_run_name} for *root_id* and all nested subflows."""
    result: Dict[UUID, str] = {}
    with get_client(sync_client=True) as client:
        root = client.read_flow_run(root_id)
        result[root.id] = root.name

        queue = [root_id]
        while queue:
            parent_id = queue.pop(0)
            children = client.read_flow_runs(
                flow_run_filter=FlowRunFilter(
                    parent_flow_run_id=FlowRunFilterParentFlowRunId(any_=[parent_id])
                )
            )
            for child in children:
                result[child.id] = child.name
                queue.append(child.id)

    return result


def _extract_variant(flow_run_name: str) -> str:
    """Extract the variant portion from a flow run name.

    ``"AmperBus GmbH - common"`` → ``"common"``
    ``"AmperBus GmbH"``          → ``"top"``
    ``"Common Pipeline"``        → ``"Common Pipeline"``  (BVG)
    """
    if " - " in flow_run_name:
        return flow_run_name.rsplit(" - ", 1)[-1]
    return flow_run_name


def _extract_step_class(task_run_name: str) -> Optional[str]:
    """Extract the PipelineStep class name from a task run name.

    ``"GTFSIngester-deb"`` → ``"GTFSIngester"``
    ``"GTFS Common Phase-0"`` → ``None``  (subflow call, not a step)

    Pipeline step tasks are single PascalCase words (no spaces) set via
    ``@task(name=self.__class__.__name__)``.  Subflow task runs contain spaces.
    """
    # Strip the trailing Prefect hash/counter: everything after the last '-'
    base = task_run_name.rsplit("-", 1)[0]
    if " " in base:
        return None  # subflow invocation, not a pipeline step
    return base


def collect_task_timings(root_flow_run_id: UUID) -> OrderedDict[str, float]:
    """Return an ordered dict of ``{variant/ClassName: seconds}`` for every
    pipeline-step task in the flow tree.

    Subflow-invocation tasks (names with spaces) are skipped.
    Duplicate labels (same step in the same variant) get a ``_N`` suffix.
    """
    flow_runs = _collect_descendant_flow_run_ids(root_flow_run_id)

    with get_client(sync_client=True) as client:
        task_runs = client.read_task_runs(
            flow_run_filter=FlowRunFilter(id=FlowRunFilterId(any_=list(flow_runs.keys()))),
            sort=TaskRunSort.EXPECTED_START_TIME_ASC,
            limit=200,
        )

    seen: Dict[str, int] = {}
    timings: OrderedDict[str, float] = OrderedDict()

    for tr in task_runs:
        step_class = _extract_step_class(tr.name)
        if step_class is None:
            continue  # skip subflow invocation tasks

        variant = _extract_variant(flow_runs.get(tr.flow_run_id, "unknown"))
        base = f"{variant}/{step_class}"

        if base in seen:
            seen[base] += 1
            label = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
            label = base

        timings[label] = tr.total_run_time.total_seconds()

    return timings


def get_flow_run_total_time(flow_run_id: UUID) -> float:
    with get_client(sync_client=True) as client:
        fr = client.read_flow_run(flow_run_id)
        return fr.total_run_time.total_seconds()


# ---------------------------------------------------------------------------
# Feature extraction from final database
# ---------------------------------------------------------------------------


def extract_features(db_path: Path) -> Dict[str, int]:
    """Count trips, rotations, and vehicles in a simulation database."""
    from eflips.x.framework import PipelineContext

    ctx = PipelineContext(work_dir=db_path.parent, current_db=db_path)
    with ctx.get_session() as session:
        return {
            "n_trips": session.query(Trip).count(),
            "n_rotations": session.query(Rotation).count(),
            "n_vehicles": session.query(Vehicle).count(),
        }


def _find_final_db(search_dir: Path) -> Path:
    """Return the highest-numbered ``step_*.db`` file under *search_dir*."""
    dbs = sorted(search_dir.rglob("step_*.db"))
    if not dbs:
        raise FileNotFoundError(f"No step databases found under {search_dir}")
    return dbs[-1]


# ---------------------------------------------------------------------------
# GTFS benchmarking
# ---------------------------------------------------------------------------


def benchmark_gtfs(
    repetitions: int,
    agency_filter: Optional[str],
    gtfs_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    from eflips.x.flows.gtfs_flow import parse_depot_locations, run_agency_flow

    if gtfs_dir is None:
        gtfs_dir = PROJECT_ROOT / "data" / "input" / "GTFS"

    excel_path = gtfs_dir / "depot_locations.xlsx"
    agencies = parse_depot_locations(excel_path)

    if agency_filter:
        fl = agency_filter.lower()
        agencies = [
            a
            for a in agencies
            if fl in a.simulation_id.lower() or any(fl in n.lower() for n in a.agency_names)
        ]
        if not agencies:
            raise ValueError(f"No agencies matched filter '{agency_filter}'")

    rows: List[Dict[str, Any]] = []

    for rep in range(repetitions):
        for agency in agencies:
            logger.info(f"[GTFS] rep {rep + 1}/{repetitions}  agency='{agency.agency_name}'")

            with tempfile.TemporaryDirectory(prefix="eflips_bench_gtfs_") as tmpdir:
                cache_root = Path(tmpdir) / "cache"
                output_root = Path(tmpdir) / "output"

                state = run_agency_flow(
                    agency=agency,
                    cache_base_root=cache_root,
                    output_base_root=output_root,
                    agency_name=agency.agency_name,
                    enable_plots=False,
                    tolerate_failures=False,
                    return_state=True,
                )

                flow_run_id = state.state_details.flow_run_id
                total_s = get_flow_run_total_time(flow_run_id)
                timings = collect_task_timings(flow_run_id)

                # Features from the depot variant's final DB
                gtfs_stem = agency.gtfs_file.stem
                slug = agency.slug
                depot_dir = cache_root / gtfs_stem / slug / "depot"
                features = extract_features(_find_final_db(depot_dir))

                row: Dict[str, Any] = {
                    "flow_type": "gtfs",
                    "agency": agency.agency_name,
                    "repetition": rep + 1,
                    **features,
                    "total_runtime_s": total_s,
                }
                for label, secs in timings.items():
                    row[label] = secs

                rows.append(row)
                logger.info(
                    f"  -> {total_s:.1f}s  trips={features['n_trips']}  "
                    f"rotations={features['n_rotations']}  vehicles={features['n_vehicles']}"
                )

    return rows


# ---------------------------------------------------------------------------
# BVG benchmarking
# ---------------------------------------------------------------------------


def benchmark_bvg(repetitions: int) -> List[Dict[str, Any]]:
    import eflips.x.flows.bvg as bvg_module

    rows: List[Dict[str, Any]] = []

    for rep in range(repetitions):
        logger.info(f"[BVG] rep {rep + 1}/{repetitions}")

        with tempfile.TemporaryDirectory(prefix="eflips_bench_bvg_") as tmpdir:
            original_work_dir = bvg_module.WORK_DIR_BASE
            bvg_module.WORK_DIR_BASE = Path(tmpdir) / "cache"

            try:
                state = bvg_module.bvg_three_scenario_flow(return_state=True)
                flow_run_id = state.state_details.flow_run_id
                total_s = get_flow_run_total_time(flow_run_id)
                timings = collect_task_timings(flow_run_id)

                # Features from the common pipeline's final DB
                common_dir = Path(tmpdir) / "cache" / "common"
                features = extract_features(_find_final_db(common_dir))

                row: Dict[str, Any] = {
                    "flow_type": "bvg",
                    "agency": "BVG",
                    "repetition": rep + 1,
                    **features,
                    "total_runtime_s": total_s,
                }
                for label, secs in timings.items():
                    row[label] = secs

                rows.append(row)
                logger.info(
                    f"  -> {total_s:.1f}s  trips={features['n_trips']}  "
                    f"rotations={features['n_rotations']}  vehicles={features['n_vehicles']}"
                )
            finally:
                bvg_module.WORK_DIR_BASE = original_work_dir

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark eflips-x flows and collect per-step timing data"
    )
    parser.add_argument(
        "--mode",
        choices=["gtfs", "bvg", "both"],
        default="both",
        help="Which flow(s) to benchmark (default: both)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Number of repetitions per agency/flow (default: 10)",
    )
    parser.add_argument(
        "--agency",
        type=str,
        default=None,
        help="Case-insensitive substring filter on GTFS agency name",
    )
    parser.add_argument(
        "--gtfs-dir",
        type=Path,
        default=None,
        help="Override GTFS input directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="Output CSV path (default: benchmark_results.csv)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    all_rows: List[Dict[str, Any]] = []

    if args.mode in ("gtfs", "both"):
        all_rows.extend(
            benchmark_gtfs(
                repetitions=args.repetitions,
                agency_filter=args.agency,
                gtfs_dir=args.gtfs_dir,
            )
        )

    if args.mode in ("bvg", "both"):
        all_rows.extend(benchmark_bvg(repetitions=args.repetitions))

    if not all_rows:
        logger.warning("No benchmark data collected.")
        return

    df = pd.DataFrame(all_rows)

    # Reorder: metadata & features first, then timing columns
    meta_cols = ["flow_type", "agency", "repetition", "n_trips", "n_rotations", "n_vehicles"]
    timing_cols = [c for c in df.columns if c not in meta_cols]
    # Ensure total_runtime_s comes right after features
    if "total_runtime_s" in timing_cols:
        timing_cols.remove("total_runtime_s")
        timing_cols = ["total_runtime_s"] + timing_cols
    df = df[meta_cols + timing_cols]

    df.to_csv(args.output, index=False)
    logger.info(f"Results written to {args.output}  ({len(df)} rows, {len(df.columns)} columns)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
