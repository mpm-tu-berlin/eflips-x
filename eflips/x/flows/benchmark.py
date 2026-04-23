#!/usr/bin/env python3
"""
Benchmark eflips-x pipeline flows and collect per-step timing data.

Runs GTFS and/or BVG flows with a clean cache each time, extracts Prefect task-level
timing, and writes a CSV with problem-size features and per-step runtimes.

Features:
- Incremental CSV writing (rows appended after each round)
- Resumable via a JSON state file
- Logging redirected to file; tqdm progress bar on console

Usage:
    python -m eflips.x.flows.benchmark --mode gtfs --repetitions 10 --agency "Havel"
    python -m eflips.x.flows.benchmark --mode bvg --repetitions 1
    python -m eflips.x.flows.benchmark --mode both --repetitions 5
    python -m eflips.x.flows.benchmark --mode gtfs --repetitions 10 --resume
"""

import argparse
import json
import logging
import sys
import tempfile
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from uuid import UUID

import pandas as pd
from prefect import get_client
from prefect.client.schemas.filters import (
    FlowRunFilter,
    FlowRunFilterId,
    FlowRunFilterParentFlowRunId,
)
from prefect.client.schemas.sorting import TaskRunSort
from tqdm import tqdm

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
    base = task_run_name.rsplit("-", 1)[0]
    if " " in base:
        return None
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
            continue

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
# State management
# ---------------------------------------------------------------------------

_STATE_FILENAME = "benchmark_state.json"


def _load_state(state_dir: Path) -> Dict[str, Any]:
    """Load the state file, returning an empty structure if it doesn't exist."""
    state_path = state_dir / _STATE_FILENAME
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"completed": [], "args": {}}


def _save_state(state_dir: Path, state: Dict[str, Any]) -> None:
    """Write the state file atomically."""
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / _STATE_FILENAME
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    tmp_path.replace(state_path)


def _is_completed(state: Dict[str, Any], flow_type: str, agency: str, rep: int) -> bool:
    """Check whether a specific round is already recorded as completed."""
    for entry in state["completed"]:
        if (
            entry["flow_type"] == flow_type
            and entry["agency"] == agency
            and entry["repetition"] == rep
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _append_row_to_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    """Append a single row to the CSV, writing the header if the file doesn't exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# GTFS benchmarking (generator)
# ---------------------------------------------------------------------------


def _iter_gtfs_work_items(
    agency_filter: Optional[str],
    gtfs_dir: Optional[Path],
    repetitions: int,
) -> List[Tuple[str, int]]:
    """Return list of (agency_name, repetition) pairs for GTFS benchmarking."""
    from eflips.x.flows.gtfs_flow import parse_depot_locations

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

    items = []
    for rep in range(repetitions):
        for agency in agencies:
            items.append((agency.agency_name, rep + 1))
    return items


def benchmark_gtfs_iter(
    repetitions: int,
    agency_filter: Optional[str],
    gtfs_dir: Optional[Path],
) -> Iterator[Dict[str, Any]]:
    """Yield one result dict per agency/repetition."""
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

    for rep in range(repetitions):
        for agency in agencies:
            logger.info(f"[GTFS] rep {rep + 1}/{repetitions}  agency='{agency.agency_name}'")

            try:
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

                    logger.info(
                        f"  -> {total_s:.1f}s  trips={features['n_trips']}  "
                        f"rotations={features['n_rotations']}  vehicles={features['n_vehicles']}"
                    )
                    yield row
            except Exception:
                logger.exception(
                    f"[GTFS] FAILED rep {rep + 1}/{repetitions}  " f"agency='{agency.agency_name}'"
                )
                yield {
                    "flow_type": "gtfs",
                    "agency": agency.agency_name,
                    "repetition": rep + 1,
                    "error": traceback.format_exc(),
                }


# ---------------------------------------------------------------------------
# BVG benchmarking (generator)
# ---------------------------------------------------------------------------


def benchmark_bvg_iter(repetitions: int) -> Iterator[Dict[str, Any]]:
    """Yield one result dict per repetition."""
    import eflips.x.flows.bvg as bvg_module

    for rep in range(repetitions):
        logger.info(f"[BVG] rep {rep + 1}/{repetitions}")

        with tempfile.TemporaryDirectory(prefix="eflips_bench_bvg_") as tmpdir:
            original_work_dir = bvg_module.WORK_DIR_BASE
            bvg_module.WORK_DIR_BASE = Path(tmpdir) / "cache"

            try:
                flow_state = bvg_module.bvg_three_scenario_flow(return_state=True)
                flow_run_id = flow_state.state_details.flow_run_id
                total_s = get_flow_run_total_time(flow_run_id)
                timings = collect_task_timings(flow_run_id)

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

                logger.info(
                    f"  -> {total_s:.1f}s  trips={features['n_trips']}  "
                    f"rotations={features['n_rotations']}  vehicles={features['n_vehicles']}"
                )
                yield row
            except Exception:
                logger.exception(f"[BVG] FAILED rep {rep + 1}/{repetitions}")
                yield {
                    "flow_type": "bvg",
                    "agency": "BVG",
                    "repetition": rep + 1,
                    "error": traceback.format_exc(),
                }
            finally:
                bvg_module.WORK_DIR_BASE = original_work_dir


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
        default=Path("data/output/benchmark_results.csv"),
        help="Output CSV path (default: data/output/benchmark_results.csv)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("data/cache"),
        help="Directory for the benchmark state file (default: data/cache)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous run using the state file",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("data/cache"),
        help="Directory for the benchmark log file (default: data/cache)",
    )
    args = parser.parse_args()

    # --- Set up logging to file instead of console ---
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "benchmark.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Remove any default handlers (e.g. StreamHandler to stderr)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)

    # Also redirect stdout/stderr from subprocesses and Prefect to the log
    # (tqdm writes to its own fd so it stays visible)
    print(f"Logging to {log_path.resolve()}", file=sys.stderr)

    # --- Load / initialize state ---
    state = {"completed": [], "args": {}}
    if args.resume:
        state = _load_state(args.state_dir)
        saved_args = state.get("args", {})
        if saved_args.get("mode") and saved_args["mode"] != args.mode:
            logger.warning(
                f"Resuming with mode={args.mode} but state was saved with "
                f"mode={saved_args['mode']}"
            )
        if saved_args.get("agency") and saved_args["agency"] != args.agency:
            logger.warning(
                f"Resuming with agency={args.agency} but state was saved with "
                f"agency={saved_args['agency']}"
            )
        n_completed = len(state["completed"])
        print(f"Resuming: {n_completed} rounds already completed", file=sys.stderr)

    # Save args into state for future resume validation
    state["args"] = {
        "mode": args.mode,
        "agency": args.agency,
        "repetitions": args.repetitions,
    }

    # --- Compute total work items for the progress bar ---
    total_items = 0
    already_done = 0

    if args.mode in ("gtfs", "both"):
        gtfs_items = _iter_gtfs_work_items(args.agency, args.gtfs_dir, args.repetitions)
        total_items += len(gtfs_items)
        for agency_name, rep in gtfs_items:
            if _is_completed(state, "gtfs", agency_name, rep):
                already_done += 1

    if args.mode in ("bvg", "both"):
        bvg_count = args.repetitions
        total_items += bvg_count
        for rep in range(1, bvg_count + 1):
            if _is_completed(state, "bvg", "BVG", rep):
                already_done += 1

    if total_items == 0:
        print("No work items to run.", file=sys.stderr)
        return

    # --- Main loop with tqdm ---
    pbar = tqdm(total=total_items, initial=already_done, desc="Benchmark", unit="run")
    rows_written = 0

    failures = 0

    def _process_row(row: Dict[str, Any]) -> None:
        nonlocal rows_written, failures
        flow_type = row["flow_type"]
        agency = row["agency"]
        rep = row["repetition"]

        if _is_completed(state, flow_type, agency, rep):
            pbar.update(1)
            return

        _append_row_to_csv(args.output, row)
        rows_written += 1

        if "error" in row:
            failures += 1
            pbar.set_postfix_str(f"FAILED {agency} rep {rep}")
        else:
            pbar.set_postfix_str(f"{agency} rep {rep}")

        state["completed"].append({"flow_type": flow_type, "agency": agency, "repetition": rep})
        _save_state(args.state_dir, state)

        pbar.update(1)

    try:
        if args.mode in ("gtfs", "both"):
            for row in benchmark_gtfs_iter(
                repetitions=args.repetitions,
                agency_filter=args.agency,
                gtfs_dir=args.gtfs_dir,
            ):
                _process_row(row)

        if args.mode in ("bvg", "both"):
            for row in benchmark_bvg_iter(repetitions=args.repetitions):
                _process_row(row)
    finally:
        pbar.close()

    if rows_written == 0 and already_done > 0:
        print("All rounds already completed (nothing new to run).", file=sys.stderr)
    elif rows_written == 0:
        logger.warning("No benchmark data collected.")
        print("No benchmark data collected.", file=sys.stderr)
    else:
        fail_msg = f", {failures} failed" if failures else ""
        print(
            f"\n{rows_written} new rows written to {args.output}  "
            f"({already_done} previously completed{fail_msg})",
            file=sys.stderr,
        )

    # Print summary if CSV exists
    if args.output.exists():
        df = pd.read_csv(args.output)
        print(f"Total rows in CSV: {len(df)}", file=sys.stderr)


if __name__ == "__main__":
    main()
