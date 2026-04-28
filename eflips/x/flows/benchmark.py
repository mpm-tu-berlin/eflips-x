#!/usr/bin/env python3
"""
Benchmark eflips-x pipeline flows and collect per-step timing data.

Runs GTFS and/or BVG flows with a clean cache each time, extracts Prefect
task-level timing, and writes a CSV with problem-size features and per-step
runtimes.

Features:
- Incremental CSV writing (rows appended after each round)
- Resumable via a JSON state file: ``--resume`` skips already-completed
  rounds *before* running their flows.
- Logging redirected to file; tqdm progress bar on console.

Usage:
    python -m eflips.x.flows.benchmark --mode gtfs --repetitions 10 --agency "Havel"
    python -m eflips.x.flows.benchmark --mode bvg --repetitions 1
    python -m eflips.x.flows.benchmark --mode both --repetitions 5
    python -m eflips.x.flows.benchmark --mode gtfs --repetitions 10 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TYPE_CHECKING,
    Tuple,
    TypedDict,
    cast,
)
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

if TYPE_CHECKING:
    from eflips.x.flows.gtfs_flow import AgencyConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

FlowType = Literal["gtfs", "bvg"]
Row = Dict[str, Any]
CompletedKey = Tuple[str, str, int]


class CompletedEntry(TypedDict):
    flow_type: str
    agency: str
    repetition: int


class BenchmarkArgs(TypedDict, total=False):
    mode: str
    agency: Optional[str]
    repetitions: int


class BenchmarkState(TypedDict):
    completed: List[CompletedEntry]
    args: BenchmarkArgs


@dataclass
class WorkItem:
    """One unit of benchmark work: a single (flow, agency, repetition) round."""

    flow_type: FlowType
    agency: str
    repetition: int
    payload: Optional["AgencyConfig"] = None

    @property
    def key(self) -> CompletedKey:
        return (self.flow_type, self.agency, self.repetition)

    def to_completed_entry(self) -> CompletedEntry:
        return {
            "flow_type": self.flow_type,
            "agency": self.agency,
            "repetition": self.repetition,
        }


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


def collect_task_timings(root_flow_run_id: UUID) -> "OrderedDict[str, float]":
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
    timings: "OrderedDict[str, float]" = OrderedDict()

    for tr in task_runs:
        step_class = _extract_step_class(tr.name)
        if step_class is None or tr.flow_run_id is None:
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
# Feature extraction
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


def _assemble_row(item: WorkItem, flow_run_id: UUID, final_db: Path) -> Row:
    """Build a CSV row from a completed flow run."""
    total_s = get_flow_run_total_time(flow_run_id)
    timings = collect_task_timings(flow_run_id)
    features = extract_features(final_db)

    row: Row = {
        "flow_type": item.flow_type,
        "agency": item.agency,
        "repetition": item.repetition,
        **features,
        "total_runtime_s": total_s,
    }
    for label, secs in timings.items():
        row[label] = secs

    logger.info(
        f"  -> {total_s:.1f}s  trips={features['n_trips']}  "
        f"rotations={features['n_rotations']}  vehicles={features['n_vehicles']}"
    )
    return row


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

_STATE_FILENAME = "benchmark_state.json"


def _empty_state() -> BenchmarkState:
    return {"completed": [], "args": {}}


def _load_state(state_dir: Path) -> BenchmarkState:
    """Load the state file, returning an empty structure if it doesn't exist."""
    state_path = state_dir / _STATE_FILENAME
    if state_path.exists():
        return cast(BenchmarkState, json.loads(state_path.read_text()))
    return _empty_state()


def _save_state(state_dir: Path, state: BenchmarkState) -> None:
    """Write the state file atomically."""
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / _STATE_FILENAME
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    tmp_path.replace(state_path)


def _completed_keys(state: BenchmarkState) -> frozenset[CompletedKey]:
    """Build an O(1)-lookup set of completed (flow_type, agency, repetition) keys."""
    return frozenset((e["flow_type"], e["agency"], e["repetition"]) for e in state["completed"])


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _append_row_to_csv(csv_path: Path, row: Row) -> None:
    """Append a single row to the CSV, writing the header if the file doesn't exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _resolve_gtfs_agencies(
    agency_filter: Optional[str], gtfs_dir: Optional[Path]
) -> List["AgencyConfig"]:
    """Parse depot_locations.xlsx and apply the optional agency filter."""
    from eflips.x.flows.gtfs_flow import parse_depot_locations

    if gtfs_dir is None:
        gtfs_dir = PROJECT_ROOT / "data" / "input" / "GTFS"

    agencies = parse_depot_locations(gtfs_dir / "depot_locations.xlsx")

    if agency_filter:
        fl = agency_filter.lower()
        agencies = [
            a
            for a in agencies
            if fl in a.simulation_id.lower() or any(fl in n.lower() for n in a.agency_names)
        ]
        if not agencies:
            raise ValueError(f"No agencies matched filter '{agency_filter}'")
    return agencies


def _plan_gtfs(
    repetitions: int, agency_filter: Optional[str], gtfs_dir: Optional[Path]
) -> List[WorkItem]:
    agencies = _resolve_gtfs_agencies(agency_filter, gtfs_dir)
    return [
        WorkItem(flow_type="gtfs", agency=a.agency_name, repetition=rep + 1, payload=a)
        for rep in range(repetitions)
        for a in agencies
    ]


def _plan_bvg(repetitions: int) -> List[WorkItem]:
    return [
        WorkItem(flow_type="bvg", agency="BVG", repetition=rep + 1) for rep in range(repetitions)
    ]


def _plan_all(args: argparse.Namespace) -> List[WorkItem]:
    plan: List[WorkItem] = []
    if args.mode in ("gtfs", "both"):
        plan.extend(_plan_gtfs(args.repetitions, args.agency, args.gtfs_dir))
    if args.mode in ("bvg", "both"):
        plan.extend(_plan_bvg(args.repetitions))
    return plan


# ---------------------------------------------------------------------------
# Per-item runners
# ---------------------------------------------------------------------------


def _run_gtfs(item: WorkItem) -> Row:
    from eflips.x.flows.gtfs_flow import run_agency_flow

    assert item.payload is not None, "GTFS work item missing AgencyConfig payload"
    agency = item.payload

    with tempfile.TemporaryDirectory(prefix="eflips_bench_gtfs_") as tmpdir:
        cache_root = Path(tmpdir) / "cache"
        output_root = Path(tmpdir) / "output"

        flow_state = run_agency_flow(
            agency=agency,
            cache_base_root=cache_root,
            output_base_root=output_root,
            agency_name=agency.agency_name,
            enable_plots=False,
            tolerate_failures=False,
            return_state=True,
        )

        flow_run_id = flow_state.state_details.flow_run_id
        assert flow_run_id is not None, "Prefect flow state has no flow_run_id"
        depot_dir = cache_root / agency.gtfs_file.stem / agency.slug / "depot"
        return _assemble_row(item, flow_run_id, _find_final_db(depot_dir))


def _run_bvg(item: WorkItem) -> Row:
    import eflips.x.flows.bvg as bvg_module

    with tempfile.TemporaryDirectory(prefix="eflips_bench_bvg_") as tmpdir:
        original_work_dir = bvg_module.WORK_DIR_BASE
        bvg_module.WORK_DIR_BASE = Path(tmpdir) / "cache"
        try:
            flow_state = bvg_module.bvg_three_scenario_flow(return_state=True)
            flow_run_id = flow_state.state_details.flow_run_id
            assert flow_run_id is not None, "Prefect flow state has no flow_run_id"
            common_dir = Path(tmpdir) / "cache" / "common"
            return _assemble_row(item, flow_run_id, _find_final_db(common_dir))
        finally:
            bvg_module.WORK_DIR_BASE = original_work_dir


def _run(item: WorkItem) -> Row:
    """Run one work item. On failure, return a row dict with an ``error`` key."""
    logger.info(f"[{item.flow_type.upper()}] rep {item.repetition}  agency='{item.agency}'")
    try:
        if item.flow_type == "gtfs":
            return _run_gtfs(item)
        return _run_bvg(item)
    except Exception:
        logger.exception(
            f"[{item.flow_type.upper()}] FAILED rep {item.repetition}  " f"agency='{item.agency}'"
        )
        return {
            "flow_type": item.flow_type,
            "agency": item.agency,
            "repetition": item.repetition,
            "error": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def _setup_logging(log_dir: Path) -> Path:
    """Redirect root logging to a file under *log_dir*; keep tqdm on console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "benchmark.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root_logger.addHandler(file_handler)
    return log_path


def _warn_args_mismatch(saved: BenchmarkArgs, current: argparse.Namespace) -> None:
    if saved.get("mode") and saved["mode"] != current.mode:
        logger.warning(
            f"Resuming with mode={current.mode} but state was saved with " f"mode={saved['mode']}"
        )
    if saved.get("agency") and saved["agency"] != current.agency:
        logger.warning(
            f"Resuming with agency={current.agency} but state was saved with "
            f"agency={saved['agency']}"
        )


def main() -> None:
    args = _parse_args()
    log_path = _setup_logging(args.log_dir)
    print(f"Logging to {log_path.resolve()}", file=sys.stderr)

    # --- Load / initialize state ---
    state: BenchmarkState = _empty_state()
    if args.resume:
        state = _load_state(args.state_dir)
        _warn_args_mismatch(state.get("args", {}), args)
        print(
            f"Resuming: {len(state['completed'])} rounds already completed",
            file=sys.stderr,
        )

    state["args"] = {
        "mode": args.mode,
        "agency": args.agency,
        "repetitions": args.repetitions,
    }

    # --- Plan and filter ---
    plan = _plan_all(args)
    if not plan:
        print("No work items to run.", file=sys.stderr)
        return

    done = _completed_keys(state)
    remaining = [item for item in plan if item.key not in done]
    already_done = len(plan) - len(remaining)

    # --- Execute remaining items ---
    pbar = tqdm(total=len(plan), initial=already_done, desc="Benchmark", unit="run")
    rows_written = 0
    failures = 0
    try:
        for item in remaining:
            row = _run(item)
            _append_row_to_csv(args.output, row)
            state["completed"].append(item.to_completed_entry())
            _save_state(args.state_dir, state)

            if "error" in row:
                failures += 1
                pbar.set_postfix_str(f"FAILED {item.agency} rep {item.repetition}")
            else:
                pbar.set_postfix_str(f"{item.agency} rep {item.repetition}")
            rows_written += 1
            pbar.update(1)
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

    if args.output.exists():
        df = pd.read_csv(args.output)
        print(f"Total rows in CSV: {len(df)}", file=sys.stderr)


if __name__ == "__main__":
    main()
