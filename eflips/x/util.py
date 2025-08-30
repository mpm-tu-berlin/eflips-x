import warnings

from prefect import task, flow
from prefect.artifacts import create_markdown_artifact
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import sqlite3
import os


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file for cache invalidation."""
    if not os.path.exists(filepath):
        return "missing"

    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def create_cache_key(
    input_db: Optional[str] = None,
    input_files: List[str] = None,
    params: Dict[str, Any] = None,
    code_version: str = "v1",
) -> str:
    """
    Create cache key based on inputs, files, and parameters. The cache key is used by Prefect to determine if a step
    can be skipped based on previous results. So it needs to stay the same if the result of the step is expected to be
    the same.

    Therefore, it includes:
       - Code version (to handle changes in the code) **This needs to be updated manually**.
       - the `poetry.lock` hash (to handle changes in dependencies)
       - Input database (if provided)
       - Input files (if provided)
       - Parameters (if provided, sorted to ensure consistent hashing)
    """
    key_parts = [f"code:{code_version}"]

    # Hash poetry.lock to capture dependency changes
    path_to_this_file = Path(__file__).resolve()
    project_root = path_to_this_file.parent.parent.parent  # Assuming this file is in eflips/x/
    poetry_lock_path = project_root / "poetry.lock"
    if poetry_lock_path.exists():
        key_parts.append(f"deps:{compute_file_hash(poetry_lock_path)}")
    else:
        warnings.warn("poetry.lock not found, dependency changes won't be tracked in cache key.")
        key_parts.append("deps:missing")

    # Hash input database
    if input_db:
        key_parts.append(f"db:{compute_file_hash(input_db)}")

    # Hash additional input files
    if input_files:
        for filepath in input_files:
            key_parts.append(f"file:{Path(filepath).name}:{compute_file_hash(filepath)}")

    # Hash parameters
    if params:
        params_str = str(sorted(params.items()))
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
        key_parts.append(f"params:{params_hash}")

    return ":".join(key_parts)


def pipeline_step(
    step_name: str,
    input_files: List[str] = None,
    params: Dict[str, Any] = None,
    code_version: str = "v1",
):
    """
    Decorator to create a pipeline step with proper caching and dependencies. This extends Prefect's task functionality
    to include input files, parameters, and a code version for cache invalidation.

    """

    def decorator(func):
        @task(
            name=f"{step_name}",
            cache_key_fn=lambda ctx, parameters: create_cache_key(
                input_db=parameters.get("input_db"),
                input_files=input_files or [],
                params=params or {},
                code_version=code_version,
            ),
        )
        def wrapper(input_db: Optional[str], output_db: str, **kwargs) -> str:
            # Copy input database to output path (preserves intermediate states)
            if input_db:
                shutil.copy2(input_db, output_db)

            # Execute the actual step function
            result = func(output_db, **kwargs)

            # Log step completion
            create_markdown_artifact(
                key=f"{step_name}-completion",
                markdown=f"## {step_name} Completed\n\nOutput: `{output_db}`",
                description=f"Step {step_name} completed successfully",
            )

            return output_db

        return wrapper

    return decorator
