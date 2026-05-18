# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

eflips-x is a Prefect-based workflow orchestrator for running complex electric bus network simulations. It integrates
multiple eFLIPS components (model, depot, ingest, eval, opt, tco) into configurable, cacheable pipeline workflows.

The framework is built around three core pipeline step types, each with automatic caching, Prefect integration, and
database state chaining.

## Core Architecture

### Pipeline Framework (`eflips/x/framework/__init__.py`)

The framework defines three abstract base classes that all pipeline steps inherit from:

**Generators** (`Generator`): Create new databases from input files. Do NOT depend on previous database state for cache
invalidation. Use cases: ingesting BVG XML schedules, GTFS data, copying an existing database (`CopyCreator`).

**Modifiers** (`Modifier`): Transform existing databases by copying and modifying them. Depend on previous database
state for cache invalidation. Use cases: vehicle scheduling, depot assignment, simulation.

**Analyzers** (`Analyzer`): Read databases and produce reports/visualizations. Do NOT modify the database. Return
results from `execute()` and provide a `visualize()` method.

### Pipeline Context (`PipelineContext`)

The `PipelineContext` manages pipeline execution state:

- `work_dir`: Working directory for database files
- `params`: Dictionary of parameters (global and step-specific)
- `current_db`: Current database path
- `artifacts`: Dictionary for storing artifacts
- `get_session()`: Context manager for database sessions

### Cache Key Computation

All pipeline steps implement `compute_cache_key()` which hashes:

- Code version (required parameter in `__init__`)
- `poetry.lock` file hash (for dependency tracking)
- Input file hashes (Generators) or current database hash (Modifiers/Analyzers)
- Additional file hashes if specified
- Relevant parameters from `PipelineContext.params`

### Parameter System

Parameters are passed through `PipelineContext.params` dictionary:

- Global parameters: no prefix (e.g., `"log_level": "INFO"`)
- Step-specific parameters: prefixed with class name (e.g., `"VehicleScheduling.charge_type": ChargeType.DEPOT`)

All steps must implement `document_params()` class method that returns a dictionary documenting their parameters.

The global parameter `"scenario_display_config"` accepts a `ScenarioDisplayConfig` instance (from `eflips.x.framework`) to control scenario ordering, display names, and baseline selection in multi-scenario visualizations. Analyzers and merge functions that accept a `config` parameter will fall back to hardcoded defaults when this is not set.

### Database Chaining

Steps execute sequentially, each producing a new database file:

- Generators create `step_001_ClassName.db`
- Modifiers copy the previous database to `step_NNN_ClassName.db` and modify it
- Analyzers create a temporary copy for read-only analysis
- Failed steps move database to `.{timestamp}.failed` extension

## Directory Structure

```
eflips/x/
├── framework/          # Core framework (PipelineStep, Generator, Modifier, Analyzer)
├── steps/
│   ├── generators/    # Create databases from input files
│   │   └── __init__.py            # BVGXMLIngester, GTFSIngester, CopyCreator
│   ├── modifiers/     # Transform databases
│   │   ├── bvg_tools.py           # BVG-specific utilities
│   │   ├── general_utilities.py   # RemoveUnusedData, AddTemperatures
│   │   ├── gtfs_utilities.py      # ConfigureVehicleTypes (parameterized)
│   │   ├── scheduling.py          # VehicleScheduling, DepotAssignment, StationElectrification
│   │   └── simulation.py          # DepotGenerator, Simulation
│   └── analyzers/     # Analyze and visualize results
│       ├── bvg_tools.py           # BVG-specific plots and TCO comparison
│       ├── input_analyzers.py     # Pre-simulation analysis
│       ├── json_export.py         # ScenarioJsonExporter
│       └── output_analyzers.py    # Post-simulation analysis
├── flows/             # Example pipeline flows
│   ├── analysis_flow.py           # Parallel visualization generation
│   ├── bvg.py                     # BVG XML-driven flow
│   ├── example.py                 # Minimal BVG example
│   └── gtfs_flow.py               # Multi-agency GTFS flow driven by depot_locations.xlsx
└── util/              # Utility functions
```

## Development Commands

### Environment Setup

Required environment variables:

```bash
export SPATIALITE_LIBRARY_PATH=/opt/homebrew/lib/mod_spatialite.dylib  # macOS
# or
export SPATIALITE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mod_spatialite.so  # Linux

export OPENROUTESERVICE_BASE_URL="https://api.openrouteservice.org/"
export OPENROUTESERVICE_API_KEY="your-api-key" # Optional, if unofficial server is used.
```

### Testing

Run all tests:

```bash
export PYTHONPATH=tests:.
poetry run pytest
```

Run tests in parallel:

```bash
poetry run pytest -n auto
```

Run a specific test file:

```bash
poetry run pytest tests/steps/generators/test_bvgxml_ingester.py
```

Run a specific test:

```bash
poetry run pytest tests/steps/generators/test_bvgxml_ingester.py::test_function_name
```

### Code Quality

Format code with black (line length 99):

```bash
poetry run black .
```

Check formatting:

```bash
poetry run black --check .
```

Type checking with mypy (strict mode):

```bash
poetry run mypy eflips --explicit-package-bases --strict
```

Check poetry configuration:

```bash
poetry check
```

### Documentation

Build documentation locally:

```bash
cd doc/
poetry run sphinx-build -b html . _build
```

View generated docs at `doc/_build/index.html`.

### Prefect

Start local Prefect server (for UI and observability):

```bash
poetry run prefect server start
```

Access UI at http://localhost:4200

## Creating New Pipeline Steps

### Generator Template

```python
from typing import Any, Dict
from sqlalchemy.orm import Session
from eflips.x.framework import Generator


class MyGenerator(Generator):
    def __init__(self, input_files=None, code_version="v1", **kwargs):
        super().__init__(input_files=input_files, code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            "MyGenerator.param_name": "Description of parameter",
        }

    def generate(self, session: Session, params: Dict[str, Any]) -> None:
        # Create database objects
        pass
```

### Modifier Template

```python
from typing import Any, Dict
from sqlalchemy.orm import Session
from eflips.x.framework import Modifier


class MyModifier(Modifier):
    def __init__(self, additional_files=None, code_version="v1", **kwargs):
        super().__init__(additional_files=additional_files, code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            "MyModifier.param_name": "Description of parameter",
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        # Modify database objects
        pass
```

### Analyzer Template

```python
from typing import Any, Dict
from sqlalchemy.orm import Session
from eflips.x.framework import Analyzer


class MyAnalyzer(Analyzer):
    def __init__(self, code_version="v1", **kwargs):
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            "MyAnalyzer.param_name": "Description of parameter",
        }

    def analyze(self, session: Session, params: Dict[str, Any]) -> Any:
        # Analyze database and return results
        return results

    def visualize(self, result: Any):
        # Create visualization (plotly.Figure, folium.Map, etc.)
        return figure
```

## Testing Patterns

Test fixtures available in `tests/conftest.py`:

- `temp_db`: Temporary database file path
- `db_session`: SQLAlchemy session for temporary database
- `test_data_dir`: Path to `data/input/Berlin Testing`
- `gtfs_test_data_dir`: Path to `data/input/GTFS`

Common test pattern:

```python
def test_my_step(temp_db, test_data_dir):
    from eflips.x.framework import PipelineContext

    work_dir = temp_db.parent
    context = PipelineContext(work_dir=work_dir, params={})

    step = MyStep(input_files=[...])
    step.execute(context=context)

    # Verify results
    with context.get_session() as session:
        assert session.query(Model).count() > 0
```

## Important Implementation Details

### Code Version Parameter

All pipeline steps MUST specify a `code_version` parameter for cache invalidation. This should be incremented when step
logic changes:

```python
def __init__(self, code_version="v2", **kwargs):  # Increment when logic changes
    super().__init__(code_version=code_version, **kwargs)
```

### Database Sessions

Always use context managers for database sessions:

```python
with context.get_session() as session:
    # Query database
    vehicles = session.query(Vehicle).all()
# Session automatically closed
```

### Error Handling in Steps

The framework automatically:

- Commits on success
- Attempts commit on failure (for debugging)
- Moves failed databases to `.{timestamp}.failed` extension (Modifiers only)
- Rolls back analyzer sessions (read-only)

### Manual Cache Invalidation

A user can force a step to re-run by deleting its output database file from `work_dir`
(e.g., `step_002_VehicleScheduling.db`). After a Prefect cache hit, `PipelineStep.execute()`
checks whether the output database actually exists. If the file is missing, it logs an `INFO`
message and calls `execute_impl()` directly to regenerate it. All downstream steps will then
re-run naturally because their cache keys depend on the (now changed) input database hash.

### Prefect Artifacts

Steps automatically create Prefect artifacts via `_create_artifact_markdown()`. Override this method to provide custom
artifact content for observability in the Prefect UI.

## Git Workflow

Uses GitHub Flow:

- `main` branch is always deployable
- Feature branches are merged via pull requests
- Branch naming: `feature/description`, `fix/description`

## Dependencies

Core dependencies:

- `eflips-model`: Data model definitions
- `eflips-depot`: Depot simulation
- `eflips-ingest`: Data ingestion utilities
- `eflips-eval`: Evaluation tools
- `eflips-opt`: Optimization algorithms
- `eflips-tco`: Total cost of ownership calculations
- `prefect[dask,sqlalchemy]`: Workflow orchestration (v3.4.6+)

Python versions: 3.12, 3.13

## Common Gotchas

1. **SpatiaLite must be installed and configured** before running tests or pipelines
2. **OpenRouteService API key required or unofficial server used** for routing operations
3. **Code version must be specified** in all pipeline step constructors
4. **Parameters are case-sensitive** and must match class names exactly
5. **Analyzers must NOT modify the database** - framework enforces read-only via temporary copy
6. **Tests use pytest-split in CI** - ensure `--store-durations` when developing for split tests

## Matplotlib usage

Do not use tight_layout, instead use layout="constrained" in plt.subplots().

## Building Analyzers

The visualize() method is optional for things that create tables. Here, we can just assume the user will view (or export) the tables themselves.

In the bare visualize() method, we try to create undecorated, standard matplotlib (or plotly…) figures. We try to do it in a way that is coductive to a subclass pre- and/or postprocessing our visualize method in order to apply a custom visual style. 