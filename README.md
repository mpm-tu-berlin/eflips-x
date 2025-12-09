[![Tests](https://github.com/mpm-tu-berlin/eflips-x/actions/workflows/unittests.yml/badge.svg)](https://github.com/mpm-tu-berlin/eflips-x/actions/workflows/unittests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# eflips-x

---

Part of the [eFLIPS/simBA](https://github.com/stars/ludgerheide/lists/ebus2030) list of projects.

---

eFLIPS has been developed within several research projects at the Department of Methods of Product Development and
Mechatronics at the Technische Universität Berlin (see https://www.tu.berlin/mpm/forschung/projekte/eflips).

With eFLIPS, electric fleets and depots can be simulated, planned and designed.
This repository contains the workflow orchestrator for running eFLIPS electric city bus network simulations.

![eflips_overview](https://user-images.githubusercontent.com/74250473/236144949-4192e840-0e3d-4b65-9f78-af8e01ad9ef3.png)

## What is eflips-x?

eflips-x is a Prefect-based workflow orchestrator that provides a reusable framework for running complex electric bus
network simulations. It integrates multiple eFLIPS components (model, depot, ingest, eval, opt, tco) into configurable,
cacheable pipeline workflows.

The framework provides three types of pipeline steps (defined in `eflips.x.framework`):

- **Generators**: Create new databases from input files (e.g., ingesting BVG XML schedules)
- **Modifiers**: Transform existing databases (e.g., vehicle scheduling, depot assignment, simulation)
- **Analyzers**: Extract insights and generate reports without modifying the database

All steps support:
- Automatic caching based on inputs and code versions
- Prefect integration for observability and orchestration
- Sequential execution through database state chaining
- Parameter discovery via the `document_params()` class method

## Supported Platforms

- macOS (Intel and Apple Silicon)
- Linux (Debian/Ubuntu and similar distributions)

**Note**: Windows is not officially supported.

## Installation

### Prerequisites

#### SpatiaLite

eflips-x requires SpatiaLite for spatial database operations. Install and configure it for your platform:

**macOS** (using Homebrew):
```bash
brew install spatialite-tools
```

Then set the library path:
```bash
export SPATIALITE_LIBRARY_PATH=/opt/homebrew/lib/mod_spatialite.dylib
```

**Debian/Ubuntu**:
```bash
sudo apt install libsqlite3-mod-spatialite
```

Then set the library path:
```bash
export SPATIALITE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mod_spatialite.so
```

**Tip**: Add the `SPATIALITE_LIBRARY_PATH` export to your shell configuration file (e.g., `~/.bashrc`, `~/.zshrc`) to
make it permanent.

### Installing eflips-x

1. Clone this git repository (or [download a specific release](https://github.com/mpm-tu-berlin/eflips-x/releases))

    ```bash
    git clone git@github.com:mpm-tu-berlin/eflips-x.git
    cd eflips-x
    ```

2. Install the packages using [Poetry](https://python-poetry.org/) (recommended). Poetry can be installed according to
   the instructions listed [here](https://python-poetry.org/docs/#installing-with-the-official-installer).

   **Supported Python versions**: 3.12, 3.13

    ```bash
    poetry env use 3.12  # or 3.13
    poetry install
    ```

3. Set up environment variables for OpenRouteService (required for routing operations):

    ```bash
    export OPENROUTESERVICE_BASE_URL="https://api.openrouteservice.org/"
    export OPENROUTESERVICE_API_KEY="your-api-key-here"
    ```

   You can obtain a free API key by registering at [openrouteservice.org](https://openrouteservice.org/). If using
   a [self-hosted instance](https://giscience.github.io/openrouteservice/run-instance/), the `OPENROUTESERVICE_BASE_URL`
   should point to your instance and the `OPENROUTESERVICE_API_KEY` may be left unset.

## Usage

### Setting up a Prefect Server

eflips-x uses Prefect 3 for workflow orchestration. For the best experience, set up a local Prefect server:

```bash
# Start a local Prefect server
poetry run prefect server start
```

This will start the Prefect server and UI at http://localhost:4200. The UI provides:
- Real-time flow execution monitoring
- Task-level observability
- Artifact viewing (plots, reports)
- Flow run history and caching insights

For more information, see
the [Prefect self-hosted server documentation](https://docs-3.prefect.io/v3/how-to-guides/self-hosted/server-cli).

### Creating a Pipeline

eflips-x provides a framework for building custom pipelines. Here's a basic example:

```python
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

from eflips.model import ChargeType
from prefect import flow

from eflips.x.framework import PipelineStep, PipelineContext
from eflips.x.steps.generators import BVGXMLIngester
from eflips.x.steps.modifiers.bvg_tools import (
   SetUpBvgVehicleTypes,
   RemoveUnusedRotations,
   MergeStations,
)
from eflips.x.steps.modifiers.general_utilities import RemoveUnusedData, AddTemperatures
from eflips.x.steps.modifiers.scheduling import VehicleScheduling, DepotAssignment
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation


@flow
def run_steps(context: PipelineContext, steps: List[PipelineStep]) -> None:
   """Run a sequence of pipeline steps."""
   for step in steps:
      step.execute(context=context)


@flow(name="my-eflips-pipeline")
def my_pipeline():
   # Create a working directory and set up parameters
    work_dir = Path("./output")
    work_dir.mkdir(exist_ok=True)

   params: Dict[str, Any] = {
      "log_level": "INFO",
      "VehicleScheduling.charge_type": ChargeType.DEPOT,
      "VehicleScheduling.minimum_break_time": timedelta(minutes=10),
      "VehicleScheduling.maximum_schedule_duration": timedelta(hours=8),
   }

   # Build the step list
   steps: List[PipelineStep] = []

    # Step 1: Ingest schedule data
    xml_files = list(Path("./data/input").glob("*.xml"))
   steps.append(BVGXMLIngester(input_files=xml_files))

   # Step 2: Set up BVG vehicle types
   steps.append(SetUpBvgVehicleTypes())

   # Step 3: Clean up data
   steps.append(RemoveUnusedRotations())
   steps.append(MergeStations())
   steps.append(RemoveUnusedData())

   # Step 4: Add temperature data (required for extended consumption simulation)
   steps.append(AddTemperatures())

   # Step 5: Run vehicle scheduling
   steps.append(VehicleScheduling())

   # Step 6: Assign depots
   steps.append(DepotAssignment())

   # Step 7: Generate depot objects and run simulation
   steps.append(DepotGenerator())
   steps.append(Simulation())

   # Create context and run pipeline
   context = PipelineContext(work_dir=work_dir, params=params)
   run_steps(steps=steps, context=context)

   return context


if __name__ == "__main__":
    my_pipeline()
```

### Discovering Parameters

Each step documents its configurable parameters through the `document_params()` class method:

```python
from eflips.x.steps.generators import BVGXMLIngester

# Print available parameters for a step
for param, description in BVGXMLIngester.document_params().items():
   print(f" - {param}: {description}")
```

Parameters are passed through the `PipelineContext.params` dictionary. Step-specific parameters are prefixed with the
class name:

```python
params = {
   "log_level": "INFO",  # Global parameter
   "BVGXMLIngester.multithreading": True,  # Step-specific parameter
   "VehicleScheduling.charge_type": ChargeType.DEPOT,
}
```

### Analyzing Results

Analyzers return results from `execute()` and provide a `visualize()` method for creating plots:

```python
import plotly
import folium
from eflips.x.steps.analyzers import (
   RotationInfoAnalyzer,
   DepartureArrivalSocAnalyzer,
   GeographicTripPlotAnalyzer,
   SpecificEnergyConsumptionAnalyzer,
)

# Run analyzers after the simulation pipeline
output_directory = work_dir / "analysis"
output_directory.mkdir(parents=True, exist_ok=True)

for analyzer_class in (
      RotationInfoAnalyzer,
      GeographicTripPlotAnalyzer,
      DepartureArrivalSocAnalyzer,
      SpecificEnergyConsumptionAnalyzer,
):
   analyzer = analyzer_class()
   result = analyzer.execute(context=pipeline)
   vis = analyzer.visualize(result)

   output_file = output_directory / f"{analyzer_class.__name__}.html"
   if isinstance(vis, plotly.graph_objs._figure.Figure):
      vis.write_html(output_file)
   elif isinstance(vis, folium.Map):
      vis.save(str(output_file))
```

Some analyzers require additional parameters. For example, `VehicleSocAnalyzer` needs a vehicle ID:

```python
from eflips.model import Vehicle
from eflips.x.steps.analyzers import VehicleSocAnalyzer

# Get vehicle IDs from the database
with pipeline.get_session() as session:
   vehicle_ids = [v.id for v in session.query(Vehicle).all()]

# Analyze each vehicle
for vehicle_id in vehicle_ids:
   pipeline.params["VehicleSocAnalyzer.vehicle_id"] = vehicle_id
   analyzer = VehicleSocAnalyzer()
   result = analyzer.execute(context=pipeline)
   vis = analyzer.visualize(*result)
   vis.write_html(f"vehicle_{vehicle_id}_soc.html")
```

### Pipeline Features

**Automatic Caching**: Steps are automatically cached based on:
- Input file hashes
- Code version
- Python dependencies (poetry.lock)
- Pipeline parameters

**Database Chaining**: Each step automatically receives the database from the previous step through the
`PipelineContext`. Use `context.get_session()` to access the current database:

```python
with pipeline.get_session() as session:
   depots = session.query(Depot).all()
   for depot in depots:
      print(f"Depot: {depot.name}, Areas: {len(depot.areas)}")
```

### Available Steps

#### Generators

- `BVGXMLIngester`: Ingest BVG XML schedule files

#### Modifiers

- **BVG Tools**: `SetUpBvgVehicleTypes`, `RemoveUnusedRotations`, `MergeStations`
- **General Utilities**: `RemoveUnusedData`, `AddTemperatures`
- **Scheduling**: `VehicleScheduling`, `DepotAssignment`
- **Simulation**: `DepotGenerator`, `Simulation`

#### Analyzers

- `RotationInfoAnalyzer`: Overview of rotation data
- `SingleRotationInfoAnalyzer`: Detailed view of a single rotation (Cytoscape graph)
- `DepartureArrivalSocAnalyzer`: State of charge at departures/arrivals
- `SpecificEnergyConsumptionAnalyzer`: Energy consumption analysis
- `VehicleSocAnalyzer`: Vehicle state of charge over time
- `DepotActivityAnalyzer`: Depot activity visualization (animated)
- `DepotEventAnalyzer`: Depot event analysis
- `GeographicTripPlotAnalyzer`: Geographic visualization of trips (Folium map)
- `PowerAndOccupancyAnalyzer`: Power demand and occupancy analysis

## Testing

Testing is done using the `pytest` framework with tests located in the `tests` directory.

**Important**: Tests will use the database specified in the `DATABASE_URL` environment variable. By default, tests use
an in-memory SQLite database, but you should ensure your configuration is correct.

To run the tests:

```bash
export PYTHONPATH=tests:.
export SPATIALITE_LIBRARY_PATH=/opt/homebrew/lib/mod_spatialite.dylib  # macOS
# or
export SPATIALITE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mod_spatialite.so  # Linux

export OPENROUTESERVICE_BASE_URL="https://api.openrouteservice.org/"
export OPENROUTESERVICE_API_KEY="your-api-key"

poetry run pytest
```

## Documentation

Documentation is automatically generated from docstrings using Sphinx and sphinx-autoapi.

### Building Documentation Locally

To build the documentation locally:

```bash
cd doc/
poetry run sphinx-build -b html . _build
```

The generated HTML documentation will be in `doc/_build/index.html`.

### Online Documentation

Documentation is available on Read the Docs (if configured for this repository).

## Development

We utilize the [GitHub Flow](https://docs.github.com/get-started/quickstart/github-flow) branching structure. This means
that the `main` branch is always deployable and that all development happens in feature branches. The feature branches
are merged into `main` via pull requests.

### Code Formatting

We use [black](https://black.readthedocs.io/en/stable/) for code formatting with a line length of 99 characters. You can
use [pre-commit](https://pre-commit.com/) to ensure the code is formatted correctly before committing:

```bash
poetry run pre-commit install
```

### Type Checking

We use [mypy](https://mypy.readthedocs.io/) for static type checking in strict mode. Run type checks with:

```bash
poetry run mypy eflips/
```

### Dependency Management

Please ensure that your `poetry.lock` and `pyproject.toml` files are consistent before committing:

```bash
poetry check
```

This is also checked by pre-commit if configured.

## License

This project is licensed under the AGPLv3 license - see the [LICENSE.md](LICENSE.md) file for details.

## Funding Notice

This code was developed as part of the project [eBus2030+](https://www.now-gmbh.de/projektfinder/e-bus-2030/) funded by
the Federal German Ministry for Digital and Transport (BMDV) under grant number 03EMF0402.