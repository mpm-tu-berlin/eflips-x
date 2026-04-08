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

<a href="https://github.com/mpm-tu-berlin/eflips-x/blob/main/Poster.pdf">
  <img style="width: 50%; max-width: 800px; height: auto;" alt="Poster" src="https://github.com/user-attachments/assets/3b4e821a-e843-455a-b7f8-5c1fc3b2ee9f" />
</a>


## What is eflips-x?

eflips-x is a Prefect-based workflow orchestrator that provides a reusable framework for running complex electric bus
network simulations. It integrates multiple eFLIPS components (model, depot, ingest, eval, opt, tco) into configurable,
cacheable pipeline workflows.

The framework provides three types of pipeline steps (defined in `eflips.x.framework`):

- **Generators**: Create new databases from input files (e.g., ingesting GTFS or BVG XML schedules)
- **Modifiers**: Transform existing databases (e.g., vehicle scheduling, depot assignment, simulation)
- **Analyzers**: Extract insights and generate reports without modifying the database

All steps support:
- Automatic caching based on inputs and code versions
- Prefect integration for observability and orchestration
- Sequential execution through database state chaining
- Parameter discovery via the `document_params()` class method

## Recommended Citation

The following publication describes the underlying models and methods used in eFLIPS:

> Heide, L., Guo, S., & Göhlich, D. (2025). From Simulation to Implementation: A Systems Model for Electric Bus Fleet Deployment in Metropolitan Areas. World Electric Vehicle Journal, 16(7), 378. https://doi.org/10.3390/wevj16070378

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

#### Gurobi (Required for Depot Assignment)

The depot assignment step depends on [eflips-opt](https://github.com/mpm-tu-berlin/eflips-opt), which uses the
[Gurobi](https://www.gurobi.com/) mathematical optimization solver. Gurobi is a commercial solver that requires a
license.

**Free academic licenses** are available for researchers, faculty, and students at recognized institutions through the
[Gurobi Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/).

After obtaining a license, activate it:
```bash
# The gurobipy Python package is installed automatically as a dependency.
# You only need to activate your license:
grbgetkey YOUR-LICENSE-KEY
```

Without a valid Gurobi license, pipelines that include the `DepotAssignment` step will fail.

#### OpenRouteService

eflips-x requires OpenRouteService for routing operations (e.g., calculating deadhead trips between depots and
stations):

```bash
export OPENROUTESERVICE_BASE_URL="https://api.openrouteservice.org/"
export OPENROUTESERVICE_API_KEY="your-api-key-here"
```

You can obtain a free API key by registering at [openrouteservice.org](https://openrouteservice.org/). If using
a [self-hosted instance](https://giscience.github.io/openrouteservice/run-instance/), the `OPENROUTESERVICE_BASE_URL`
should point to your instance and the `OPENROUTESERVICE_API_KEY` may be left unset.

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

## Usage

### Setting up a Prefect Server

eflips-x uses Prefect 3 for workflow orchestration. For the best experience, set up a local Prefect server:

```bash
# Start a local Prefect server
poetry run prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
poetry run prefect server start
# Note that now, you will always have to start the server before running pipelines.
```

This will start the Prefect server and UI at http://localhost:4200. The UI provides:
- Real-time flow execution monitoring
- Task-level observability
- Artifact viewing (plots, reports)
- Flow run history and caching insights

For more information, see
the [Prefect self-hosted server documentation](https://docs-3.prefect.io/v3/how-to-guides/self-hosted/server-cli).

### Included Example Flows

The repository includes ready-to-run example flows in `eflips/x/flows/`. These use
[GTFS](https://gtfs.org/) (General Transit Feed Specification) data, a standard open format that is
available for most public transit agencies worldwide.

#### Generalized Multi-Agency GTFS Flow

A configurable flow that reads agency and depot configuration from an Excel file
(`data/input/GTFS/depot_locations.xlsx`) and runs both depot and opportunity charging variants for
each configured agency. Supports parallel execution across agencies.

```bash
# Run all configured agencies
poetry run python -m eflips.x.flows.gtfs_flow

# Filter to a specific agency
poetry run python -m eflips.x.flows.gtfs_flow --agency "Potsdam"

# Enable plot generation
poetry run python -m eflips.x.flows.gtfs_flow --plots

# Run agencies in parallel
poetry run python -m eflips.x.flows.gtfs_flow --parallel
```

See [`eflips/x/flows/gtfs_flow.py`](eflips/x/flows/gtfs_flow.py) for the full source.

### Building Your Own Pipeline

eflips-x provides a framework for building custom pipelines. Here is a GTFS-based example that works
with any transit agency's public GTFS data:

```python
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

from eflips.depot.api import SmartChargingStrategy
from eflips.model import ChargeType
from prefect import flow

from eflips.x.flows import run_steps
from eflips.x.framework import PipelineStep, PipelineContext
from eflips.x.steps.generators import GTFSIngester
from eflips.x.steps.modifiers.bvg_tools import MergeStations
from eflips.x.steps.modifiers.general_utilities import RemoveUnusedData
from eflips.x.steps.modifiers.gtfs_utilities import ConfigureVehicleTypes
from eflips.x.steps.modifiers.scheduling import VehicleScheduling, DepotAssignment
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation


@flow(name="my-gtfs-pipeline")
def my_pipeline():
    work_dir = Path("./output")
    work_dir.mkdir(exist_ok=True)

    params: Dict[str, Any] = {
        "log_level": "INFO",
        # GTFS ingestion settings
        "GTFSIngester.bus_only": True,
        "GTFSIngester.duration": "WEEK",
        "GTFSIngester.agency_name": "My Transit Agency",
        # Vehicle configuration
        "ConfigureVehicleTypes.battery_capacity": 360.0,  # kWh
        "ConfigureVehicleTypes.consumption": 1.5,  # kWh/km
        # Scheduling settings
        "VehicleScheduling.charge_type": ChargeType.DEPOT,
        "VehicleScheduling.minimum_break_time": timedelta(minutes=0),
        "VehicleScheduling.maximum_schedule_duration": timedelta(hours=24),
        # Depot configuration (lon, lat)
        "DepotAssignment.depot_config": [
            {
                "depot_station": (13.3509, 52.5145),  # (lon, lat)
                "name": "Main Depot",
                "vehicle_type": ["default_bus"],
                "capacity": 9999,
            },
        ],
        # Simulation settings
        "Simulation.repetition_period": timedelta(weeks=1),
        "Simulation.smart_charging": SmartChargingStrategy.EVEN,
    }

    steps: List[PipelineStep] = [
        # Phase 1: Ingest and prepare data
        GTFSIngester(input_files=[Path("./data/my_agency.zip")]),
        MergeStations(),
        RemoveUnusedData(),
        ConfigureVehicleTypes(),
        # Phase 2: Schedule and simulate
        VehicleScheduling(),
        DepotAssignment(),
        DepotGenerator(),
        Simulation(),
    ]

    context = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps=steps, context=context)

    return context


if __name__ == "__main__":
    my_pipeline()
```

For pipelines using BVG XML data (a proprietary format), see the BVG-specific example in
[`eflips/x/flows/example.py`](eflips/x/flows/example.py).

### Discovering Parameters

Each step documents its configurable parameters through the `document_params()` class method:

```python
from eflips.x.steps.generators import GTFSIngester

# Print available parameters for a step
for param, description in GTFSIngester.document_params().items():
    print(f" - {param}: {description}")
```

Parameters are passed through the `PipelineContext.params` dictionary. Step-specific parameters are prefixed with the
class name:

```python
params = {
    "log_level": "INFO",  # Global parameter
    "GTFSIngester.bus_only": True,  # Step-specific parameter
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

For a convenient way to generate all available plots at once, use the built-in analysis flow:

```python
from eflips.x.flows import generate_all_plots

generate_all_plots(context=pipeline, output_dir=work_dir / "visualizations")
```

### Pipeline Features

**Automatic Caching**: Steps are automatically cached based on:
- Input file hashes
- Code version
- Python dependencies (poetry.lock)
- Pipeline parameters

To force a specific step to re-run, delete its output database file from `work_dir`
(e.g., `step_002_VehicleScheduling.db`). The framework detects the missing file after a cache
hit and re-executes the step automatically. All downstream steps re-run as a consequence,
since their cache keys depend on the input database hash.

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

| Step | Description |
|---|---|
| `GTFSIngester` | Ingest GTFS (General Transit Feed Specification) data |
| `BVGXMLIngester` | Ingest BVG XML schedule files (proprietary format) |
| `CopyCreator` | Copy an existing database to start a branching workflow |

#### Modifiers

| Step | Description |
|---|---|
| **Scheduling** | |
| `VehicleScheduling` | Create optimal vehicle rotation plans |
| `DepotAssignment` | Assign vehicles to depots (requires Gurobi) |
| `IntegratedScheduling` | Iterative scheduling + depot assignment for feasible opportunity charging schedules (requires Gurobi) |
| `StationElectrification` | Determine which stations to electrify for opportunity charging |
| **General Utilities** | |
| `RemoveUnusedData` | Remove orphaned database objects |
| `AddTemperatures` | Add temperature data for consumption simulation |
| `ConfigureVehicleTypes` | Set battery capacity, consumption, and charging curves (GTFS utility) |
| `CalculateConsumptionScaling` | Calculate consumption scaling factors |
| `RemoveConsumptionLuts` | Remove consumption lookup tables |
| **BVG-Specific** | |
| `SetUpBvgVehicleTypes` | Configure BVG vehicle types |
| `RemoveUnusedRotations` | Remove unused rotations from BVG data |
| `MergeStations` | Merge duplicate or nearby stations |
| `ReduceToNDaysNDepots` | Reduce dataset to N days and N depots |
| **Simulation** | |
| `DepotGenerator` | Generate depot infrastructure objects |
| `Simulation` | Run the vehicle and charging simulation |

#### Analyzers

| Step | Description |
|---|---|
| **Pre-Simulation** (work before simulation) | |
| `RotationInfoAnalyzer` | Overview of rotation data |
| `GeographicTripPlotAnalyzer` | Geographic visualization of trips (Folium map) |
| `SingleRotationInfoAnalyzer` | Detailed view of a single rotation (Cytoscape graph) |
| **Post-Simulation** (require simulation results) | |
| `DepartureArrivalSocAnalyzer` | State of charge at departures/arrivals |
| `SpecificEnergyConsumptionAnalyzer` | Energy consumption analysis |
| `VehicleSocAnalyzer` | Vehicle state of charge over time |
| `DepotEventAnalyzer` | Depot event analysis |
| `DepotActivityAnalyzer` | Depot activity visualization (animated video) |
| `PowerAndOccupancyAnalyzer` | Power demand and occupancy analysis |
| `InteractiveMapAnalyzer` | Interactive map with links to depot/station plots |
| **Export** | |
| `ScenarioJsonExporter` | Export scenario data as JSON |
| `VehicleTypeDepotPlotAnalyzer` | Vehicle type distribution across depots (BVG-specific) |
| `InsufficientChargingTimeAnalyzer` | Identify rotations with insufficient charging time |

## Testing

Testing is done using the `pytest` framework with tests located in the `tests` directory.
Tests use in-memory SQLite databases by default.

```bash
export PYTHONPATH=tests:.
export SPATIALITE_LIBRARY_PATH=/opt/homebrew/lib/mod_spatialite.dylib  # macOS
# or
export SPATIALITE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mod_spatialite.so  # Linux

poetry run pytest
```

## Documentation

Documentation is automatically generated from docstrings using Sphinx and sphinx-autoapi.

To build the documentation locally:

```bash
cd doc/
poetry run sphinx-build -b html . _build
```

The generated HTML documentation will be in `doc/_build/index.html`.

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
