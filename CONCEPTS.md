# eflips-x Concepts

A developer's guide to understanding the architecture, design patterns, and domain concepts of eflips-x.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Domain Concepts](#2-domain-concepts)
3. [Core Architectural Concepts](#3-core-architectural-concepts)
4. [Framework Components Deep Dive](#4-framework-components-deep-dive)
5. [Flows: Orchestrating Pipelines](#5-flows-orchestrating-pipelines)
6. [Building Custom Solutions](#6-building-custom-solutions)
7. [Best Practices & Design Principles](#7-best-practices--design-principles)
8. [Common Patterns & Recipes](#8-common-patterns--recipes)
9. [Integration with eFLIPS Ecosystem](#9-integration-with-eflips-ecosystem)
10. [Advanced Topics](#10-advanced-topics)

---

## 1. Introduction

### What is eflips-x?

eflips-x is a **Prefect-based workflow orchestrator** for running complex electric bus network simulations. It provides a reusable framework that integrates multiple eFLIPS components into configurable, cacheable pipeline workflows.

Part of the broader [eFLIPS/simBA](https://github.com/stars/ludgerheide/lists/ebus2030) ecosystem, eflips-x was developed at the Department of Methods of Product Development and Mechatronics at Technische Universität Berlin for simulating, planning, and designing electric fleets and depots.

### The Problem It Solves

Electric bus network planning involves complex, multi-step workflows:
1. Ingesting transit schedule data from various formats
2. Configuring vehicle and infrastructure parameters
3. Running vehicle scheduling algorithms
4. Assigning vehicles to depots
5. Simulating energy consumption and charging
6. Analyzing results and generating reports

Without a framework, developers face:
- **Repetitive boilerplate**: Database management, caching, parameter handling
- **Poor reproducibility**: Hard to track which inputs produced which outputs
- **Slow iteration**: Re-running expensive simulations unnecessarily
- **Limited observability**: Difficulty monitoring long-running workflows

eflips-x solves these problems by providing:
- **Database state chaining**: Immutable progression through pipeline stages
- **Automatic caching**: Smart invalidation based on inputs, code, and dependencies
- **Prefect integration**: Workflow orchestration, monitoring, and observability
- **Modular architecture**: Compose and reuse pipeline steps

### The eFLIPS Ecosystem

eflips-x orchestrates several eFLIPS components:
- **eflips.model**: Database schema and ORM for transit networks
- **eflips.depot**: Depot layout generation and optimization
- **eflips.ingest**: Data ingestion utilities for various formats
- **eflips.eval**: Evaluation and optimization algorithms
- **eflips.opt**: Optimization solvers
- **eflips.tco**: Total cost of ownership calculations

See [Section 9](#9-integration-with-eflips-ecosystem) for details on how these integrate.

---

## 2. Domain Concepts

Before diving into the framework architecture, it's helpful to understand the electric bus planning domain. This section introduces key terminology used throughout eflips-x.

### 2.1 Transit Network Elements

**Route**: A fixed path through a transit network, typically identified by a route number or name (e.g., "Line 100").

**Trip**: A single journey along a route from start to end at a specific time. A route may have dozens of trips per day.

**Rotation**: A sequence of trips assigned to a single vehicle during its operating day. A rotation represents a vehicle's complete work schedule, including trips and break times.

**Station**: A physical location where buses stop to pick up or drop off passengers. Stations have geographic coordinates and may serve multiple routes.

### 2.2 Electric Bus Operations

**Vehicle Type**: A category of buses with shared characteristics: battery capacity, energy consumption rate, charging capabilities, and physical dimensions.

**Schedule**: The complete set of trips and their timings that define service for a transit network.

**State of Charge (SoC)**: The current battery level of an electric vehicle, typically expressed as a percentage (0-100%) or in kWh.

**Charging Types**:
- **Depot Charging**: Vehicles charge at a central depot between rotations or during extended breaks
- **Opportunity Charging**: Vehicles charge during short breaks at strategically placed charging stations along routes

**Energy Consumption**: The amount of electrical energy (kWh) used by a vehicle, influenced by route characteristics (distance, elevation, traffic), temperature, and driving behavior.

### 2.3 Infrastructure

**Depot**: A central facility where buses are stored, maintained, and charged. Depots contain multiple charging areas and have capacity constraints.

**Area**: A subdivision within a depot where vehicles park and charge. Areas have specific charging infrastructure and occupancy limits.

**Charging Infrastructure**: The physical equipment (chargers, power connections) that supplies electricity to vehicle batteries. Characterized by maximum power output and charging protocols.

### 2.4 Simulation Outputs

**Event**: A time-stamped occurrence in the simulation, such as:
- Vehicle arrival at/departure from a station
- Charging start/end
- Trip start/end
- Vehicle entering/leaving a depot area

**Consumption Analysis**: Evaluation of energy usage patterns, specific energy consumption (kWh/km), and factors affecting consumption.

**Power Demand**: The electrical power required at charging infrastructure over time, used for grid planning and infrastructure sizing.

### 2.5 The Electric Bus Planning Workflow

A typical workflow progresses through these stages:

```
Schedule Data → Ingestion → Vehicle/Depot Setup → Scheduling → Simulation → Analysis
     ↓              ↓                ↓                 ↓             ↓           ↓
  (GTFS, XML)   Generator        Modifier          Modifier      Modifier   Analyzer
```

1. **Schedule Ingestion**: Import transit schedule data (GTFS, BVG XML, etc.)
2. **Configuration**: Set up vehicle types, depot locations, and infrastructure parameters
3. **Vehicle Scheduling**: Assign trips to vehicles, creating rotations
4. **Depot Assignment**: Assign rotations to specific depots
5. **Simulation**: Simulate vehicle movements, energy consumption, and charging
6. **Analysis**: Generate reports, visualizations, and insights

Each stage transforms the database state, creating a clear data lineage from inputs to outputs.

---

## 3. Core Architectural Concepts

eflips-x is built on several key architectural principles that shape how pipelines are designed and executed.

### 3.1 The Pipeline Philosophy

#### Why Database-Centric?

eflips-x uses an **immutable database state progression** pattern. Each pipeline step creates a new database snapshot rather than modifying the previous one:

```
step_001_BVGXMLIngester.db
  → step_002_SetUpBvgVehicleTypes.db
    → step_003_RemoveUnusedData.db
      → step_004_VehicleScheduling.db
        → ...
```

**Benefits**:

1. **Debugging**: Inspect intermediate states to diagnose issues
2. **Rollback**: Re-run from any point without starting over
3. **Reproducibility**: Clear data lineage from inputs to outputs
4. **Partial Re-runs**: Skip expensive early steps when cached

**Trade-offs**:
- Disk space usage (mitigated by SQLite's compact format)
- Copy overhead (mitigated by copy-on-write for Modifiers)

This pattern treats databases as **values**, not mutable state containers—a functional programming concept applied to data pipelines.

#### Why Prefect Integration?

Prefect provides the orchestration layer for eflips-x:

**Workflow Observability**:
- Real-time monitoring through Prefect UI
- Task-level execution tracking
- Artifact creation (plots, reports, progress updates)

**Automatic Caching**:
- Multi-factor cache key computation
- Intelligent cache invalidation
- Transparent cache hits/misses

**Task Parallelization**:
- Concurrent execution of independent tasks
- ProcessPool/ThreadPool task runners
- Future-based result handling

**Reliability**:
- Automatic retries on transient failures
- Flow state persistence
- Execution history

The integration is lightweight: eflips-x steps wrap themselves in Prefect tasks automatically, requiring minimal boilerplate from developers.

### 3.2 The Three-Step Pattern

eflips-x defines three types of pipeline steps, each with distinct semantics:

```
Generator: ∅ → Database
Modifier:  Database₁ → Database₂
Analyzer:  Database → Results
```

#### Design Rationale

**Separation of Concerns**:
- **Generators** handle external data ingestion
- **Modifiers** perform transformations and computations
- **Analyzers** extract insights without side effects

**Composability**:
Mix and match steps freely. Any Modifier can follow any Generator. Any Analyzer can read any compatible database.

**Caching Granularity**:
- Generators: Cache based on input files + code + parameters
- Modifiers: Cache based on input DB + code + parameters
- Analyzers: Cache based on input DB + code + parameters (but don't create new DBs)

**Read-Only Guarantee**:
Analyzers cannot corrupt the pipeline state. They work on temporary copies, ensuring data integrity.

#### When to Use Each

**Use a Generator when**:
- Starting a pipeline from external data
- Ingesting schedules (GTFS, XML, CSV)
- Creating synthetic test data
- The operation doesn't depend on a previous database

**Use a Modifier when**:
- Transforming existing data
- Running scheduling algorithms
- Simulating vehicle operations
- Enriching data (adding temperatures, distances)
- The operation builds on previous results

**Use an Analyzer when**:
- Generating reports or visualizations
- Computing statistics
- Exporting data
- The operation produces outputs but doesn't modify the pipeline state

### 3.3 PipelineContext: The State Manager

`PipelineContext` is the central data structure that flows through pipeline execution:

```python
@dataclass
class PipelineContext:
    work_dir: Path                      # Output directory
    params: Dict[str, Any]              # Configuration parameters
    current_db: Optional[Path]          # Current database state
    step_count: int                     # Sequential step counter
    artifacts: Dict[str, Any]           # Execution artifacts
```

#### Why Centralized State Management?

**Single Source of Truth**:
The context tracks the current database state. Every step knows where to read from and where to write to.

**Parameter Distribution**:
All steps receive the same `params` dictionary, enabling:
- Global parameters (e.g., `log_level`)
- Step-specific parameters (e.g., `VehicleScheduling.charge_type`)
- Runtime parameter discovery via `document_params()`

**Artifact Collection**:
Steps can store arbitrary data in `context.artifacts` for later retrieval or debugging.

**Sequential Naming**:
The `step_count` ensures deterministic, ordered database naming, making it easy to trace pipeline progression.

#### Design Patterns

**Dependency Injection Container**: The context provides database sessions, working directories, and configuration without steps managing these directly.

**State Pattern**: `current_db` transitions through states as steps execute, with the context managing state validity.

**Context Object Pattern**: Encapsulates the execution environment, passed explicitly rather than accessed globally.

#### Usage Example

```python
# Steps query the context for configuration
def modify(self, session, params):
    charge_type = params.get(f"{self.__class__.__name__}.charge_type", ChargeType.DEPOT)

# Flows query intermediate results
with context.get_session() as session:
    vehicles = session.query(Vehicle).all()
    print(f"Pipeline has {len(vehicles)} vehicles")
```

### 3.4 Caching Strategy

eflips-x implements **multi-factor cache invalidation** to balance speed and correctness.

#### Cache Key Composition

Each step type computes a SHA256 hash from multiple inputs:

**Generators**:
```
work_dir name + output_db name + class name + code_version +
poetry.lock hash + input file hashes + parameters hash
```

**Modifiers**:
```
work_dir name + output_db name + class name + code_version +
poetry.lock hash + input DB hash + additional file hashes + parameters hash
```

**Analyzers**:
```
work_dir name + class name + code_version + poetry.lock hash +
input DB hash + additional file hashes + parameters hash
```

#### Why This Matters

**Correctness**: Cache is invalidated when:
- Input data changes (file content hashes)
- Code logic changes (`code_version` increment)
- Dependencies change (`poetry.lock` hash)
- Parameters change (serialized hash)
- Previous results change (input DB hash for Modifiers)

**Development Speed**: Iterate quickly by caching expensive operations (simulation, routing, optimization) while ensuring correctness.

**Reproducibility**: Same inputs + same code = same outputs, guaranteed by cryptographic hashing.

#### Manual Versioning

Developers must increment `code_version` when changing step logic:

```python
class VehicleScheduling(Modifier):
    def __init__(self, code_version: str = "v3", **kwargs):  # Increment when logic changes
        super().__init__(code_version=code_version, **kwargs)
```

The alternative would entail an awful amount of Python self-introspection to detect code changes automatically. Maybe in the future…

---

## 4. Framework Components Deep Dive

This section explores the implementation details of each framework component.

### 4.1 PipelineStep Abstract Base Class

`PipelineStep` is the parent class for all pipeline operations, defining the template method pattern:

```python
class PipelineStep(ABC):
    def execute(self, context: PipelineContext) -> None:
        # 1. Configure logging
        self.set_log_level(context)

        # 2. Create Prefect task wrapper (once)
        if self._prefect_task is None:
            self._create_prefect_task()

        # 3. Generate output database path
        output_db = context.get_next_db_path(self.__class__.__name__)

        # 4. Execute via Prefect (with caching)
        self._prefect_task(context=context, output_db=output_db)

        # 5. Update context with new database
        context.set_current_db(output_db)
```

#### Key Features

**Template Method Pattern**: `execute()` defines the workflow structure, with subclasses implementing specific behavior via abstract methods:
- `execute_impl()`: Core logic
- `compute_cache_key()`: Cache key computation
- `document_params()`: Parameter documentation
- `_create_artifact_markdown()`: Observability reporting

**Self-Documentation**: Every step must implement `document_params()`, enabling runtime discovery:

```python
@classmethod
def document_params(cls) -> Dict[str, str]:
    return {
        "log_level": "Global logging level",
        f"{cls.__name__}.charge_type": "Charging strategy (DEPOT or OPPORTUNITY)",
        f"{cls.__name__}.minimum_break_time": "Minimum break duration for scheduling",
    }
```

**Logging Hierarchy**: Steps can have step-specific or global log levels:
1. Check `params["{ClassName}.log_level"]`
2. Fall back to `params["log_level"]`
3. Default to `WARNING`

**Prefect Task Wrapping**: Steps automatically wrap themselves in Prefect tasks with caching:

```python
@task(
    name=self.__class__.__name__,
    cache_key_fn=lambda ctx, params: self.compute_cache_key(...) if self.cache_enabled else None,
)
def task_wrapper(context, output_db):
    result = self.execute_impl(context, output_db)
    create_markdown_artifact(...)  # For Prefect UI
    return result
```

This happens transparently—developers don't interact with Prefect directly.

### 4.2 Generator

Generators create new databases from scratch, without depending on previous pipeline state.

#### Characteristics

**No Input Database**: Cache key excludes previous database (unlike Modifiers).

**Input Files**: Typically accept `input_files` list for data ingestion.

**Schema Creation**: Responsible for creating database schema via `Base.metadata.create_all()`.

**Population**: Implement `generate(session, params)` to populate the database.

#### Implementation Pattern

```python
class MyDataIngester(Generator):
    def __init__(self, input_files: List[Path], code_version: str = "v1", **kwargs):
        super().__init__(input_files=input_files, code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.encoding": "File encoding (default: utf-8)",
        }

    def generate(self, session: Session, params: Dict[str, Any]) -> None:
        # Read input files and populate database
        for input_file in self.input_files:
            data = parse_file(input_file, params)
            for item in data:
                session.add(Route(...))
                session.add(Trip(...))
        session.flush()  # Framework commits
```

#### Error Handling

If generation fails, the framework attempts to commit anyway (for debugging), then re-raises the exception. The incomplete database remains for inspection.

### 4.3 Modifier

Modifiers transform an existing database into a new one using a **copy-on-write** pattern.

#### Characteristics

**Input Database Required**: Cache key includes hash of `context.current_db`.

**Copy-on-Write**: Framework copies input DB to output DB before modification, preserving intermediate states.

**Additional Files**: Can accept configuration files (e.g., depot locations, temperature data).

**Modification**: Implement `modify(session, params)` to transform the database.

#### Implementation Pattern

```python
class MyTransformation(Modifier):
    def __init__(self, code_version: str = "v1", **kwargs):
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.threshold": "Filtering threshold",
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        # Transform existing data
        threshold = params.get(f"{self.__class__.__name__}.threshold", 0.5)

        rotations = session.query(Rotation).all()
        for rotation in rotations:
            if rotation.total_distance < threshold:
                session.delete(rotation)

        session.flush()  # Framework commits
```

#### Failed Database Handling

If modification fails:
1. Framework attempts to commit (for debugging)
2. Moves incomplete database to `.{timestamp}.failed` extension
3. Re-raises exception

This allows developers to inspect failed states without cluttering the working directory.

#### Session Management

**Flush, Don't Commit**: Steps call `session.flush()` to persist changes within the transaction. The framework handles `session.commit()`.

**No Rollback Needed**: The framework manages rollback on exceptions.

### 4.4 Analyzer

Analyzers read databases and produce reports/visualizations without modifying pipeline state.

#### Characteristics

**Read-Only Execution**: Work on temporary database copies to prevent accidental writes.

**Return Values**: Unlike Generators/Modifiers, `execute()` returns results instead of updating `context.current_db`.

**Visualization Contract**: Typically implement a static `visualize(results)` method for creating plots.

**No Output Database**: Cache key ignores output database path.

#### Implementation Pattern

```python
class MyAnalysis(Analyzer):
    def __init__(self, code_version: str = "v1", **kwargs):
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.vehicle_id": "Vehicle ID to analyze",
        }

    def analyze(self, session: Session, params: Dict[str, Any]) -> Any:
        # Extract data from database
        vehicle_id = params[f"{self.__class__.__name__}.vehicle_id"]
        vehicle = session.query(Vehicle).filter_by(id=vehicle_id).one()

        events = session.query(Event).filter_by(vehicle_id=vehicle_id).all()

        # Return structured results
        return {
            "vehicle": vehicle,
            "events": events,
            "total_energy": sum(e.energy_charged for e in events if e.event_type == "charging"),
        }

    @staticmethod
    def visualize(results: Dict[str, Any]) -> plotly.graph_objs.Figure:
        # Create visualization from results
        fig = go.Figure()
        fig.add_trace(go.Scatter(...))
        return fig
```

#### Usage Pattern

```python
# In a flow
analyzer = MyAnalysis()
results = analyzer.execute(context=context)  # Returns data
vis = analyzer.visualize(results)            # Create plot
vis.write_html("output.html")                # Save
```

#### Temporary Database Protection

Analyzers execute on a temporary copy of the current database:

```python
def execute(self, context: PipelineContext) -> Any:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_db = Path(temp_dir) / f"{self.__class__.__name__}_temp.db"
        shutil.copy2(context.current_db, output_db)  # Temporary copy

        # Execute analysis on copy
        result = self._prefect_task(context=context, output_db=output_db)
    # Temporary DB deleted automatically
    return result
```

This ensures even poorly-behaved analyzers can't corrupt pipeline state.

---

## 5. Flows: Orchestrating Pipelines

Flows are the orchestration layer that composes pipeline steps into complete workflows.

### 5.1 What Are Flows?

**Definition**: Flows are Prefect `@flow` decorated functions that coordinate the execution of pipeline steps.

**Difference from Steps**:

| Aspect | Flows | Steps |
|--------|-------|-------|
| **Abstraction Level** | High-level orchestration | Low-level operations |
| **What They Do** | Compose and execute steps | Create, transform, or analyze databases |
| **Reusability** | Nest and compose flows | Reuse steps across flows |
| **Execution Model** | Prefect task runners for parallelism | Sequential execution within flows |
| **State Management** | Manage `PipelineContext` | Receive context, update state |
| **Database Interaction** | Don't touch databases directly | Core responsibility |
| **Location** | `eflips/x/flows/` | `eflips/x/steps/` |

Flows are **composition** (arranging steps), steps are **operations** (doing work).

### 5.2 Flow Patterns

#### Pattern 1: Linear Sequential Flow

The simplest pattern: execute steps one after another.

```python
from prefect import flow
from eflips.x.framework import PipelineContext

@flow(name="simple-pipeline")
def simple_pipeline():
    work_dir = Path("./output")
    params = {"log_level": "INFO"}

    steps = [
        BVGXMLIngester(input_files=[...]),
        SetUpBvgVehicleTypes(),
        VehicleScheduling(),
        Simulation(),
    ]

    context = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps=steps, context=context)
```

**When to use**: Straightforward pipelines where each step depends on the previous one and no intermediate queries are needed.

#### Pattern 2: Two-Phase Execution

Split execution into phases when later steps need information from earlier results.

```python
@flow(name="two-phase-pipeline")
def two_phase_pipeline():
    context = PipelineContext(work_dir=work_dir, params=params)

    # Phase 1: Data ingestion and setup
    initial_steps = [
        GTFSIngester(input_files=[...]),
        RemoveUnusedData(),
    ]
    run_steps(steps=initial_steps, context=context)

    # Query intermediate results
    with context.get_session() as session:
        vehicle_types = session.query(VehicleType).all()
        # Use results to configure next phase
        params["VehicleScheduling.vehicle_types"] = [vt.id for vt in vehicle_types]

    # Phase 2: Scheduling and simulation
    simulation_steps = [
        VehicleScheduling(),
        DepotAssignment(),
        Simulation(),
    ]
    run_steps(steps=simulation_steps, context=context)
```

**When to use**:
- Later steps need data computed by earlier steps
- Dynamic configuration based on intermediate results
- Conditional step inclusion

**Example**: `eflips/x/flows/swu_gtfs_flow.py`

#### Pattern 3: Parallel Analysis

Use Prefect tasks to run independent analyzers concurrently.

```python
from prefect import flow, task
from prefect.task_runners import ProcessPoolTaskRunner

@task
def execute_analyzer(analyzer_class, context, output_file):
    analyzer = analyzer_class()
    result = analyzer.execute(context=context)
    vis = analyzer.visualize(result)
    vis.write_html(output_file)

@flow(task_runner=ProcessPoolTaskRunner())
def parallel_analysis(context, output_dir):
    all_futures = []

    # Submit all analyzers in parallel
    for analyzer_class in [RotationInfoAnalyzer, DepartureArrivalSocAnalyzer, ...]:
        future = execute_analyzer.submit(analyzer_class, context, output_file)
        all_futures.append(future)

    # Wait for all to complete
    from prefect import wait
    wait(all_futures)
```

**When to use**:
- Multiple independent analyses
- Expensive computations that can run in parallel
- Report generation

**Example**: `eflips/x/flows/analysis_flow.py`

#### Pattern 4: Flow Nesting

Reuse flows as building blocks within other flows.

```python
from eflips.x.flows import run_steps, generate_all_plots

@flow(name="complete-workflow")
def complete_workflow():
    # Execute main pipeline
    steps = [Generator(), Modifier1(), Modifier2()]
    context = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps=steps, context=context)  # Nested flow

    # Execute analysis subflow
    generate_all_plots(  # Nested flow
        context=context,
        output_dir=work_dir / "analysis",
        include_videos=False
    )
```

**When to use**:
- Modular workflow design
- Reusing common patterns (like analysis)
- Separating concerns

**Reusable flows**:
- `run_steps()`: Execute a sequence of steps
- `generate_all_plots()`: Generate all standard visualizations

### 5.3 Flows vs. Steps: Decision Guide

**Create a Flow when you want to**:
- Compose multiple steps into a complete workflow
- Parallelize independent operations
- Implement conditional execution logic
- Query intermediate results between steps
- Provide a complete, runnable pipeline for a use case

**Create a Step when you want to**:
- Perform a specific data operation
- Create a reusable transformation or analysis
- Integrate a new algorithm or external tool
- Provide a cacheable unit of work

**Rule of thumb**: If it orchestrates, it's a flow. If it operates on data, it's a step.

---

## 6. Building Custom Solutions

This section provides practical guidance for adapting eflips-x to your specific simulation needs.

### 6.1 Adapting eflips-x for Your Use Case

There are three strategies for customization, often used in combination:

#### Strategy 1: Chain Existing Modules

**When**: Your workflow can be built entirely from existing steps by configuring parameters.

**Process**:
1. Review available steps (see README or `eflips/x/steps/`)
2. Determine the required sequence
3. Configure parameters for your scenario
4. Create a flow that chains them together

**Example**: Different charging strategies using existing components:

```python
# Depot charging scenario
params["VehicleScheduling.charge_type"] = ChargeType.DEPOT
params["VehicleScheduling.minimum_break_time"] = timedelta(hours=4)

# Opportunity charging scenario
params["VehicleScheduling.charge_type"] = ChargeType.OPPORTUNITY
params["VehicleScheduling.minimum_break_time"] = timedelta(minutes=10)
```

**Benefits**:
- No code to write or maintain
- Fully tested components
- Immediate caching and observability

**Limitations**:
- Constrained to existing functionality
- May require workarounds for edge cases

#### Strategy 2: Create New Modules

**When**: You need functionality not provided by existing steps.

**Process**:
1. Determine the appropriate step type (Generator/Modifier/Analyzer)
2. Subclass the corresponding base class
3. Implement required abstract methods
4. Define parameters and documentation
5. Set a `code_version`
6. Use in your flows like any built-in step

**Example Use Cases**:
- Ingesting a proprietary schedule format (Generator)
- Implementing a custom scheduling algorithm (Modifier)
- Creating domain-specific visualizations (Analyzer)

**Benefits**:
- Full control over implementation
- Automatic integration with framework (caching, logging, Prefect)
- Reusable across projects

**Effort**:
- Requires understanding framework contracts
- Must implement all abstract methods
- Responsibility for testing and versioning

See Sections 6.2-6.4 for detailed guides.

#### Strategy 3: Hybrid Approach

**When**: Most workflow can use existing steps, but one or two steps need customization.

**Process**:
1. Start with existing steps for standard operations
2. Create lightweight, flow-specific custom steps for unique logic
3. Combine in a single flow

**Example**: Custom vehicle type configuration:

```python
# In your flow file
class ConfigureCustomVehicleTypes(Modifier):
    """Flow-specific modifier for custom vehicle configuration."""

    def __init__(self, code_version: str = "v1", **kwargs):
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {}

    def modify(self, session, params):
        for vt in session.query(VehicleType).all():
            vt.battery_capacity = 350  # kWh
            vt.consumption = 1.2       # kWh/km
        session.flush()

# Use alongside built-in steps
@flow
def custom_pipeline():
    steps = [
        GTFSIngester(input_files=[...]),           # Built-in
        ConfigureCustomVehicleTypes(),              # Custom
        VehicleScheduling(),                        # Built-in
        Simulation(),                               # Built-in
    ]
    run_steps(steps, context)
```

**Benefits**:
- Pragmatic: use what exists, build only what's needed
- Keeps custom code close to usage
- Faster development

**Example**: `eflips/x/flows/swu_gtfs_flow.py` (ConfigureVehicleTypes modifier)

### 6.2 Creating a New Generator

Generators create new databases from external data sources.

#### Step-by-Step Guide

**1. Subclass Generator**

```python
from pathlib import Path
from typing import Any, Dict, List, Union
from eflips.x.framework import Generator
from sqlalchemy.orm import Session

class MyDataIngester(Generator):
    pass  # We'll implement methods next
```

**2. Implement `__init__` with input_files**

```python
def __init__(
    self,
    input_files: List[Union[str, Path]],
    code_version: str = "v1",  # Increment when logic changes
    **kwargs
):
    super().__init__(input_files=input_files, code_version=code_version, **kwargs)
```

The `input_files` are automatically included in cache key computation.

**3. Implement `generate(session, params)`**

This is where you populate the database:

```python
def generate(self, session: Session, params: Dict[str, Any]) -> None:
    """
    Read input files and populate the database with transit network data.

    The framework provides:
    - session: A SQLAlchemy session connected to the new database
    - params: Configuration parameters from PipelineContext

    The database schema is already created. Your job is to populate it.
    """
    # Get step-specific parameters
    encoding = params.get(f"{self.__class__.__name__}.encoding", "utf-8")

    # Process input files
    for input_file in self.input_files:
        with open(input_file, encoding=encoding) as f:
            data = parse_my_format(f)  # Your parsing logic

        # Create ORM objects
        for route_data in data["routes"]:
            route = Route(
                name=route_data["name"],
                # ... other fields
            )
            session.add(route)

        for trip_data in data["trips"]:
            trip = Trip(
                route_id=trip_data["route_id"],
                # ... other fields
            )
            session.add(trip)

    # Flush changes (framework will commit)
    session.flush()
```

**4. Implement `document_params()`**

```python
@classmethod
def document_params(cls) -> Dict[str, str]:
    return {
        "log_level": "Global logging level (DEBUG, INFO, WARNING, ERROR)",
        f"{cls.__name__}.encoding": "File encoding for input files (default: utf-8)",
        f"{cls.__name__}.skip_validation": "Skip data validation (default: False)",
    }
```

Parameters prefixed with class name are step-specific. Others are global.

**5. Set code_version**

Increment the `code_version` default whenever you change the `generate()` logic:

```python
def __init__(self, input_files, code_version: str = "v2", **kwargs):  # v1 -> v2
    super().__init__(input_files=input_files, code_version=code_version, **kwargs)
```

This invalidates cached results when code changes.

**6. Test caching behavior**

Run your pipeline twice. The second run should use cached results:

```
INFO:prefect.task:Task run 'MyDataIngester-abc123' - Using cached result.
```

#### Complete Minimal Example

```python
from pathlib import Path
from typing import Any, Dict, List, Union
from eflips.x.framework import Generator
from eflips.model import Route, Trip
from sqlalchemy.orm import Session

class SimpleCSVIngester(Generator):
    """Ingest transit data from CSV files."""

    def __init__(
        self,
        input_files: List[Union[str, Path]],
        code_version: str = "v1",
        **kwargs
    ):
        super().__init__(input_files=input_files, code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.delimiter": "CSV delimiter character (default: ',')",
        }

    def generate(self, session: Session, params: Dict[str, Any]) -> None:
        import csv
        delimiter = params.get(f"{self.__class__.__name__}.delimiter", ",")

        for input_file in self.input_files:
            with open(input_file, newline='') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    route = Route(name=row["route_name"])
                    session.add(route)

        session.flush()
```

### 6.3 Creating a New Modifier

Modifiers transform an existing database into a new one.

#### Step-by-Step Guide

**1. Subclass Modifier**

```python
from eflips.x.framework import Modifier
from sqlalchemy.orm import Session
from typing import Any, Dict

class MyTransformation(Modifier):
    pass  # We'll implement methods next
```

**2. Implement `__init__`**

```python
def __init__(
    self,
    code_version: str = "v1",  # Increment when logic changes
    **kwargs
):
    super().__init__(code_version=code_version, **kwargs)

    # Optional: Store configuration
    # self.my_config = kwargs.get("my_config")
```

Modifiers typically don't need `input_files` (unlike Generators), but can use `additional_files` for configuration files:

```python
def __init__(
    self,
    additional_files: Optional[List[Path]] = None,
    code_version: str = "v1",
    **kwargs
):
    super().__init__(additional_files=additional_files, code_version=code_version, **kwargs)
```

**3. Implement `modify(session, params)`**

Transform the database in place:

```python
def modify(self, session: Session, params: Dict[str, Any]) -> None:
    """
    Modify the database by transforming existing data.

    The framework provides:
    - session: A SQLAlchemy session connected to a COPY of the input database
    - params: Configuration parameters from PipelineContext

    The input database has been copied to output_db. Modify it in place.
    """
    # Get parameters
    threshold = params.get(f"{self.__class__.__name__}.threshold", 100.0)

    # Query existing data
    rotations = session.query(Rotation).all()

    # Transform it
    for rotation in rotations:
        if rotation.total_distance < threshold:
            # Remove short rotations
            session.delete(rotation)
        else:
            # Enrich remaining rotations
            rotation.is_long_distance = True

    # Flush changes (framework will commit)
    session.flush()
```

**4. Handle parameters and validation**

```python
@classmethod
def document_params(cls) -> Dict[str, str]:
    return {
        f"{cls.__name__}.threshold": "Minimum rotation distance in km (default: 100.0)",
        f"{cls.__name__}.mode": "Processing mode: 'strict' or 'lenient' (default: 'strict')",
    }

def modify(self, session: Session, params: Dict[str, Any]) -> None:
    # Get and validate parameters
    threshold = params.get(f"{self.__class__.__name__}.threshold", 100.0)
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")

    mode = params.get(f"{self.__class__.__name__}.mode", "strict")
    if mode not in ["strict", "lenient"]:
        raise ValueError(f"mode must be 'strict' or 'lenient', got {mode}")

    # Your logic here...
```

**5. Session.flush() vs. Session.commit()**

**Use `session.flush()`**: Persists changes within the transaction. The framework handles `commit()`.

**Don't use `session.commit()`**: The framework manages transaction boundaries and will commit after your `modify()` returns successfully.

```python
def modify(self, session: Session, params: Dict[str, Any]) -> None:
    # Make changes
    rotation.total_distance = compute_distance(rotation)

    # Flush to database (within transaction)
    session.flush()

    # DON'T commit - framework handles this
    # session.commit()  # ❌ Don't do this
```

**6. Error handling**

Let exceptions propagate. The framework will:
- Attempt to commit anyway (for debugging)
- Move the failed database to `.{timestamp}.failed`
- Re-raise the exception

```python
def modify(self, session: Session, params: Dict[str, Any]) -> None:
    result = complex_computation()

    if result is None:
        # Let exception propagate
        raise ValueError("Computation failed")

    # No need to catch and rollback - framework handles it
```

#### Complete Minimal Example

```python
from eflips.x.framework import Modifier
from eflips.model import Rotation
from sqlalchemy.orm import Session
from typing import Any, Dict

class RemoveShortRotations(Modifier):
    """Remove rotations below a distance threshold."""

    def __init__(self, code_version: str = "v1", **kwargs):
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.min_distance_km": "Minimum rotation distance in km (default: 50.0)",
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        min_distance = params.get(f"{self.__class__.__name__}.min_distance_km", 50.0)

        rotations = session.query(Rotation).all()
        removed_count = 0

        for rotation in rotations:
            if rotation.total_distance < min_distance:
                session.delete(rotation)
                removed_count += 1

        self.logger.info(f"Removed {removed_count} rotations below {min_distance} km")
        session.flush()
```

### 6.4 Creating a New Analyzer

Analyzers extract insights from databases without modifying pipeline state.

#### Step-by-Step Guide

**1. Subclass Analyzer**

```python
from eflips.x.framework import Analyzer
from sqlalchemy.orm import Session
from typing import Any, Dict

class MyAnalysis(Analyzer):
    pass  # We'll implement methods next
```

**2. Implement `__init__`**

```python
def __init__(self, code_version: str = "v1", **kwargs):
    super().__init__(code_version=code_version, **kwargs)
```

**3. Implement `analyze(session, params)` returning results**

Extract data and return structured results:

```python
def analyze(self, session: Session, params: Dict[str, Any]) -> Any:
    """
    Analyze the database and return results.

    The framework provides:
    - session: A SQLAlchemy session connected to a READ-ONLY copy
    - params: Configuration parameters

    Return any structure you want - it's passed to visualize().
    """
    # Get parameters
    vehicle_id = params.get(f"{self.__class__.__name__}.vehicle_id")
    if vehicle_id is None:
        raise ValueError(f"{self.__class__.__name__}.vehicle_id is required")

    # Query data
    vehicle = session.query(Vehicle).filter_by(id=vehicle_id).one()
    events = session.query(Event).filter_by(vehicle_id=vehicle_id).order_by(Event.time_start).all()

    # Compute metrics
    total_distance = sum(e.distance for e in events if e.distance)
    total_energy = sum(e.energy_consumed for e in events if e.energy_consumed)

    # Return structured results
    return {
        "vehicle": vehicle,
        "events": events,
        "total_distance_km": total_distance / 1000,
        "total_energy_kwh": total_energy,
        "efficiency_kwh_per_km": total_energy / total_distance if total_distance > 0 else 0,
    }
```

**4. Implement static `visualize(results)` method**

Create a visualization from the results:

```python
import plotly.graph_objects as go

@staticmethod
def visualize(results: Dict[str, Any]) -> go.Figure:
    """
    Create a visualization from analysis results.

    This is a static method - it doesn't have access to instance state.
    All information must come from the results dictionary.
    """
    events = results["events"]

    # Extract time series data
    times = [e.time_start for e in events]
    soc = [e.soc_start for e in events if e.soc_start is not None]

    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=soc,
        mode='lines',
        name='State of Charge'
    ))

    fig.update_layout(
        title=f"Vehicle {results['vehicle'].id} - State of Charge",
        xaxis_title="Time",
        yaxis_title="SoC (%)",
    )

    return fig
```

**5. Choose visualization library**

eflips-x analyzers commonly use:

- **Plotly** (`plotly.graph_objects.Figure`): Interactive plots, good for time series and scatter plots
- **Folium** (`folium.Map`): Geographic visualizations
- **Matplotlib** (`matplotlib.animation.FuncAnimation`): Animations and static plots
- **Cytoscape** (`dash_cytoscape.Cytoscape`): Network graphs

**6. Return appropriate format**

The `visualize()` method should return an object that can be saved:

```python
# Plotly Figure
vis.write_html(output_file)

# Folium Map
vis.save(str(output_file))

# Matplotlib Animation
vis.save(output_file, writer=writer)
```

See `eflips/x/flows/example.py` (lines 54-86) for the complete `save_visualization()` helper.

#### Complete Minimal Example

```python
from eflips.x.framework import Analyzer
from eflips.model import Vehicle, Event
from sqlalchemy.orm import Session
from typing import Any, Dict
import plotly.graph_objects as go

class VehicleDistanceAnalyzer(Analyzer):
    """Analyze total distance traveled per vehicle."""

    def __init__(self, code_version: str = "v1", **kwargs):
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {}  # No parameters needed

    def analyze(self, session: Session, params: Dict[str, Any]) -> Dict[str, Any]:
        vehicles = session.query(Vehicle).all()

        results = []
        for vehicle in vehicles:
            events = session.query(Event).filter_by(vehicle_id=vehicle.id).all()
            total_distance = sum(e.distance for e in events if e.distance)

            results.append({
                "vehicle_id": vehicle.id,
                "total_distance_km": total_distance / 1000,
            })

        return {"vehicles": results}

    @staticmethod
    def visualize(results: Dict[str, Any]) -> go.Figure:
        vehicles = results["vehicles"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[v["vehicle_id"] for v in vehicles],
            y=[v["total_distance_km"] for v in vehicles],
            name="Total Distance"
        ))

        fig.update_layout(
            title="Total Distance by Vehicle",
            xaxis_title="Vehicle ID",
            yaxis_title="Distance (km)",
        )

        return fig
```

### 6.5 Composing Complex Flows

Complex workflows often require advanced composition patterns.

#### When to Split into Phases

**Indicators**:
- Later steps need data computed by earlier steps
- Conditional step inclusion based on intermediate results
- Dynamic parameter configuration

**Example**:

```python
@flow
def adaptive_pipeline():
    context = PipelineContext(work_dir=work_dir, params=params)

    # Phase 1: Initial setup
    run_steps([GTFSIngester(...), RemoveUnusedData()], context)

    # Query to make decisions
    with context.get_session() as session:
        route_count = session.query(Route).count()

    # Phase 2: Conditional logic
    if route_count > 100:
        # Large network: use simplified scheduling
        params["VehicleScheduling.max_iterations"] = 1000
    else:
        # Small network: use precise scheduling
        params["VehicleScheduling.max_iterations"] = 10000

    run_steps([VehicleScheduling(), Simulation()], context)
```

#### Dynamic Task Generation

Generate tasks based on database contents:

```python
from prefect import flow, task
from prefect.task_runners import ProcessPoolTaskRunner

@task
def analyze_rotation(rotation_id, context, output_dir):
    params = context.params.copy()
    params["SingleRotationInfoAnalyzer.rotation_id"] = rotation_id

    analyzer = SingleRotationInfoAnalyzer()
    result = analyzer.execute(context=context)
    vis = analyzer.visualize(result)
    vis.write_html(output_dir / f"rotation_{rotation_id}.html")

@flow(task_runner=ProcessPoolTaskRunner())
def analyze_all_rotations(context, output_dir):
    # Query database for rotation IDs
    with context.get_session() as session:
        rotation_ids = [r.id for r in session.query(Rotation).all()]

    # Generate task for each rotation
    futures = []
    for rotation_id in rotation_ids:
        future = analyze_rotation.submit(rotation_id, context, output_dir)
        futures.append(future)

    # Wait for all
    from prefect import wait
    wait(futures)
```

This pattern is used in `eflips/x/flows/analysis_flow.py` for per-rotation, per-vehicle, and per-depot analyses.

#### Conditional Execution

Execute steps conditionally:

```python
@flow
def conditional_pipeline(include_simulation: bool = True):
    steps = [
        GTFSIngester(...),
        VehicleScheduling(),
    ]

    if include_simulation:
        steps.append(Simulation())

    context = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps, context)

    # Conditional analysis
    if include_simulation:
        # Post-simulation analyzers
        run_post_sim_analysis(context)
    else:
        # Pre-simulation analyzers only
        run_pre_sim_analysis(context)
```

#### Platform-Specific Considerations

Different task runners have different trade-offs:

```python
import platform
from prefect.task_runners import ProcessPoolTaskRunner, ThreadPoolTaskRunner

# ProcessPool for CPU-bound tasks on Unix
# ThreadPool for macOS (due to multiprocessing issues with SQLite)
if platform.system() == "Darwin":
    TASK_RUNNER = ThreadPoolTaskRunner()
else:
    TASK_RUNNER = ProcessPoolTaskRunner()

@flow(task_runner=TASK_RUNNER)
def platform_aware_analysis(context):
    # Analysis logic...
    pass
```

This pattern is used in `eflips/x/flows/analysis_flow.py`.

---

## 7. Best Practices & Design Principles

This section distills lessons learned from building and using eflips-x.

### 7.1 Versioning

#### Semantic Versioning for code_version

**Pattern**: Use simple semantic versioning like `"v1"`, `"v2"`, `"v3"`.

**When to increment**:
- Logic changes: Modified algorithms, different computations
- Bug fixes that affect outputs
- Changes to database schema or structure
- Parameter semantics changes

**When NOT to increment**:
- Code refactoring without behavior changes
- Performance optimizations that don't affect results
- Logging or documentation changes
- Cosmetic changes

**Example**:

```python
# Version 1: Simple distance calculation
class ComputeDistances(Modifier):
    def __init__(self, code_version: str = "v1", **kwargs):
        super().__init__(code_version=code_version, **kwargs)

    def modify(self, session, params):
        for trip in session.query(Trip).all():
            trip.distance = haversine_distance(trip.start_station, trip.end_station)
        session.flush()

# Version 2: Use actual routing instead of straight-line distance
# code_version incremented to invalidate cache
class ComputeDistances(Modifier):
    def __init__(self, code_version: str = "v2", **kwargs):  # v1 -> v2
        super().__init__(code_version=code_version, **kwargs)

    def modify(self, session, params):
        for trip in session.query(Trip).all():
            trip.distance = get_route_distance(trip.start_station, trip.end_station)  # Changed
        session.flush()
```

#### Poetry.lock Role in Caching

The framework automatically hashes `poetry.lock` and includes it in cache keys. This means:

**Dependency updates invalidate caches**: When you update packages (e.g., `eflips.model` schema changes), caches are automatically invalidated.

**No manual intervention needed**: You don't need to increment `code_version` when only dependencies change.

**Lock your dependencies**: Always commit `poetry.lock` to version control to ensure reproducible builds.

### 7.2 Parameter Design

#### Namespacing with Class Names

**Pattern**: Prefix step-specific parameters with the class name:

```python
params[f"{self.__class__.__name__}.threshold"] = 0.5
```

**Why**: Prevents naming collisions when multiple steps use similar parameter names.

**Example**:

```python
params = {
    "log_level": "INFO",  # Global
    "VehicleScheduling.charge_type": ChargeType.DEPOT,  # Step-specific
    "VehicleScheduling.minimum_break_time": timedelta(hours=2),
    "Simulation.charge_type": ChargeType.DEPOT,  # Different step, same param name
    "Simulation.temperature_model": "constant",
}
```

#### Global vs. Step-Specific Parameters

**Global parameters** (no prefix):
- `log_level`: Logging configuration
- `work_dir`: Working directory (usually in context, not params)
- Domain-wide settings that apply to all steps

**Step-specific parameters** (with class name prefix):
- Algorithm configuration
- Step behavior toggles
- Data filtering criteria

#### Parameter Discovery Workflow

Users can discover available parameters at runtime:

```python
from eflips.x.steps.generators import BVGXMLIngester

for param, description in BVGXMLIngester.document_params().items():
    print(f"{param}: {description}")
```

**Output**:
```
log_level: Global logging level (DEBUG, INFO, WARNING, ERROR)
BVGXMLIngester.multithreading: Enable parallel processing (default: True)
BVGXMLIngester.skip_validation: Skip input validation (default: False)
```

This makes the framework self-documenting.

#### Type Safety Considerations

Parameters are passed as `Dict[str, Any]`, which isn't type-safe. Best practices:

**Validate early**:

```python
def modify(self, session, params):
    threshold = params.get(f"{self.__class__.__name__}.threshold", 0.5)
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be numeric, got {type(threshold)}")
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
```

**Provide defaults**:

```python
threshold = params.get(f"{self.__class__.__name__}.threshold", 0.5)  # Default: 0.5
```

**Document types in docstrings**:

```python
@classmethod
def document_params(cls) -> Dict[str, str]:
    return {
        f"{cls.__name__}.threshold": "Numeric threshold value (float, default: 0.5)",
    }
```

### 7.3 Database Session Management

#### Never Create Engines/Sessions Manually

**Don't do this**:

```python
# ❌ Bad
def modify(self, session, params):
    db_url = "sqlite:///my_db.db"
    engine = create_engine(db_url)
    other_session = Session(engine)
    # ...
```

**Why**: The framework manages database connections. Manual session creation can cause:
- Connection leaks
- Transaction conflicts
- Incorrect database references

#### Use context.get_session()

For querying intermediate results in flows:

```python
@flow
def my_flow():
    run_steps(initial_steps, context)

    # ✅ Good: Use context session
    with context.get_session() as session:
        vehicles = session.query(Vehicle).all()
        print(f"Found {len(vehicles)} vehicles")

    run_steps(next_steps, context)
```

For step implementation, the session is provided:

```python
def modify(self, session, params):
    # ✅ Session already provided - just use it
    rotations = session.query(Rotation).all()
```

#### Transaction Boundaries

**Steps**: The framework manages transactions. Each step executes in a single transaction:

```
Step.execute() {
    session = Session(engine)
    try:
        self.generate/modify/analyze(session, params)
        session.commit()  # Framework commits
    except:
        session.rollback()  # Framework rolls back
        raise
}
```

**Flows**: Flows don't have automatic transactions. If you query in a flow, you're reading committed data:

```python
with context.get_session() as session:
    # Read committed data from current_db
    data = session.query(Model).all()
# Session closed automatically
```

#### Flush vs. Commit

**In steps, use `session.flush()`**:

```python
def modify(self, session, params):
    rotation.total_distance = compute_distance(rotation)
    session.flush()  # ✅ Persist changes within transaction
    # Framework will commit after modify() returns
```

**Don't use `session.commit()`**:

```python
def modify(self, session, params):
    rotation.total_distance = compute_distance(rotation)
    session.commit()  # ❌ Don't do this - framework handles commit
```

**Why**: The framework needs control over transaction boundaries to handle errors and move failed databases to `.failed` files.

### 7.4 Error Handling

#### Let Exceptions Propagate

**Don't catch and suppress**:

```python
# ❌ Bad
def modify(self, session, params):
    try:
        risky_operation()
    except Exception as e:
        self.logger.error(f"Failed: {e}")
        # Continuing without raising means corrupt data might be committed
```

**Do let exceptions propagate**:

```python
# ✅ Good
def modify(self, session, params):
    risky_operation()  # If this fails, exception propagates
    # Framework will:
    # 1. Attempt to commit (for debugging)
    # 2. Move DB to .failed
    # 3. Re-raise exception
```

**Why**: The framework needs to know when steps fail to properly handle failed databases and stop pipeline execution.

#### Framework Handles Rollback

You don't need to manage rollback:

```python
# ❌ Unnecessary
def modify(self, session, params):
    try:
        modify_data()
        session.flush()
    except Exception as e:
        session.rollback()  # Framework does this
        raise
```

```python
# ✅ Simpler
def modify(self, session, params):
    modify_data()
    session.flush()
    # Framework handles rollback on exception
```

#### Failed Database Debugging

When a step fails, inspect the `.failed` database:

```bash
$ ls output/
step_001_BVGXMLIngester.db
step_002_SetUpBvgVehicleTypes.db
step_003_VehicleScheduling.2025-12-15T10:30:45.failed  # Failed step

$ sqlite3 output/step_003_VehicleScheduling.2025-12-15T10:30:45.failed
sqlite> SELECT COUNT(*) FROM rotations;
sqlite> -- Inspect partial results to diagnose failure
```

The framework attempts to commit even on failure, so partial results are often available for debugging.

#### Logging Practices

**Use the provided logger**:

```python
def modify(self, session, params):
    self.logger.info(f"Processing {len(rotations)} rotations")
    self.logger.debug(f"Using threshold: {threshold}")
    self.logger.warning(f"Found {invalid_count} invalid rotations")
```

**Don't use print()**:

```python
# ❌ Bad
def modify(self, session, params):
    print("Processing rotations")  # Not captured by logging system
```

**Why**: The framework's logging system respects log levels, integrates with Prefect, and writes to appropriate outputs.

### 7.5 Cache Key Design

#### What to Include in Cache Keys

The base classes handle most cache key computation, but you control:

1. **code_version**: Increment when logic changes
2. **input_files**: Automatically hashed by Generator
3. **additional_files**: Automatically hashed by Modifier/Analyzer
4. **Parameters**: Automatically hashed from `context.params`

**Usually you don't override `compute_cache_key()`**. The default implementation is sufficient for most cases.

**When to override**: If you have non-standard cache invalidation needs:

```python
def compute_cache_key(self, context, output_db):
    # Get base cache key
    base_key = super().compute_cache_key(context, output_db)

    # Add custom factors
    custom_factor = f"special:{self.my_special_config}"

    # Combine and hash
    final_key = f"{base_key}:{custom_factor}"
    return hashlib.sha256(final_key.encode()).hexdigest()
```

#### When to Invalidate Caches

Caches are automatically invalidated when:
- Input files change (content hash changes)
- `code_version` increments
- `poetry.lock` changes (dependencies updated)
- Parameters change
- Input database changes (for Modifiers)

**Manual invalidation**: Delete the output database file:

```bash
rm output/step_003_VehicleScheduling.db
```

Next run will recompute this step and all subsequent steps.

#### Balancing Specificity vs. Reusability

**Too specific**: Cache never hits when it should

```python
# ❌ Including timestamp means cache never hits
key_parts.append(f"timestamp:{datetime.now().isoformat()}")
```

**Too general**: Cache hits when it shouldn't

```python
# ❌ Not including code_version means logic changes don't invalidate cache
# (Don't do this - base class includes it automatically)
```

**Right balance**: Include everything that affects output, nothing that doesn't.

The framework's default cache key computation achieves this balance for typical use cases.

### 7.6 Testing Strategies

#### Unit Testing Individual Steps

Test steps in isolation with minimal fixtures:

```python
import pytest
from pathlib import Path
from eflips.x.framework import PipelineContext
from eflips.x.steps.modifiers.my_modifier import MyModifier

def test_my_modifier(tmp_path):
    # Set up minimal database
    context = PipelineContext(work_dir=tmp_path, params={})

    # Populate with test data
    setup_test_database(context)

    # Execute step
    modifier = MyModifier()
    modifier.execute(context)

    # Verify results
    with context.get_session() as session:
        rotations = session.query(Rotation).all()
        assert len(rotations) == expected_count
        assert all(r.total_distance > 0 for r in rotations)
```

#### Integration Testing Flows

Test complete workflows:

```python
def test_complete_pipeline(tmp_path, test_input_files):
    params = {"log_level": "WARNING"}

    steps = [
        BVGXMLIngester(input_files=test_input_files),
        SetUpBvgVehicleTypes(),
        VehicleScheduling(),
    ]

    context = PipelineContext(work_dir=tmp_path, params=params)
    run_steps(steps, context)

    # Verify final state
    with context.get_session() as session:
        vehicles = session.query(Vehicle).all()
        assert len(vehicles) > 0
        assert all(v.vehicle_type_id is not None for v in vehicles)
```

#### Database Fixtures

Create reusable test databases:

```python
@pytest.fixture
def populated_database(tmp_path):
    """Fixture providing a database with sample transit data."""
    context = PipelineContext(work_dir=tmp_path, params={})

    # Create and populate database
    steps = [
        TestDataGenerator(num_routes=5, num_trips=50),
    ]
    run_steps(steps, context)

    return context
```

#### Mocking External Dependencies

Mock external services like OpenRouteService:

```python
from unittest.mock import patch

def test_routing_step(tmp_path, populated_database):
    with patch('eflips.x.steps.modifiers.routing.get_route_distance') as mock:
        mock.return_value = 15000  # meters

        modifier = ComputeRoutingDistances()
        modifier.execute(populated_database)

        # Verify mocked function was called
        assert mock.call_count > 0
```

**Why**: Avoid rate limits, network dependencies, and non-determinism in tests.

---

## 8. Common Patterns & Recipes

This section provides code snippets for frequently-used operations.

### 8.1 Querying Intermediate Results

**Pattern**: Query the database between pipeline phases:

```python
@flow
def my_flow():
    # Phase 1
    run_steps(initial_steps, context)

    # Query intermediate results
    with context.get_session() as session:
        vehicles = session.query(Vehicle).all()
        depots = session.query(Depot).all()

        print(f"Pipeline has {len(vehicles)} vehicles across {len(depots)} depots")

        # Use results to configure next phase
        if len(vehicles) > 100:
            params["Simulation.simplified_mode"] = True

    # Phase 2
    run_steps(simulation_steps, context)
```

**Use cases**:
- Logging progress
- Conditional step inclusion
- Dynamic parameter configuration
- Validation between stages

### 8.2 Per-Entity Analysis

**Pattern**: Analyze each vehicle/rotation/depot individually:

```python
from prefect import flow, task

def query_all_ids(context, model_class):
    """Helper to query all IDs for a model."""
    with context.get_session() as session:
        return [obj.id for obj in session.query(model_class).all()]

@task
def analyze_entity(analyzer_class, entity_id, param_name, context, output_dir):
    """Task to analyze a single entity."""
    context.params[f"{analyzer_class.__name__}.{param_name}"] = entity_id

    analyzer = analyzer_class()
    result = analyzer.execute(context=context)
    vis = analyzer.visualize(result)

    output_file = output_dir / f"{analyzer_class.__name__}_{param_name}_{entity_id}.html"
    vis.write_html(output_file)

@flow
def analyze_all_vehicles(context, output_dir):
    """Analyze each vehicle in parallel."""
    vehicle_ids = query_all_ids(context, Vehicle)

    futures = []
    for vehicle_id in vehicle_ids:
        future = analyze_entity.submit(
            VehicleSocAnalyzer,
            vehicle_id,
            "vehicle_id",
            context,
            output_dir
        )
        futures.append(future)

    from prefect import wait
    wait(futures)
```

**Used for**:
- Per-vehicle SoC plots
- Per-rotation info graphs
- Per-depot activity animations

**See**: `eflips/x/flows/analysis_flow.py` (lines 333-520)

### 8.3 Conditional Step Execution

**Pattern 1: Based on parameters**

```python
@flow
def conditional_flow(scenario: str):
    steps = [
        GTFSIngester(input_files=[...]),
        RemoveUnusedData(),
    ]

    if scenario == "depot_charging":
        params["VehicleScheduling.charge_type"] = ChargeType.DEPOT
        steps.append(DepotAssignment())
    elif scenario == "opportunity_charging":
        params["VehicleScheduling.charge_type"] = ChargeType.OPPORTUNITY
        steps.append(OpportunityChargerPlacement())

    steps.append(VehicleScheduling())
    steps.append(Simulation())

    context = PipelineContext(work_dir=work_dir, params=params)
    run_steps(steps, context)
```

**Pattern 2: Based on intermediate results**

```python
@flow
def adaptive_flow():
    context = PipelineContext(work_dir=work_dir, params=params)

    # Phase 1
    run_steps([GTFSIngester(...), RemoveUnusedData()], context)

    # Check characteristics
    with context.get_session() as session:
        total_trips = session.query(Trip).count()
        total_distance = session.query(func.sum(Trip.distance)).scalar()

    # Phase 2: Adapt based on data
    if total_distance > 100000:  # Long-distance network
        steps = [LongDistanceScheduling(), LongDistanceSimulation()]
    else:  # Urban network
        steps = [UrbanScheduling(), UrbanSimulation()]

    run_steps(steps, context)
```

### 8.4 Flow Reuse and Nesting

**Pattern**: Import and call sub-flows:

```python
from eflips.x.flows import run_steps, generate_all_plots

@flow(name="complete-workflow")
def complete_workflow():
    # Main pipeline
    context = PipelineContext(work_dir=work_dir, params=params)

    steps = [
        BVGXMLIngester(input_files=[...]),
        VehicleScheduling(),
        Simulation(),
    ]

    # Reuse run_steps flow
    run_steps(steps, context)

    # Reuse analysis flow
    generate_all_plots(
        context=context,
        output_dir=work_dir / "analysis",
        include_videos=False,
        pre_simulation_only=False
    )
```

**Benefits**:
- DRY: Don't repeat analysis logic
- Maintainability: Update in one place
- Consistency: Same analysis across all pipelines

**Reusable flows in eflips-x**:
- `run_steps(context, steps)`: Execute step sequence
- `generate_all_plots(context, output_dir, ...)`: Generate all standard visualizations

### 8.5 Working with Spatial Data

eflips-x uses SpatiaLite for spatial database operations.

**Querying geographic data**:

```python
from eflips.model import Station
from geoalchemy2.functions import ST_Distance

with context.get_session() as session:
    # Find stations within radius
    reference_point = (13.3777, 52.5162)  # Berlin coordinates (lon, lat)

    nearby_stations = (
        session.query(Station)
        .filter(
            ST_Distance(
                Station.geom,
                f"POINT({reference_point[0]} {reference_point[1]})"
            ) < 1000  # meters
        )
        .all()
    )
```

**Computing distances**:

```python
from geoalchemy2.functions import ST_Distance

with context.get_session() as session:
    stations = session.query(Station).all()

    for i, station1 in enumerate(stations):
        for station2 in stations[i+1:]:
            distance = session.query(
                ST_Distance(station1.geom, station2.geom)
            ).scalar()

            print(f"Distance {station1.name} -> {station2.name}: {distance:.2f}m")
```

**Important**: Ensure `SPATIALITE_LIBRARY_PATH` is set before using spatial functions (see README).

---

## 9. Integration with eFLIPS Ecosystem

eflips-x orchestrates several eFLIPS components. Understanding how they integrate helps you leverage the full ecosystem.

### eflips.model: Database Schema and ORM

**What it provides**:
- SQLAlchemy ORM models for transit networks
- Database schema definitions
- Spatial data support via GeoAlchemy2

**Key models**:
- `Route`, `Trip`, `Rotation`: Schedule structure
- `Station`: Geographic locations
- `Vehicle`, `VehicleType`: Fleet composition
- `Depot`, `Area`: Infrastructure
- `Event`: Simulation outputs
- `ChargeType`: Enum for charging strategies

**Usage in eflips-x**:

```python
from eflips.model import Route, Trip, Rotation, Vehicle, Depot, Event, ChargeType

def modify(self, session, params):
    # Query using ORM
    rotations = session.query(Rotation).filter(Rotation.allow_opportunity_charging == True).all()

    # Create new objects
    new_vehicle = Vehicle(
        vehicle_type_id=vehicle_type.id,
        name="Bus 123"
    )
    session.add(new_vehicle)
```

**Schema creation**:

```python
from eflips.model import Base, create_engine

db_url = f"sqlite:////{db_path}"
engine = create_engine(db_url)
Base.metadata.create_all(engine)  # Creates all tables
```

**See**: Generators use this to create database schema (eflips/x/framework/__init__.py:293)

### eflips.depot: Depot Layout Generation

**What it provides**:
- Automatic depot layout generation
- Charging infrastructure placement
- Spatial optimization algorithms

**Usage in eflips-x**: The `DepotGenerator` modifier uses eflips.depot to create depot layouts:

```python
from eflips.x.steps.modifiers.simulation import DepotGenerator

steps.append(DepotGenerator())  # Creates depot layout objects
```

This step queries depot coordinates and capacities, then generates:
- `Area` objects (parking/charging zones)
- Physical layouts
- Charging infrastructure assignments

### eflips.ingest: Data Ingestion Utilities

**What it provides**:
- Parsers for transit data formats (GTFS, BVG XML, etc.)
- Data validation and cleaning
- Coordinate system conversions

**Usage in eflips-x**: Generators use eflips.ingest utilities:

```python
# Inside BVGXMLIngester
from eflips.ingest import parse_bvg_xml

def generate(self, session, params):
    for input_file in self.input_files:
        data = parse_bvg_xml(input_file)
        # Populate database with parsed data
```

### eflips.eval: Evaluation and Optimization

**What it provides**:
- Vehicle scheduling algorithms
- Depot assignment optimization
- Charging strategy evaluation

**Usage in eflips-x**: Modifiers wrap eflips.eval algorithms:

```python
# VehicleScheduling modifier
from eflips.eval import schedule_vehicles

def modify(self, session, params):
    charge_type = params[f"{self.__class__.__name__}.charge_type"]
    schedules = schedule_vehicles(session, charge_type=charge_type)
    # Apply schedules to database
```

### eflips.opt: Optimization Solvers

**What it provides**:
- Generic optimization problem solvers
- Mixed-integer programming interfaces
- Heuristic algorithms

**Usage**: Backend for eflips.eval algorithms. Typically not used directly in eflips-x steps.

### eflips.tco: Total Cost of Ownership

**What it provides**:
- TCO calculations for electric bus fleets
- Cost modeling (vehicles, infrastructure, energy, maintenance)
- Economic analysis

**Usage in eflips-x**: Can be used in custom Analyzers:

```python
from eflips.tco import compute_tco

class TCOAnalyzer(Analyzer):
    def analyze(self, session, params):
        vehicles = session.query(Vehicle).all()
        depots = session.query(Depot).all()

        tco = compute_tco(vehicles, depots, energy_price=params["energy_price_kwh"])

        return {"total_cost": tco, "vehicles": len(vehicles), "depots": len(depots)}
```

### How eflips-x Ties Them Together

eflips-x provides the **orchestration glue**:

1. **eflips.model** defines the data schema
2. **eflips.ingest** populates the schema (via Generators)
3. **eflips.eval** transforms the data (via Modifiers)
4. **eflips.depot** generates infrastructure (via Modifiers)
5. **eflips.tco** analyzes economics (via Analyzers)
6. **eflips-x** orchestrates the workflow with caching, observability, and composition

**The benefit**: You can focus on domain logic (scheduling algorithms, ingestion parsers, etc.) while eflips-x handles workflow concerns (caching, database management, parallelization, monitoring).

---

## 10. Advanced Topics

### 10.1 Custom Task Runners

Prefect supports different task runners for parallelization:

**ProcessPoolTaskRunner**:
- Runs tasks in separate processes
- True parallelism (bypasses Python GIL)
- Higher overhead (process creation)
- Best for CPU-bound tasks

**ThreadPoolTaskRunner**:
- Runs tasks in threads
- Lower overhead
- Subject to Python GIL
- Best for I/O-bound tasks
- Required on macOS for SQLite compatibility

**Example**:

```python
import platform
from prefect import flow, task
from prefect.task_runners import ProcessPoolTaskRunner, ThreadPoolTaskRunner

# Choose based on platform
if platform.system() == "Darwin":  # macOS
    TASK_RUNNER = ThreadPoolTaskRunner()
else:  # Linux
    TASK_RUNNER = ProcessPoolTaskRunner(max_workers=4)

@task
def cpu_intensive_analysis(data):
    # Expensive computation
    return process(data)

@flow(task_runner=TASK_RUNNER)
def parallel_flow(datasets):
    futures = [cpu_intensive_analysis.submit(data) for data in datasets]

    from prefect import wait
    wait(futures)
```

**See**: `eflips/x/flows/analysis_flow.py` for platform-specific runner selection.

### 10.2 Prefect Artifacts

Artifacts provide observability in the Prefect UI.

**Markdown artifacts**:

```python
from prefect.artifacts import create_markdown_artifact

def execute_impl(self, context, output_db):
    # Do work...

    # Create progress artifact
    create_markdown_artifact(
        key="processing-progress",
        markdown=f"""## Processing Status

        - Rotations processed: {rotation_count}
        - Vehicles scheduled: {vehicle_count}
        - Estimated completion: 85%
        """,
        description="Real-time progress"
    )
```

The framework automatically creates completion artifacts for each step (see eflips/x/framework/__init__.py:186-190).

**Custom artifacts during long operations**:

```python
def modify(self, session, params):
    rotations = session.query(Rotation).all()

    for i, rotation in enumerate(rotations):
        process_rotation(rotation)

        # Update progress every 100 rotations
        if i % 100 == 0:
            create_markdown_artifact(
                key="rotation-processing",
                markdown=f"Processed {i}/{len(rotations)} rotations ({i/len(rotations)*100:.1f}%)",
                description="Progress update"
            )

    session.flush()
```

**View in Prefect UI**: http://localhost:4200 → Flow Runs → Artifacts

### 10.3 Working Directory Structure

eflips-x creates a predictable working directory structure:

```
work_dir/
├── step_001_BVGXMLIngester.db
├── step_002_SetUpBvgVehicleTypes.db
├── step_003_RemoveUnusedData.db
├── step_004_VehicleScheduling.db
├── step_005_Simulation.db
└── analysis/
    ├── RotationInfoAnalyzer.html
    ├── VehicleSocAnalyzer_vehicle_1.html
    ├── VehicleSocAnalyzer_vehicle_2.html
    └── ...
```

**Database naming**: `step_{count:03d}_{ClassName}.db`

**Benefits**:
- Chronological ordering
- Easy to identify which step created which DB
- Simple to find intermediate states

**Cleanup strategies**:

```python
# Keep only final database
import shutil

def cleanup_intermediate_dbs(work_dir: Path):
    """Remove all but the final database."""
    db_files = sorted(work_dir.glob("step_*.db"))

    if len(db_files) > 1:
        for db_file in db_files[:-1]:  # Keep last
            db_file.unlink()
            print(f"Removed {db_file.name}")

# After pipeline completes
cleanup_intermediate_dbs(context.work_dir)
```

**Selective cleanup**:

```python
def cleanup_before_step(work_dir: Path, step_name: str):
    """Remove databases before a specific step."""
    for db_file in work_dir.glob(f"step_*_{step_name}.db"):
        db_file.unlink()
        print(f"Invalidated cache for {step_name}")

# Force re-run of VehicleScheduling and later steps
cleanup_before_step(work_dir, "VehicleScheduling")
```

**Archive strategies**:

```python
def archive_run(work_dir: Path, archive_dir: Path):
    """Archive a completed run."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    archive_path = archive_dir / f"run_{timestamp}"
    shutil.copytree(work_dir, archive_path)

    print(f"Archived to {archive_path}")
```

---

## Conclusion

eflips-x provides a powerful framework for building complex electric bus simulation pipelines. By understanding its architectural concepts, design patterns, and integration points, you can:

- **Adapt** existing pipelines to new use cases through parameter configuration
- **Extend** the framework with custom Generators, Modifiers, and Analyzers
- **Compose** complex workflows from reusable building blocks
- **Leverage** automatic caching, observability, and reproducibility

**Next steps**:
1. Review the [README](README.md) for installation and usage instructions
2. Explore [`eflips/x/flows/example.py`](eflips/x/flows/example.py) for a complete pipeline example
3. Browse available steps in `eflips/x/steps/` to see what's already built
4. Build your first custom step following the guides in Section 6
5. Join the eFLIPS community to share your work and learn from others

**Resources**:
- [eFLIPS Project Page](https://www.tu.berlin/mpm/forschung/projekte/eflips)
- [Prefect Documentation](https://docs.prefect.io/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)

Happy simulating!
