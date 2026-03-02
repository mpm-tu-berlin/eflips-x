from typing import Dict, Any, Optional, List
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy.orm.session

from eflips.x.framework import Modifier, Analyzer, PipelineStep
from eflips.x.framework import PipelineContext
from eflips.opt.transition_planning.transition_planning import (
    ParameterRegistry,
    ConstraintRegistry,
    ExpressionRegistry,
    TransitionPlannerModel,
    SetVariableRegistry,
)
from eflips.model import (
    Scenario,
    Rotation,
    Event,
    EventType,
    Trip,
    Depot,
)
from eflips.tco import init_tco_parameters, get_params_from_file
from eflips.tco import TCOCalculator as TCOCalculatorBase


class TCOParameterConfigurator(Modifier):
    """Configures TCO parameters from scenario-specific defaults."""

    def __init__(
        self,
        tco_params_path: Path,
        scenario_name: str = "berlin",
        code_version: str = "1",
        **kwargs: Any,
    ) -> None:
        super().__init__(code_version=code_version, tco_params_path=tco_params_path, **kwargs)
        self.scenario_name = scenario_name
        self.tco_params_path = tco_params_path

    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
        }

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:

        params = get_params_from_file(self.tco_params_path)

        scenario = session.query(Scenario).all()[0]

        init_tco_parameters(
            scenario=scenario,
            scenario_params=params.SCENARIO_TCO,
            vehicle_type_params=params.VEHICLE_TYPES,
            battery_type_params=params.BATTERY_TYPES,
            charging_point_type_params=params.CHARGING_POINT_TYPES,
            charging_infra_params=params.CHARGING_INFRASTRUCTURE,
        )


class PlaygroundAnalyzer(Analyzer):

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)
        self.result: Optional[Dict[str, float]] = None

    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
        }

    def analyze(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Any:

        charging_stations = (
            session.query(Event.station_id)
            .filter(Event.event_type == EventType.CHARGING_OPPORTUNITY)
            .distinct()
            .all()
        )
        charging_stations = [row[0] for row in charging_stations]

        station_depot_mapping = {}
        all_trips = session.query(Trip).all()
        for trip in all_trips:
            departure_station_id = trip.route.departure_station_id
            if departure_station_id in charging_stations:
                depot_station = trip.rotation.trips[0].route.departure_station
                if departure_station_id not in station_depot_mapping:

                    station_depot_mapping[trip.route.departure_station_id] = [depot_station.id]
                else:
                    station_depot_mapping[trip.route.departure_station_id].append(depot_station.id)

        for station_id, depot_ids in station_depot_mapping.items():
            station_depot_mapping[station_id] = list(set(depot_ids))  # Remove duplicates
        return station_depot_mapping


class TCOCalculator(Analyzer):

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)
        self.result: Optional[Dict[str, float]] = None

    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
        }

    def analyze(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Dict:
        scenario = session.query(Scenario).all()[0]
        tco_calculator = TCOCalculatorBase(scenario)
        tco_calculator.calculate()
        tco_calculator.visualize()  # TODO make it align with the framework logging
        self.result = tco_calculator.tco_by_type
        return self.result


class TransitionPlanner(Analyzer):

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)
        self.result: Optional[Dict[str, Any]] = None

    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
            f"{cls.__name__}.name": """
            Name of the transition planner model instance.

            Default: None
                        """.strip(),
            f"{cls.__name__}.sets": """
        Registered sets to be used in the transition planner model.
        Default: None
                    """.strip(),
            f"{cls.__name__}.variables": """
        Registered variables to be used in the transition planner model.
        Default: None
                    """.strip(),
            f"{cls.__name__}.constraints": """
        Registered constraints to be used in the transition planner model.
        Default: None
                    """.strip(),
            f"{cls.__name__}.expressions": """
        Registered expressions to be used in the transition planner model.
        Default: None
                    """.strip(),
            f"{cls.__name__}.objective_components": """
                Components of objective function to be optimized in the transition planner model. Must be included in expressions
                Default: None
                            """.strip(),
        }

    def analyze(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Dict:

        scenario = session.query(Scenario).all()[0]
        transition_planner_parameters = ParameterRegistry(session, scenario)

        set_variable_registry = SetVariableRegistry(transition_planner_parameters)
        constraint_registry = ConstraintRegistry(transition_planner_parameters)
        expression_registry = ExpressionRegistry(transition_planner_parameters)

        cn = self.__class__.__name__
        model = TransitionPlannerModel(
            name=params.get(f"{cn}.name", "TransitionPlannerModel"),
            params=transition_planner_parameters,
            set_variable_registry=set_variable_registry,
            constraint_registry=constraint_registry,
            expression_registry=expression_registry,
            sets=params.get(f"{cn}.sets", []),
            variables=params.get(f"{cn}.variables", []),
            constraints=params.get(f"{cn}.constraints", []),
            expressions=params.get(f"{cn}.expressions", []),
            objective_components=params.get(f"{cn}.objective_components", []),
        )
        model.solve()
        (
            yearly_vehicle_assignment,
            yearly_cost_breakdown,
            station_built_year,
            depot_charger_count_year,
        ) = model.get_results()  # TODO make it align with the framework logging

        depot_name_map = {
            depot.station_id: depot.name
            for depot in session.query(Depot).filter(Depot.scenario_id == scenario.id).all()
        }
        depot_charger_count_year["depot_name"] = depot_charger_count_year["depot_id"].map(
            depot_name_map
        )

        electrified_blocks = {}
        unelectrified_blocks = {}
        accumulated_electrified_blocks = set()

        for year in range(1, scenario.tco_parameters["project_duration"] + 1):
            electrified_blocks[year] = model.get_electrified_blocks(year=year)
            accumulated_electrified_blocks.update(electrified_blocks[year])

            if accumulated_electrified_blocks:
                unelectrified_query = session.query(Rotation.id).filter(
                    Rotation.scenario_id == scenario.id,
                    Rotation.id.notin_(accumulated_electrified_blocks),
                )
            else:
                unelectrified_query = session.query(Rotation.id).filter(
                    Rotation.scenario_id == scenario.id,
                )
            unelectrified_blocks[year] = [row[0] for row in unelectrified_query.all()]

        self.result = {
            # "electrified_blocks": electrified_blocks,
            "unelectrified_blocks": unelectrified_blocks,
            "yearly_vehicle_assignment": yearly_vehicle_assignment,
            "yearly_cost_breakdown": yearly_cost_breakdown,
            "station_built_year": station_built_year,
            "depot_charger_count_year": depot_charger_count_year,
        }
        return self.result


def run_transition_planner(
    context: PipelineContext,
) -> tuple[Path, Optional[Dict[str, Any]]]:
    """Run the transition planner model.

    Args:
        context: Pipeline context containing all parameters under namespaced keys:
            - ``TCOParameterConfigurator.tco_params_path``: Path to TCO params file.
            - ``TCOParameterConfigurator.scenario_name``: Scenario name (e.g. "berlin").
            - ``TransitionPlanner.name``: Model instance name.
            - ``TransitionPlanner.sets``: Registered sets.
            - ``TransitionPlanner.variables``: Registered variables.
            - ``TransitionPlanner.constraints``: Registered constraints.
            - ``TransitionPlanner.expressions``: Registered expressions.
            - ``TransitionPlanner.objective_components``: Objective components.

    Returns:
        A tuple of (output_db_path, transition_planner_result).
        transition_planner_result contains 'electrified_vehicles' and
        'electrified_blocks' dicts keyed by year.
    """

    tco_params_path = context.params["TCOParameterConfigurator.tco_params_path"]
    scenario_name = context.params.get("TCOParameterConfigurator.scenario_name", "berlin")

    tco_parameter_config = TCOParameterConfigurator(
        scenario_name=scenario_name, tco_params_path=tco_params_path
    )
    transition_planner = TransitionPlanner()
    steps: List[PipelineStep] = [tco_parameter_config, transition_planner]
    from eflips.x.flows import run_steps

    run_steps(context=context, steps=steps)
    return context.current_db, transition_planner.result


def generate_transition_plan_plots(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate and save plots from the results of run_transition_planner.

    Produces four plots:
    - yearly_electrification.png: cumulative and newly electrified vehicles per year
    - yearly_cost_breakdown.png: stacked bar of cost components per year
    - station_build_timeline.png: number of stations built per year
    - depot_charger_count.png: charger count per depot over time

    Args:
        results: The result dict returned by run_transition_planner (second element of tuple).
        output_dir: Directory to save the plots into.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yearly_vehicle_assignment: pd.Series = results["yearly_vehicle_assignment"]
    yearly_cost_breakdown: pd.DataFrame = results["yearly_cost_breakdown"]
    station_built_year: pd.Series = results["station_built_year"]
    depot_charger_count_year: pd.DataFrame = results["depot_charger_count_year"]

    # save the data for debugging
    yearly_vehicle_assignment.to_csv(output_dir / "yearly_vehicle_assignment.csv")
    yearly_cost_breakdown.to_csv(output_dir / "yearly_cost_breakdown.csv")
    station_built_year.to_csv(output_dir / "station_built_year.csv")
    depot_charger_count_year.to_csv(output_dir / "depot_charger_count_year.csv")

    # --- 1. Yearly & cumulative vehicle electrification ---
    fig, ax = plt.subplots(figsize=(10, 5))
    years = yearly_vehicle_assignment.index
    ax.bar(years, yearly_vehicle_assignment.values, label="Newly electrified", alpha=0.7)
    ax.plot(
        years,
        yearly_vehicle_assignment.cumsum().values,
        marker="o",
        color="tab:red",
        label="Cumulative electrified",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of vehicles")
    ax.set_title("Vehicle electrification over time")
    ax.legend()
    ax.set_xticks(years)
    fig.tight_layout()
    fig.savefig(output_dir / "yearly_electrification.png", dpi=150)
    plt.close(fig)

    # --- 2. Yearly cost breakdown (stacked bar) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_cost_breakdown.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cost (EUR)")
    ax.set_title("Yearly cost breakdown")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "yearly_cost_breakdown.png", dpi=150)
    plt.close(fig)

    # --- 3. Station build timeline ---
    if not station_built_year.empty:
        builds_per_year = station_built_year.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(builds_per_year.index, builds_per_year.values)
        ax.set_xlabel("Year")
        ax.set_ylabel("Stations built")
        ax.set_title("Charging stations built per year")
        ax.set_xticks(builds_per_year.index)
        fig.tight_layout()
        fig.savefig(output_dir / "station_build_timeline.png", dpi=150)
        plt.close(fig)

    # --- 4. Depot charger count over time ---
    if not depot_charger_count_year.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        label_col = (
            "depot_name" if "depot_name" in depot_charger_count_year.columns else "depot_id"
        )
        for label, group in depot_charger_count_year.groupby(label_col):
            group = group.sort_values("year")
            ax.step(group["year"], group["charger_count"], where="post", label=str(label))
        ax.set_xlabel("Year")
        ax.set_ylabel("Charger count")
        ax.set_title("Depot charger count over time")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / "depot_charger_count.png", dpi=150)
        plt.close(fig)


def run_tco_calculation(
    workdir: Path,
    input_db: Path,
    tco_params_path: Path,
    scenario_name: str = "berlin",
    log_level: str = "INFO",
) -> tuple[Path, Optional[Dict[str, float]]]:
    """Run the TCO analysis.

    Args:
        workdir: Working directory for pipeline outputs.
        input_db: Path to input database.
        scenario_name: Name of scenario defaults to use (e.g., "berlin").
        log_level: Logging level.

    Returns:
        A tuple of (output_db_path, tco_result).
        tco_result is a dict with TCO values categorized by type.
    """

    context_params: Dict[str, Any] = {
        "log_level": log_level,
    }
    context = PipelineContext(
        work_dir=workdir,
        current_db=input_db,
        params=context_params,
    )

    tco_parameter_config = TCOParameterConfigurator(
        scenario_name=scenario_name,
        tco_params_path=tco_params_path,
    )
    tco_calculator = TCOCalculator()
    steps: List[PipelineStep] = [tco_parameter_config, tco_calculator]
    from eflips.x.flows import run_steps

    run_steps(context=context, steps=steps)
    return context.current_db, tco_calculator.result
