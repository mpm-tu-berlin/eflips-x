from typing import Dict, Any, Optional, List
from pathlib import Path

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
from eflips.model import Scenario, VehicleType, Area, Station, Rotation
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

        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_rotations = session.query(Rotation).all()
        rotation_ids = []
        rotation_trip_counts = []
        rotation_mileages = []

        for rotation in all_rotations:
            rotation_ids.append(rotation.id)
            rotation_trip_counts.append(len(rotation.trips))
            rotation_mileages.append(sum(trip.route.distance for trip in rotation.trips))

        df = pd.DataFrame(
            {
                "rotation_id": rotation_ids,
                "trip_count": rotation_trip_counts,
                "mileage": rotation_mileages,
            }
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(df["trip_count"], bins=20)
        axes[0].set_xlabel("Trip Count")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Distribution of Trip Counts")

        axes[1].hist(df["mileage"], bins=20)
        axes[1].set_xlabel("Mileage (m)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Distribution of Mileages")

        fig.tight_layout()

        self.fig = fig
        return self.fig


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

        model = TransitionPlannerModel(
            name=params.get("name", "TransitionPlannerModel"),
            params=transition_planner_parameters,
            set_variable_registry=set_variable_registry,
            constraint_registry=constraint_registry,
            expression_registry=expression_registry,
            sets=params.get("sets", []),
            variables=params.get("variables", []),
            constraints=params.get("constraints", []),
            expressions=params.get("expressions", []),
            objective_components=params.get("objective_components", []),
        )
        model.solve()
        yearly_vehicle_assignment, yearly_cost_breakdown = (
            model.visualize()
        )  # TODO make it align with the framework logging

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
        }
        return self.result


def run_transition_planner(
    workdir: Path,
    input_db: Path,
    scenario_name: str = "berlin",
    log_level: str = "INFO",
    sets: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    expressions: Optional[List[str]] = None,
    objective_components: Optional[List[str]] = None,
) -> tuple[Path, Optional[Dict[str, Any]]]:
    """Run the transition planner model.

    Args:
        workdir: Working directory for pipeline outputs.
        input_db: Path to input database.
        scenario_name: Name of scenario defaults to use (e.g., "berlin").
        log_level: Logging level.
        sets: Registered sets for the transition planner model.
        variables: Registered variables for the transition planner model.
        constraints: Registered constraints for the transition planner model.
        expressions: Registered expressions for the transition planner model.
        objective_components: Components of objective function to optimize.

    Returns:
        A tuple of (output_db_path, transition_planner_result).
        transition_planner_result contains 'electrified_vehicles' and
        'electrified_blocks' dicts keyed by year.
    """

    context_params: Dict[str, Any] = {
        "log_level": log_level,
        "sets": sets if sets is not None else [],
        "variables": variables if variables is not None else [],
        "constraints": constraints if constraints is not None else [],
        "expressions": expressions if expressions is not None else [],
        "objective_components": objective_components if objective_components is not None else [],
    }
    context = PipelineContext(
        work_dir=workdir,
        current_db=input_db,
        params=context_params,
    )

    tco_parameter_config = TCOParameterConfigurator(scenario_name=scenario_name)
    transition_planner = TransitionPlanner()
    steps: List[PipelineStep] = [tco_parameter_config, transition_planner]
    from eflips.x.flows import run_steps

    run_steps(context=context, steps=steps)
    return context.current_db, transition_planner.result


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
