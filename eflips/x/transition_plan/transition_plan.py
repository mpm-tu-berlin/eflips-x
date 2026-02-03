from dataclasses import dataclass, field, asdict
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
from eflips.model import Scenario, VehicleType, Area, Station, Depot, Rotation
from eflips.tco import init_tco_parameters


@dataclass
class VehicleTypeConfig:
    name_short: str
    name: str
    useful_life: int
    procurement_cost: float
    cost_escalation: float
    average_electricity_consumption: float
    procurement_cost_diesel_equivalent: float
    cost_escalation_diesel_equivalent: float
    average_diesel_consumption: float

    def to_dict(self, vehicle_id: int) -> Dict[str, Any]:
        return {
            "id": vehicle_id,
            "name": self.name,
            "useful_life": self.useful_life,
            "procurement_cost": self.procurement_cost,
            "cost_escalation": self.cost_escalation,
            "average_electricity_consumption": self.average_electricity_consumption,
            "procurement_cost_diesel_equivalent": self.procurement_cost_diesel_equivalent,
            "cost_escalation_diesel_equivalent": self.cost_escalation_diesel_equivalent,
            "average_diesel_consumption": self.average_diesel_consumption,
        }


@dataclass
class BatteryTypeConfig:
    name: str
    procurement_cost: float
    useful_life: int
    cost_escalation: float

    def to_dict(self, battery_id: Optional[int] = None) -> Dict[str, Any]:
        return {
            "id": battery_id,
            "name": self.name,
            "procurement_cost": self.procurement_cost,
            "useful_life": self.useful_life,
            "cost_escalation": self.cost_escalation,
        }


@dataclass
class ChargingPointTypeConfig:
    type: str
    name: str
    procurement_cost: float
    useful_life: int
    cost_escalation: float

    def to_dict(self, charger_id: Optional[int] = None) -> Dict[str, Any]:
        return {
            "id": charger_id,
            "type": self.type,
            "name": self.name,
            "procurement_cost": self.procurement_cost,
            "useful_life": self.useful_life,
            "cost_escalation": self.cost_escalation,
        }


@dataclass
class ChargingInfrastructureConfig:
    type: str
    name: str
    procurement_cost: float
    useful_life: int
    cost_escalation: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioTCOConfig:
    project_duration: int = 10
    interest_rate: float = 0.05
    inflation_rate: float = 0.02
    staff_cost: float = 25.0
    fuel_cost: Dict[str, float] = field(
        default_factory=lambda: {"diesel": 1, "electricity": 0.1794}
    )
    vehicle_maint_cost: Dict[str, float] = field(
        default_factory=lambda: {"diesel": 0.5, "electricity": 0.35}
    )
    infra_maint_cost: float = 1000
    cost_escalation_rate: Dict[str, float] = field(
        default_factory=lambda: {
            "general": 0.02,
            "staff": 0.03,
            "diesel": 0.07,
            "electricity": 0.038,
        }
    )
    annual_budget_limit: float = 2.0e7
    # TODO add proper keys for depot times
    depot_time_plan: Dict[str, int] = field(
        default_factory=lambda: {
            "BF RL": 2032,
            "BF KL": 2028,
            "BF SN": 2027,
            "BF I": 2030,
            "BF S": 2030,
            "BF B": 2030,
            "BF C": 2034,
            "BF M": 2035,
            "BF L": 2030,
        }
    )
    current_year: int = 2026
    max_station_construction_per_year: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TCOParameterConfigurator(Modifier):
    # Vehicle type configurations keyed by name_short
    # Battery prices from "Wirtschaftlichkeit von Elektromobilität in gewerblichen Anwendungen", April 2015
    # Price is a prognose for 2025 in an optimistic scenario
    DEFAULT_VEHICLE_CONFIGS: Dict[str, VehicleTypeConfig] = {
        "EN": VehicleTypeConfig(
            name_short="EN",
            name="Ebusco 3.0 12 large battery",
            useful_life=14,
            procurement_cost=340000.0,
            cost_escalation=-0.02,
            average_electricity_consumption=1.48,
            procurement_cost_diesel_equivalent=275000.0,
            cost_escalation_diesel_equivalent=0.02,
            average_diesel_consumption=0.449,
        ),
        "DD": VehicleTypeConfig(
            name_short="DD",
            name="Solaris Urbino 18 large battery",
            useful_life=14,
            procurement_cost=603000.0,
            cost_escalation=-0.02,
            average_electricity_consumption=2.16,
            procurement_cost_diesel_equivalent=330000.0,
            cost_escalation_diesel_equivalent=0.02,
            average_diesel_consumption=0.589,
        ),
        "GN": VehicleTypeConfig(
            name_short="GN",
            name="Alexander Dennis Enviro500EV large battery",
            useful_life=14,
            procurement_cost=650000.0,
            cost_escalation=-0.02,
            average_electricity_consumption=2.16,
            procurement_cost_diesel_equivalent=510000.0,
            cost_escalation_diesel_equivalent=0.02,
            average_diesel_consumption=0.589,
        ),
    }

    # Battery configurations keyed by vehicle name_short (shares battery with vehicle)
    DEFAULT_BATTERY_CONFIGS: Dict[str, BatteryTypeConfig] = {
        "EN": BatteryTypeConfig(
            name="Ebusco 3.0 12 large battery",
            procurement_cost=190,
            useful_life=7,
            cost_escalation=-0.03,
        ),
        "GN": BatteryTypeConfig(
            name="Solaris Urbino 18 large battery",
            procurement_cost=190,
            useful_life=7,
            cost_escalation=-0.03,
        ),
        "DD": BatteryTypeConfig(
            name="Alexander Dennis Enviro500EV large battery",
            procurement_cost=190,
            useful_life=7,
            cost_escalation=-0.03,
        ),
    }

    # Charging point configurations
    # Prices from Jefferies and Göhlich (2020)
    DEFAULT_CHARGING_POINT_CONFIGS: Dict[str, ChargingPointTypeConfig] = {
        "depot": ChargingPointTypeConfig(
            type="depot",
            name="Depot Charging Point",
            procurement_cost=100000.0,
            useful_life=20,
            cost_escalation=0,
        ),
        "opportunity": ChargingPointTypeConfig(
            type="opportunity",
            name="Opportunity Charging Point",
            procurement_cost=250000.0,
            useful_life=20,
            cost_escalation=0,
        ),
    }

    # Charging infrastructure configurations
    # Prices from Jefferies and Göhlich (2020)
    DEFAULT_CHARGING_INFRA_CONFIGS = [
        ChargingInfrastructureConfig(
            type="depot",
            name="Depot Charging Infrastructure",
            procurement_cost=2000000.0,
            useful_life=20,
            cost_escalation=0,
        ),
        ChargingInfrastructureConfig(
            type="station",
            name="Opportunity Charging Infrastructure",
            procurement_cost=500000.0,
            useful_life=20,
            cost_escalation=0,
        ),
    ]

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
        }

    def _query_vehicle_and_battery_ids_from_name_short(
        self, session: sqlalchemy.orm.session.Session, scenario_id: int
    ) -> Dict[str, tuple[int, Optional[int]]]:
        """Query vehicle type and battery type IDs from the database.

        Returns:
            A dict mapping vehicle type name_short to a tuple of
            (vehicle_type_id, battery_type_id). battery_type_id is None
            if the vehicle type has no associated battery.
        """
        result = {}
        for name_short in self.DEFAULT_VEHICLE_CONFIGS.keys():
            row = (
                session.query(VehicleType.id, VehicleType.battery_type_id)
                .filter(
                    VehicleType.scenario_id == scenario_id,
                    VehicleType.name_short == name_short,
                )
                .one_or_none()
            )
            if row:
                vehicle_id = row[0]
                battery_id = row[1] if row[1] is not None else None
                result[name_short] = (vehicle_id, battery_id)
        return result

    def _query_charging_point_ids(
        self, session: sqlalchemy.orm.session.Session, scenario_id: int
    ) -> Dict[str, Optional[int]]:
        """Query charging point type IDs from the database."""
        depot_row = (
            session.query(Area.charging_point_type_id)
            .filter(
                Area.scenario_id == scenario_id,
                Area.charging_point_type_id.isnot(None),
            )
            .first()
        )
        station_row = (
            session.query(Station.charging_point_type_id)
            .filter(
                Station.scenario_id == scenario_id,
                Station.charging_point_type_id.isnot(None),
            )
            .first()
        )
        return {
            "depot": depot_row[0] if depot_row else None,
            "opportunity": station_row[0] if station_row else None,
        }

    def _build_tco_parameters(self, session: sqlalchemy.orm.session.Session) -> None:
        """Build and initialize TCO parameters from configuration."""
        scenario = session.query(Scenario).all()
        scenario_id = scenario[0].id

        vehicle_and_battery_ids = self._query_vehicle_and_battery_ids_from_name_short(
            session, scenario_id
        )
        charging_point_ids = self._query_charging_point_ids(session, scenario_id)

        vehicle_types = [
            config.to_dict(vehicle_and_battery_ids[name_short][0])
            for name_short, config in self.DEFAULT_VEHICLE_CONFIGS.items()
            if name_short in vehicle_and_battery_ids
        ]

        battery_types = [
            config.to_dict(vehicle_and_battery_ids.get(name_short, (None, None))[1])
            for name_short, config in self.DEFAULT_BATTERY_CONFIGS.items()
        ]

        charging_point_types = [
            config.to_dict(charging_point_ids.get(cp_type))
            for cp_type, config in self.DEFAULT_CHARGING_POINT_CONFIGS.items()
        ]

        charging_infrastructure = [
            config.to_dict() for config in self.DEFAULT_CHARGING_INFRA_CONFIGS
        ]

        scenario_tco_parameters = ScenarioTCOConfig().to_dict()

        init_tco_parameters(
            scenario=scenario[0],
            scenario_tco_parameters=scenario_tco_parameters,
            vehicle_types=vehicle_types,
            battery_types=battery_types,
            charging_point_types=charging_point_types,
            charging_infrastructure=charging_infrastructure,
        )

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        self._build_tco_parameters(session)


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
    log_level: str = "INFO",
    sets: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    expressions: Optional[List[str]] = None,
    objective_components: Optional[List[str]] = None,
) -> tuple[Path, Optional[Dict[str, Any]]]:
    """Run the transition planner model.

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

    tco_parameter_config = TCOParameterConfigurator()
    transition_planner = TransitionPlanner()
    steps: List[PipelineStep] = [tco_parameter_config, transition_planner]
    from eflips.x.flows import run_steps

    run_steps(context=context, steps=steps)
    return context.current_db, transition_planner.result
