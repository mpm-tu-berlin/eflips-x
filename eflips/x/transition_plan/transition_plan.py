from typing import Dict, Any, Optional, List
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy.orm.session
from eflips.depot.api import (
    delete_depots,
    generate_depot_layout,
    simple_consumption_simulation,
    simulate_scenario,
)
from eflips.transition.parameter_registry import DieselFleetParams, ConstraintsParams
from sqlalchemy import func

from eflips.x.framework import Modifier, Analyzer, PipelineStep
from eflips.x.framework import PipelineContext
from eflips.transition.transition_planning import (
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
    VehicleType,
    EnergySource,
    Vehicle,
)
from eflips.impact.tco import init_tco_params
from eflips.impact.utils import complete_fleet
from eflips.impact.tco import calculate_tco
from eflips.impact.lca import init_lca_params, calculate_lca

from eflips.x.transition_plan.multi_stage_simulation import CreateHybridFleet


class TCOParameterConfigurator(Modifier):
    """Configures TCO parameters from scenario-specific defaults."""

    # TODO modifier this for the new eflips.tco version

    def __init__(
        self,
        code_version: str = "1",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            code_version=code_version,
            **kwargs,
        )

    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
        }

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:

        fleet_info_path = params.get(f"{self.__class__.__name__}.fleet_info_path")
        tco_params_path = params.get(f"{self.__class__.__name__}.tco_params_path")

        scenario = session.query(Scenario).one()

        complete_fleet(scenario=scenario, json_path=fleet_info_path, delete_existing_data=True)

        init_tco_params(scenario=scenario, json_path=tco_params_path)


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
        raise NotImplementedError


class LCACalculator(Analyzer):
    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)
        self.result: Optional[Dict[str, float]] = None

    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
        }

    def analyze(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Dict:
        scenario = session.query(Scenario).all()[0]
        init_lca_params(
            scenario=scenario,
            lca_json_path=params.get(f"{self.__class__.__name__}.lca_json_path"),
            overrides_json_path=params.get(f"{self.__class__.__name__}.overrides_json_path"),
        )
        result = calculate_lca(scenario=scenario)
        result.plot_by_type(
            save_path=params.get(f"{self.__class__.__name__}.plot_by_type_save_path")
        )
        result.plot_by_scope(
            save_path=params.get(f"{self.__class__.__name__}.plot_by_scope_save_path")
        )


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
            f"{cls.__name__}.plot_save_path": """
        Output file path for the internal TransitionPlannerModel.plot_results()
        figure. Parent directories are created if missing. If None, the model
        falls back to its own default filename in the current working
        directory.

        Default: None
                    """.strip(),
        }

    def analyze(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Dict:
        cn = self.__class__.__name__

        scenario = session.query(Scenario).one()

        # --- Record electric vehicle count per VT before any mutation ---
        # This is the denominator for diesel_to_ebus_ratio later.
        electric_count_by_type: Dict[int, int] = dict(
            session.query(Vehicle.vehicle_type_id, func.count(Vehicle.id))
            .join(VehicleType, VehicleType.id == Vehicle.vehicle_type_id)
            .filter(
                Vehicle.scenario_id == scenario.id,
                VehicleType.energy_source == EnergySource.BATTERY_ELECTRIC,
            )
            .group_by(Vehicle.vehicle_type_id)
            .all()
        )

        # --- Diesel simulation inside a savepoint that we roll back ---
        # Everything mutated below is reverted before we build the ParameterRegistry,
        # so the transition planner sees the unchanged (electric) scenario state and
        # the DB file of this Analyzer is not touched at the end.
        savepoint = session.begin_nested()

        # Clear prior simulation state so the diesel fleet can be re-simulated.
        session.query(Rotation).filter(Rotation.scenario_id == scenario.id).update(
            {"vehicle_id": None}
        )
        session.query(Event).filter(Event.scenario_id == scenario.id).delete()
        session.query(Vehicle).filter(Vehicle.scenario_id == scenario.id).delete()
        # delete_depots is called internally by generate_depot_layout; run here for clarity.
        delete_depots(scenario, session)

        # Reassign rotations to their diesel counterpart (name_short substring match,
        # mirroring ParameterRegistry._fetch_diesel_vehicle_types).
        all_vts = session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id).all()
        electric_vts = [vt for vt in all_vts if vt.energy_source == EnergySource.BATTERY_ELECTRIC]
        diesel_vts = [vt for vt in all_vts if vt.energy_source == EnergySource.DIESEL]
        electric_to_diesel: Dict[int, VehicleType] = {}
        for ev in electric_vts:
            if not ev.name_short:
                continue
            candidates = [
                dv for dv in diesel_vts if dv.name_short and ev.name_short in dv.name_short
            ]
            if candidates:
                electric_to_diesel[ev.id] = candidates[0]

        rotations = session.query(Rotation).filter(Rotation.scenario_id == scenario.id).all()
        for rot in rotations:
            diesel_vt = electric_to_diesel.get(rot.vehicle_type_id)
            if diesel_vt is not None:
                rot.vehicle_type = diesel_vt
        session.flush()

        # Simulate the diesel fleet to produce Vehicle rows we can count.
        generate_depot_layout(scenario=scenario, charging_power=300, delete_existing_depot=True)
        simple_consumption_simulation(scenario=scenario, initialize_vehicles=True)
        simulate_scenario(scenario, ignore_unstable_simulation=True)
        simple_consumption_simulation(scenario=scenario, initialize_vehicles=False)

        # Count diesel vehicles per diesel VT, map back to electric VT id.
        diesel_counts_by_vt_id: Dict[int, int] = dict(
            session.query(Vehicle.vehicle_type_id, func.count(Vehicle.id))
            .join(VehicleType, VehicleType.id == Vehicle.vehicle_type_id)
            .filter(
                Vehicle.scenario_id == scenario.id,
                VehicleType.energy_source == EnergySource.DIESEL,
            )
            .group_by(Vehicle.vehicle_type_id)
            .all()
        )
        diesel_id_to_electric_id = {dv.id: ev_id for ev_id, dv in electric_to_diesel.items()}
        bus_count_by_type: Dict[int, int] = {
            diesel_id_to_electric_id[dv_id]: count
            for dv_id, count in diesel_counts_by_vt_id.items()
            if dv_id in diesel_id_to_electric_id
        }
        diesel_to_ebus_ratio: Dict[int, float] = {
            vt_id: bus_count_by_type[vt_id] / electric_count_by_type[vt_id]
            for vt_id in bus_count_by_type
            if electric_count_by_type.get(vt_id, 0) > 0
        }

        # Roll back to the electric state before handing control to the planner.
        savepoint.rollback()
        session.expire_all()

        scenario = session.query(Scenario).one()

        diesel_fleet_params = DieselFleetParams(
            bus_count_by_type=bus_count_by_type,
            diesel_to_ebus_ratio=diesel_to_ebus_ratio,
        )

        constraints = params.get(f"{cn}.constraint_params")
        transition_planner_parameters = ParameterRegistry(
            session=session,
            scenario=scenario,
            constraints_params=constraints,
            diesel_fleet_params=diesel_fleet_params,
        )

        set_variable_registry = SetVariableRegistry(transition_planner_parameters)
        constraint_registry = ConstraintRegistry(transition_planner_parameters)
        expression_registry = ExpressionRegistry(transition_planner_parameters)

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

        results = model.get_results(save_results=True)

        plot_save_path = params.get(f"{cn}.plot_save_path")
        if plot_save_path is not None:
            plot_save_path = Path(plot_save_path)
            plot_save_path.parent.mkdir(parents=True, exist_ok=True)
            model.plot_results(*results, save_path=str(plot_save_path))
        else:
            model.plot_results(*results)

        electrified_blocks: Dict[int, List[int]] = {}
        unelectrified_blocks: Dict[int, List[int]] = {}
        accumulated_electrified_blocks: set[int] = set()

        for year in range(1, constraints.project_duration + 1):
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
            "unelectrified_blocks": unelectrified_blocks,
        }
        return self.result


def run_transition_planner(
    context: PipelineContext,
) -> tuple[Path, Optional[Dict[str, Any]]]:

    # TODO fix all steps in this workflow to align with the single scenario database concept.
    """Prepare the database for transition planning.

    Runs :class:`CreateHybridFleet` to create a diesel counterpart for every
    electric vehicle type in the scenario, then :class:`TCOParameterConfigurator`
    to populate the TCO parameters from a single JSON file covering both electric
    and diesel vehicle types.

    ``TransitionPlanner`` itself is intentionally not wired in here yet; it will
    be reintroduced once adapted to the new single-scenario ``ParameterRegistry``.

    Args:
        context: Pipeline context with:
            - ``scenario_id``: ID of the scenario to operate on.
            - ``TCOParameterConfigurator.tco_params_path``: Path to TCO JSON.

        The following ``TransitionPlanner.*`` params are passed through unchanged
        and reserved for when the planner step is wired back in:
            - ``TransitionPlanner.constraint_params``: Constraint parameters.
            - ``TransitionPlanner.name``: Model instance name.
            - ``TransitionPlanner.sets``: Registered sets.
            - ``TransitionPlanner.variables``: Registered variables.
            - ``TransitionPlanner.constraints``: Registered constraints.
            - ``TransitionPlanner.expressions``: Registered expressions.
            - ``TransitionPlanner.objective_components``: Objective components.

    Returns:
        A tuple of ``(output_db_path, None)``. The ``None`` is a placeholder for
        the transition-planner result that used to be returned here.
    """

    create_hybrid_fleet = CreateHybridFleet()
    tco_parameter_config = TCOParameterConfigurator()
    transition_planner = TransitionPlanner()

    steps: List[PipelineStep] = [create_hybrid_fleet, tco_parameter_config, transition_planner]
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
