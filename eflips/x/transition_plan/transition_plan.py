import math
from typing import Dict, Any, Optional, List
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy.orm.session
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
from eflips.eval.output.prepare import power_and_occupancy
from eflips.impact.tco import init_tco_params
from eflips.impact.utils import complete_fleet
from eflips.impact.lca import init_lca_params, calculate_lca

from eflips.x.steps.generators import CopyCreator
from eflips.x.steps.modifiers.general_utilities import (
    CreateDieselVehicleTypes,
    VehicleTypeBlockAssignment,
    CompleteFleet,
    TCOConfigurator,
)
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation


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


class DieselFleetAnalyzer(Analyzer):
    """Read diesel fleet counts + depot slot demand off a simulated diesel DB.

    Read-only counterpart to the diesel simulation: it assumes the database has
    already been converted to diesel and simulated (via
    :class:`~eflips.x.steps.modifiers.general_utilities.CreateDieselVehicleTypes`,
    :class:`~eflips.x.steps.modifiers.general_utilities.VehicleTypeBlockAssignment`,
    :class:`~eflips.x.steps.modifiers.simulation.DepotGenerator` and
    :class:`~eflips.x.steps.modifiers.simulation.Simulation`), and simply measures
    the result. Nothing is mutated or simulated here.

    The only output is a
    :class:`~eflips.transition.parameter_registry.DieselFleetParams` (returned by
    :meth:`analyze`):

        - ``bus_count_by_type``: ``{electric_vt_id: diesel_bus_count}``
        - ``initial_depot_capacities``: ``{depot.station_id: total_slot}``
        - ``slot_to_bus_ratio``: total depot slots / total bus footprint

    The diesel↔electric vehicle-type correspondence is recovered from the database
    via the ``"Diesel {name_short}"`` naming convention (both the electric and the
    diesel :class:`~eflips.model.VehicleType` records survive the conversion, only
    the electric :class:`~eflips.model.Vehicle` rows are replaced). The
    ``diesel_to_ebus_ratio`` is *not* populated here -- it is derived from the
    electric scenario by :class:`~eflips.transition.parameter_registry.ParameterRegistry`.
    """

    # Reference vehicle length (m) used to normalize a vehicle type's depot
    # footprint into "standard 12 m slots".
    STANDARD_SLOT_LENGTH_M = 12.0

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)
        self.result: Optional[DieselFleetParams] = None

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> DieselFleetParams:
        scenario = session.query(Scenario).one()

        # Both electric and diesel vehicle types survive the diesel conversion;
        # recover their correspondence via the "Diesel {name_short}" convention
        # (mirroring ParameterRegistry._fetch_diesel_vehicle_types).
        all_vts = session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id).all()
        electric_vts = [vt for vt in all_vts if vt.energy_source == EnergySource.BATTERY_ELECTRIC]
        diesel_vts = [vt for vt in all_vts if vt.energy_source == EnergySource.DIESEL]
        # Diesel VTs do not carry a length, so the slot size factor is derived from
        # the electric counterpart's length.
        electric_length_by_id: Dict[int, Optional[float]] = {
            ev.id: ev.length for ev in electric_vts
        }
        electric_to_diesel: Dict[int, VehicleType] = {}
        for ev in electric_vts:
            if not ev.name_short:
                continue
            candidates = [
                dv for dv in diesel_vts if dv.name_short and ev.name_short in dv.name_short
            ]
            if candidates:
                electric_to_diesel[ev.id] = candidates[0]

        # Count the simulated diesel vehicles per diesel VT, map back to electric VT id.
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

        slot_amount_all_buses = sum(
            [
                count * electric_length_by_id.get(vt_id) / self.STANDARD_SLOT_LENGTH_M
                for vt_id, count in bus_count_by_type.items()
            ]
        )

        # Required depot slots for the simulated fleet, keyed by depot station id.
        initial_depot_capacities = self._evaluate_initial_depot_capacities(
            session=session,
            scenario=scenario,
            diesel_id_to_electric_id=diesel_id_to_electric_id,
            electric_length_by_id=electric_length_by_id,
        )

        total_slot_amount = sum(initial_depot_capacities.values())

        self.result = DieselFleetParams(
            bus_count_by_type=bus_count_by_type,
            initial_depot_capacities=initial_depot_capacities,
            slot_to_bus_ratio=(
                total_slot_amount / slot_amount_all_buses if slot_amount_all_buses > 0 else 0.0
            ),
        )
        return self.result

    def _evaluate_initial_depot_capacities(
        self,
        session: sqlalchemy.orm.session.Session,
        scenario: Scenario,
        diesel_id_to_electric_id: Dict[int, int],
        electric_length_by_id: Dict[int, Optional[float]],
    ) -> Dict[int, int]:
        """Estimate the depot slots required by the simulated fleet.

        For every depot the *charging areas* (areas that recorded at least one
        ``STANDBY_DEPARTURE`` event) are grouped by vehicle type. For each group
        the peak simultaneous occupancy is read from :func:`power_and_occupancy`
        (the ``occupancy_total`` column, i.e. every vehicle physically present,
        not only those actively charging) and weighted by a size factor
        ``length / STANDARD_SLOT_LENGTH_M``. The length is taken from the electric
        counterpart of the (diesel) area vehicle type, falling back to a factor of
        ``1.0`` when no length is available.

        The depot's slot count is the rounded-up (``math.ceil``) sum of these
        weighted peaks::

            total_slot = ceil( Σ_vt  peak_occupancy_vt * (length_vt / 12) )

        Depots without any charging area are skipped. The result maps directly
        onto :attr:`DieselFleetParams.initial_depot_capacities`.

        :return: ``{depot.station_id: total_slot}``
        """
        initial_depot_capacities: Dict[int, int] = {}

        depots = session.query(Depot).filter(Depot.scenario_id == scenario.id).all()
        for depot in depots:
            area_ids = [area.id for area in depot.areas]
            if not area_ids:
                continue

            # Charging areas: those with at least one CHARGING_DEPOT event.
            standby_area_ids = {
                row[0]
                for row in session.query(Event.area_id)
                .filter(
                    Event.scenario_id == scenario.id,
                    Event.area_id.in_(area_ids),
                    Event.event_type == EventType.STANDBY_DEPARTURE,
                )
                .distinct()
                .all()
            }
            if not standby_area_ids:
                continue

            # Group the charging areas by their vehicle type.
            area_ids_by_vt: Dict[int, List[int]] = {}
            for area in depot.areas:
                if area.id in standby_area_ids:
                    area_ids_by_vt.setdefault(area.vehicle_type_id, []).append(area.id)

            weighted_sum = 0.0
            for vt_id, vt_area_ids in area_ids_by_vt.items():
                df = power_and_occupancy(area_id=vt_area_ids, session=session)
                peak_occupancy = float(df["occupancy_total"].max())

                # Diesel area VTs have no length; map back to the electric VT.
                electric_id = diesel_id_to_electric_id.get(vt_id, vt_id)
                length = electric_length_by_id.get(electric_id)
                size_factor = length / self.STANDARD_SLOT_LENGTH_M if length else 1.0
                weighted_sum += peak_occupancy * size_factor

            initial_depot_capacities[depot.station_id] = math.ceil(weighted_sum)

        return initial_depot_capacities


class TransitionPlanner(Analyzer):

    # Key under which the caller stashes the diesel fleet result -- a
    # :class:`~eflips.transition.parameter_registry.DieselFleetParams` produced by
    # the standalone diesel-simulation flow (:func:`run_diesel_simulation`) -- in
    # ``context.params`` for this step to consume.
    DIESEL_FLEET_PARAMS_PARAM = "TransitionPlanner.diesel_fleet_params"

    @staticmethod
    def _diesel_fleet_params_from(params: Dict[str, Any]) -> DieselFleetParams:
        """Fetch the :class:`DieselFleetParams` from the pipeline parameters.

        The diesel fleet is measured by the standalone diesel-simulation flow
        (:func:`run_diesel_simulation`), which runs on its own database and returns
        a :class:`~eflips.transition.parameter_registry.DieselFleetParams`. The
        caller stashes it under :attr:`DIESEL_FLEET_PARAMS_PARAM`.
        ``ParameterRegistry`` derives ``diesel_to_ebus_ratio`` from the electric
        scenario itself, so no electric denominator has to be supplied here.
        """
        obj = params.get(TransitionPlanner.DIESEL_FLEET_PARAMS_PARAM)
        if isinstance(obj, DieselFleetParams):
            return obj
        raise ValueError(
            "TransitionPlanner requires the diesel fleet composition. Set "
            f"'{TransitionPlanner.DIESEL_FLEET_PARAMS_PARAM}' in the pipeline "
            "parameters to a DieselFleetParams (e.g. the value returned by "
            "run_diesel_simulation)."
        )

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
            f"{cls.__name__}.csv_save_dir": """
        Directory path into which get_results() saves all CSV output files
        (vehicle_assignment_detailed.csv, yearly_cost_breakdown.csv, etc.).
        The directory is created if it does not exist. If None, no CSVs are
        written.

        Default: None
                    """.strip(),
        }

    def analyze(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Dict:
        cn = self.__class__.__name__

        scenario = session.query(Scenario).one()

        # Diesel fleet composition is measured by the standalone diesel-simulation
        # flow (run on its own database) and handed over through params.
        diesel_fleet_params = self._diesel_fleet_params_from(params)

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

        csv_save_dir = params.get(f"{cn}.csv_save_dir")
        results = model.get_results(
            save_dir=Path(csv_save_dir) if csv_save_dir is not None else None
        )

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
    """Run the electric transition-planning pipeline.

    Operates on the *electric* database referenced by ``context.current_db`` and
    runs, in order:

    1. :class:`~eflips.x.steps.modifiers.general_utilities.CreateDieselVehicleTypes`
       -- add a diesel :class:`~eflips.model.VehicleType` for every electric one
       (vehicle-type records only; the electric ``Vehicle`` rows are left intact,
       which the planner needs to derive ``diesel_to_ebus_ratio``).
    2. :class:`~eflips.x.steps.modifiers.general_utilities.CompleteFleet` -- rebuild
       the BatteryType / ChargingPointType topology from ``CompleteFleet.fleet_json``.
    3. :class:`~eflips.x.steps.modifiers.general_utilities.TCOConfigurator` -- write
       the TCO parameters from ``TCOConfigurator.tco_json``.
    4. :class:`TransitionPlanner` -- solve the MILP and return the per-year
       unelectrified blocks.

    The diesel fleet composition is **not** simulated here. It is supplied through
    the ``TransitionPlanner.diesel_fleet_params`` parameter -- a
    :class:`~eflips.transition.parameter_registry.DieselFleetParams` (e.g. the value
    returned by the standalone :func:`run_diesel_simulation` flow). Wiring that
    diesel flow into this one is a separate, later task; this function stays
    self-contained and only consumes the parameter.

    Args:
        context: Pipeline context with ``context.current_db`` pointing at the
            electric scenario and, in ``context.params``:

            - ``TransitionPlanner.diesel_fleet_params``: diesel fleet result
              (a ``DieselFleetParams``).
            - ``CompleteFleet.fleet_json``: fleet topology JSON path.
            - ``TCOConfigurator.tco_json``: TCO parameter JSON path.
            - the ``TransitionPlanner.*`` model configuration params
              (``constraint_params``, ``name``, ``sets``, ``variables``,
              ``constraints``, ``expressions``, ``objective_components``, ...).

    Returns:
        A tuple of ``(output_db_path, transition_planner_result)`` where the
        result is the dict returned by :class:`TransitionPlanner`:

        - ``"unelectrified_blocks"``: ``{year: [rotation_id, ...]}``
        - ``"depot_electric_slots_by_year"``: tidy DataFrame with columns
          ``operational_year``, ``ready_year``, ``depot_id``, ``electric_slots``
          -- the cumulative size-weighted (length/12) depot-charger footprint of
          the electric fleet (electrified + under construction) occupying each
          depot in each operational year.
        - ``"depot_electric_slots_by_year_map"``: the same as a nested dict
          ``{operational_year: {station_id: electric_slots}}`` for convenient
          consumption by the diesel depot-config builder.
    """
    from eflips.x.flows import run_steps

    transition_planner = TransitionPlanner()
    run_steps(
        context=context,
        steps=[
            CreateDieselVehicleTypes(),
            CompleteFleet(),
            TCOConfigurator(),
            transition_planner,
        ],
    )

    return context.current_db, transition_planner.result


def run_diesel_simulation(context: PipelineContext) -> DieselFleetParams:
    """Run a self-contained diesel simulation and report its fleet counts.

    This flow is independent of :func:`run_transition_planner` and runs on its own
    context/database. The caller builds the ``context`` and points
    ``context.current_db`` at the source (electric, already simulated) database;
    that file is only *copied* into ``context.work_dir`` (by
    :class:`~eflips.x.steps.generators.CopyCreator`), never modified, so it can be
    the same database the electric flow uses.

    Steps, in order:

    1. :class:`~eflips.x.steps.generators.CopyCreator` -- seed the diesel database
       as a copy of ``context.current_db`` inside ``context.work_dir``.
    2. :class:`~eflips.x.steps.modifiers.general_utilities.CreateDieselVehicleTypes`
       -- add a diesel :class:`~eflips.model.VehicleType` for every electric one.
    3. :class:`~eflips.x.steps.modifiers.general_utilities.VehicleTypeBlockAssignment`
       -- reassign *all* rotations to their diesel counterpart (no ``block_ids``).
    4. :class:`~eflips.x.steps.modifiers.simulation.DepotGenerator` -- build the
       depot layout.
    5. :class:`~eflips.x.steps.modifiers.simulation.Simulation` -- simulate the
       diesel fleet (produces diesel :class:`~eflips.model.Vehicle` rows).
    6. :class:`DieselFleetAnalyzer` -- read the resulting diesel bus counts and
       depot slot demand.

    Args:
        context: Pipeline context with:

            - ``context.current_db``: the source database to branch from (copied,
              never modified).
            - ``context.work_dir``: working directory for the diesel database and
              its step files. Should be distinct from the electric scenario's
              directory so the two databases never collide.
            - ``context.params``: optional simulation-step parameters, e.g.
              ``{"DepotGenerator.charging_power_kw": 90.0,
              "Simulation.ignore_unstable_simulation": True}``.

    Returns:
        The :class:`~eflips.transition.parameter_registry.DieselFleetParams` read
        from the simulated diesel database.
    """
    from eflips.x.flows import run_steps

    if context.current_db is None:
        raise ValueError(
            "run_diesel_simulation requires context.current_db to point at the "
            "source database to branch the diesel simulation from."
        )

    # Seed the diesel database as a copy of the source referenced by the context,
    # then convert and simulate it in place (each modifier chains onto the previous
    # database file).
    CopyCreator(input_files=[context.current_db]).execute(context=context)
    run_steps(
        context=context,
        steps=[
            CreateDieselVehicleTypes(),
            VehicleTypeBlockAssignment(),
            DepotGenerator(),
            Simulation(),
        ],
    )

    # Read-only measurement of the simulated diesel fleet.
    return DieselFleetAnalyzer().execute(context=context)


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
