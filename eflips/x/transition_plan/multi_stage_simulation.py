import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy.orm.session
import eflips.model
from eflips.depot.api import SmartChargingStrategy
from eflips.eval.output import prepare as eval_output_prepare
from collections import defaultdict

from eflips.model import (
    Area,
    AssocRouteStation,
    ChargingPointType,
    Depot,
    EnergySource,
    Event,
    EventType,
    Rotation,
    Route,
    Scenario,
    Station,
    StopTime,
    Trip,
    TripType,
    Vehicle,
    VehicleType,
)
from geoalchemy2.shape import from_shape, to_shape
from prefect import flow, task
from prefect.futures import wait
from prefect.task_runners import ProcessPoolTaskRunner
from sqlalchemy import func
from sqlalchemy.orm import Session

from eflips.x.flows.analysis_flow import execute_simple_analyzer
from eflips.x.framework import Modifier, PipelineContext, PipelineStep, Analyzer
from eflips.x.steps.analyzers import (
    RotationInfoAnalyzer,
    GeographicTripPlotAnalyzer,
    DepartureArrivalSocAnalyzer,
    DepotEventAnalyzer,
    SpecificEnergyConsumptionAnalyzer,
)
from eflips.x.steps.modifiers.scheduling import DepotAssignment
from eflips.x.steps.modifiers.general_utilities import CreateDieselVehicleTypes
from eflips.x.steps.modifiers.simulation import DepotGenerator, Simulation
from eflips.opt.depot_rotation_matching import DepotRotationOptimizer


logger = logging.getLogger(__name__)


class CreateHybridFleet(Modifier):
    """
    Create a diesel counterpart for every electric vehicle type in the scenario
    and optionally reassign a set of blocks to the diesel types.

    A diesel VT is created for every ``EnergySource.BATTERY_ELECTRIC`` VehicleType
    using the ``"Diesel {name_short}"`` naming convention expected by
    :class:`eflips.transition.parameter_registry.ParameterRegistry`. Diesel VTs
    get near-zero consumption and ``energy_source=EnergySource.DIESEL``.

    When ``unelectrified_block_ids`` is given, the matching rotations are
    reassigned to their diesel counterpart. Otherwise only the VT records are
    created — safe to reuse upstream of TCO/transition-planner steps.

    Idempotent: a diesel VT whose ``name_short`` already exists in the scenario
    is reused, never duplicated.
    """

    DIESEL_CONSUMPTION = 0.0001  # Near-zero consumption for diesel simulation
    DIESEL_PREFIX = "Diesel "

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            **super().document_params(),
            f"{cls.__name__}.unelectrified_block_ids": """
    Optional list of Rotation IDs to reassign to diesel vehicle types. When set,
    those rotations are treated as diesel blocks and attached to the matching
    diesel VehicleType (near-zero consumption, constant-consumption estimation).

    Default: None
                """.strip(),
            f"{cls.__name__}.scenario_id": """
    Scenario to create diesel vehicle types in. Falls back to the global
    ``scenario_id`` parameter, and finally to the scenario of the first
    unelectrified block when only ``unelectrified_block_ids`` is given.

    Default: None
                """.strip(),
        }

    def _create_diesel_vehicle_type(
        self,
        session: sqlalchemy.orm.session.Session,
        electric_type: VehicleType,
        scenario: Scenario,
    ) -> VehicleType:
        """Create a diesel version of an electric vehicle type."""
        diesel_type = VehicleType(
            scenario=scenario,
            name=f"{self.DIESEL_PREFIX}{electric_type.name}",
            name_short=f"{self.DIESEL_PREFIX}{electric_type.name_short}",
            battery_capacity=electric_type.battery_capacity,
            charging_curve=electric_type.charging_curve,
            opportunity_charging_capable=electric_type.opportunity_charging_capable,
            consumption=self.DIESEL_CONSUMPTION,
            battery_capacity_reserve=electric_type.battery_capacity_reserve,
            minimum_charging_power=electric_type.minimum_charging_power,
            charging_efficiency=electric_type.charging_efficiency,
            energy_source=EnergySource.DIESEL,
        )
        session.add(diesel_type)
        return diesel_type

    def _create_all_diesel_types(
        self,
        session: sqlalchemy.orm.session.Session,
        scenario: Scenario,
    ) -> Dict[str, VehicleType]:
        """Create a diesel counterpart for every electric VehicleType in the scenario.

        Returns a mapping keyed by the electric VT's ``name_short``. Existing
        diesel counterparts (matched by ``"Diesel {name_short}"``) are reused.
        """
        electric_types = (
            session.query(VehicleType)
            .filter(
                VehicleType.scenario_id == scenario.id,
                VehicleType.energy_source == EnergySource.BATTERY_ELECTRIC,
            )
            .all()
        )

        if len(electric_types) == 0:
            logger.warning(
                f"No electric vehicle types found in scenario {scenario.id}. "
                f"CreateHybridFleet convert them to electric for now"
            )

            all_types = (
                session.query(VehicleType).filter(VehicleType.energy_source.is_(None)).all()
            )
            for vt in all_types:
                vt.energy_source = EnergySource.BATTERY_ELECTRIC
        session.flush()
        session.expire_all()

        electric_types = (
            session.query(VehicleType)
            .filter(
                VehicleType.scenario_id == scenario.id,
                VehicleType.energy_source == EnergySource.BATTERY_ELECTRIC,
            )
            .all()
        )

        diesel_types: Dict[str, VehicleType] = {}
        for electric_type in electric_types:
            diesel_short = f"{self.DIESEL_PREFIX}{electric_type.name_short}"
            existing = (
                session.query(VehicleType)
                .filter(
                    VehicleType.scenario_id == scenario.id,
                    VehicleType.name_short == diesel_short,
                )
                .one_or_none()
            )
            if existing is not None:
                diesel_types[electric_type.name_short] = existing
            else:
                diesel_types[electric_type.name_short] = self._create_diesel_vehicle_type(
                    session, electric_type, scenario
                )
        return diesel_types

    def _assign_diesel_types(
        self,
        blocks: List[Rotation],
        diesel_types: Dict[str, VehicleType],
    ) -> None:
        """Assign diesel vehicle types to unelectrified blocks."""
        for block in blocks:
            original_short = block.vehicle_type.name_short
            if original_short not in diesel_types:
                raise ValueError(
                    f"Unknown vehicle type '{original_short}' for diesel conversion. "
                    f"Available types: {list(diesel_types.keys())}"
                )
            block.vehicle_type = diesel_types[original_short]

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """Create diesel VTs and optionally reassign unelectrified blocks to them."""
        unelectrified_block_ids = params.get(
            f"{self.__class__.__name__}.unelectrified_block_ids", None
        )

        scenario = session.query(Scenario).one()

        diesel_types = self._create_all_diesel_types(session, scenario)

        if unelectrified_block_ids:
            print(
                f"Configuring hybrid fleet with unelectrified blocks: "
                f"{unelectrified_block_ids}"
            )
            unelectrified_blocks = (
                session.query(Rotation).filter(Rotation.id.in_(unelectrified_block_ids)).all()
            )
            if unelectrified_blocks:
                self._assign_diesel_types(unelectrified_blocks, diesel_types)

        session.flush()


class DeleteDepotEvents(Modifier):
    """Delete all depot events to simulate a greenfield scenario for each stage."""

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {**super().document_params()}

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """Delete all depot events."""

        session.query(Event).delete()
        session.query(Rotation).update({Rotation.vehicle_id: None})
        session.query(Vehicle).delete()
        logger.info(f"Deleted depot events and vehicles.")

        session.flush()


class PartialDepotAssignment(DepotAssignment):
    """
    Reassign a subset of rotations to depots, running the optimizer exactly once.

    This is a single-shot variant of
    :class:`eflips.x.steps.modifiers.scheduling.DepotAssignment`, tailored for the
    year-by-year transition plan where only the rotations electrified in the current
    stage should be (re)assigned to a depot. It differs from the parent in two ways:

    1. **Partial reassignment.** Only the rotations listed in
       ``PartialDepotAssignment.rotation_ids`` take part in the optimization. All other
       rotations keep their current depot, deadhead (ferry/return) trips and depot
       electrification untouched. If ``rotation_ids`` is ``None`` (the default), every
       rotation of the scenario is reassigned, matching :class:`DepotAssignment`'s scope.
       This is delegated to :class:`~eflips.opt.depot_rotation_matching.DepotRotationOptimizer`,
       which accepts the subset via its ``reassigned_rotations`` constructor argument.

    2. **No capacity-reduction loop.** :class:`DepotAssignment` iteratively shrinks depot
       capacity to find the tightest feasible assignment. This modifier optimizes once at
       the capacities given in ``depot_config`` and writes the result. Consequently the
       ``depot_usage``, ``step_size`` and ``max_iterations`` parameters do not apply.

    The two logging helpers (:meth:`_get_depot_rotation_assignments` and
    :meth:`_log_assignments`) are inherited from :class:`DepotAssignment` unchanged.

    .. note::
       **Capacity accounting in partial mode.** The underlying optimizer builds its depot
       capacity constraint from the occupancy of the *reassigned* rotations only. Occupancy
       already consumed by untouched rotations sitting in the same depots is **not**
       subtracted from the available capacity. When reassigning into depots that already
       host other rotations (e.g. earlier-stage electric rotations), the effective
       capacities in ``depot_config`` should account for that pre-existing occupancy.
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary describing the configurable parameters:
            - PartialDepotAssignment.rotation_ids: Rotations to reassign (None = all)
            - PartialDepotAssignment.depot_config: List of depot configuration dicts
            - PartialDepotAssignment.base_url: Base URL for routing service (ORS)
        """
        return {
            f"{cls.__name__}.rotation_ids": """
            The rotations that should be reassigned to depots. Only these rotations are
            re-optimized; all other rotations keep their current depot, deadhead trips and
            depot electrification. If None, every rotation of the scenario is reassigned
            (same scope as DepotAssignment).

            Note: the optimizer's depot capacity constraint only counts the occupancy of the
            reassigned rotations. Capacity already used by untouched rotations in the same
            depots is not subtracted, so set the capacities in depot_config accordingly.

            Default: None (reassign all rotations)
            Type: Optional[List[int]] (rotation IDs); an empty list is rejected
            Example: [123, 456, 789]
            """.strip(),
            f"{cls.__name__}.depot_config": """
            A list of depot configurations. Each configuration is a dict with:
            - "depot_station": Station ID or (lon, lat) tuple
            - "capacity": Depot capacity in 12m bus equivalents
            - "vehicle_type": List of allowed vehicle type IDs
            - "name": Depot name (for new depots only)

            Required: True
            Type: List[Dict]
            Example: depots_for_bvg(db_session) from eflips.x.steps.modifiers.bvg_tools
            """.strip(),
            f"{cls.__name__}.base_url": """
            Base URL for the OpenRouteService (ORS) routing API.
            Used to calculate travel distances between depots and rotation start/end points.

            Required: True (falls back to the OPENROUTESERVICE_BASE_URL environment variable)
            Type: str
            Example: "http://mpm-v-ors.mpm.tu-berlin.de:8080/ors/"
            """.strip(),
        }

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """
        Reassign the selected rotations to depots in a single optimization pass.

        This method:
        1. Validates parameters and that exactly one scenario exists
        2. Logs the pre-optimization depot assignments
        3. Initializes the DepotRotationOptimizer restricted to ``rotation_ids``
        4. Runs the optimization once at the capacities in ``depot_config``
        5. Writes optimized assignments back to the database
        6. Logs comparison of before/after assignments

        Parameters:
        -----------
        session : sqlalchemy.orm.session.Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters:
            - PartialDepotAssignment.rotation_ids (optional): Rotations to reassign; None = all
            - PartialDepotAssignment.depot_config (required): List of depot configuration dicts
            - PartialDepotAssignment.base_url (optional): ORS routing URL; falls back to the
              OPENROUTESERVICE_BASE_URL environment variable

        Returns:
        --------
        None
            This modifier modifies the database in place by updating rotation depot assignments

        Raises:
        -------
        ValueError
            If required parameters are missing/invalid, if scenario count != 1, or if the
            optimization is infeasible at the given capacities
        """
        rotation_ids_key = f"{self.__class__.__name__}.rotation_ids"
        depot_config_key = f"{self.__class__.__name__}.depot_config"
        base_url_key = f"{self.__class__.__name__}.base_url"

        # Validate required parameters
        if depot_config_key not in params:
            raise ValueError(
                f"Required parameter '{depot_config_key}' not provided. "
                "Please specify a depot configuration."
            )
        if base_url_key in params:
            base_url = params[base_url_key]
        else:
            base_url = os.environ.get("OPENROUTESERVICE_BASE_URL")
            if not base_url:
                raise ValueError(
                    f"Required parameter '{base_url_key}' not provided. "
                    "Also, environment variable 'OPENROUTESERVICE_BASE_URL' is not set. "
                    "Please specify the base URL for the routing service."
                )
            self.logger.debug(
                "Taking base_url from environment variable OPENROUTESERVICE_BASE_URL"
            )

        rotation_ids = params.get(rotation_ids_key, None)
        depot_config = params[depot_config_key]

        # Validate rotation_ids: None (all rotations) or a non-empty list of rotation IDs.
        if rotation_ids is not None:
            if not isinstance(rotation_ids, list):
                raise ValueError(
                    f"rotation_ids must be a list of rotation IDs or None, "
                    f"got {type(rotation_ids).__name__}"
                )
            if len(rotation_ids) == 0:
                raise ValueError(
                    "rotation_ids must not be an empty list. "
                    "Use None to reassign all rotations of the scenario."
                )
            if not all(isinstance(rid, int) for rid in rotation_ids):
                raise ValueError("All elements in rotation_ids must be integers (rotation IDs)")

        # Validate parameters
        if not isinstance(depot_config, list):
            raise ValueError(
                f"depot_config must be a list of depot configurations, got {type(depot_config).__name__}"
            )
        for depot in depot_config:
            if not isinstance(depot, dict):
                raise ValueError(
                    f"Each depot configuration must be a dict, got {type(depot).__name__}"
                )
        if not isinstance(base_url, str):
            raise ValueError(f"base_url must be a string, got {type(base_url).__name__}")

        # Make sure there is just one scenario
        scenario_q = session.query(Scenario)
        if scenario_q.count() != 1:
            raise ValueError(f"Expected exactly one scenario, found {scenario_q.count()}")
        scenario = scenario_q.one()

        self.logger.info(
            f"Partial depot assignment parameters: "
            f"rotation_ids={'all' if rotation_ids is None else len(rotation_ids)}, "
            f"base_url={base_url}"
        )

        # Get the pre-optimization depot assignments for logging
        pre_optimization_assignments = self._get_depot_rotation_assignments(session)
        self._log_assignments(pre_optimization_assignments, session)
        self.logger.info("Completed logging pre-optimization depot assignments")

        self.logger.info(f"Loaded {len(depot_config)} depot configurations")

        # Set the base URL for routing service
        os.environ["BASE_URL"] = base_url

        # Initialize the optimizer restricted to the requested rotations. Passing rotation_ids
        # through unchanged lets the optimizer resolve None -> all rotations and validate the IDs.
        optimizer = DepotRotationOptimizer(session, scenario.id, reassigned_rotations=rotation_ids)

        optimizer.get_depot_from_input(depot_config)
        optimizer.data_preparation()

        # Single optimization pass. Unlike DepotAssignment, an infeasible result here is a hard
        # failure (there is no capacity-reduction loop to fall back to), so re-raise clearly.
        try:
            optimizer.optimize(time_report=True)
        except ValueError as exc:
            raise ValueError(
                "Partial depot assignment is infeasible at the given depot capacities. "
                "Consider increasing depot capacity or reducing the set of reassigned rotations."
            ) from exc
        self.logger.info("Optimization successful")

        # Write optimization results back to the database
        optimizer.write_optimization_results(delete_original_data=True)

        assert optimizer.data["result"] is not None
        assert isinstance(optimizer.data["rotation"], pd.DataFrame)
        assert isinstance(optimizer.data["result"], pd.DataFrame)
        assert optimizer.data["result"].shape[0] == optimizer.data["rotation"].shape[0]

        self.logger.info("Wrote optimization results to database")

        # Generate post-optimization depot assignments for logging
        # We need to flush and expunge the session for geom to be converted to binary
        session.flush()
        session.expunge_all()
        post_optimization_assignments = self._get_depot_rotation_assignments(session)

        self._log_assignments(post_optimization_assignments, session)
        self.logger.info("Completed logging post-optimization depot assignments")

        # Go through all depots (union of pre and post optimization keys) and list the changes
        all_stations = set(pre_optimization_assignments.keys()).union(
            post_optimization_assignments.keys()
        )
        for station_id in sorted(all_stations):
            station = session.query(Station).filter(Station.id == station_id).one()
            pre_rotations = pre_optimization_assignments.get(station.id, [])
            post_rotations = post_optimization_assignments.get(station.id, [])

            pre_count = len(pre_rotations)
            post_count = len(post_rotations)

            self.logger.info(f"Depot '{station.name}' (ID {station_id}): ")
            if pre_count == post_count and set(r.id for r in pre_rotations) == set(
                r.id for r in post_rotations
            ):
                self.logger.info(f"\tNo change in assignments ({pre_count} rotations)")
            else:
                self.logger.info(f"\tChanged from {pre_count} to {post_count} rotations")

        self.logger.info("Partial depot assignment optimization completed successfully")

        return None


class VehicleTypeStatistics(Analyzer):
    """Per-vehicle-type fleet size and peak depot occupancy (by depot).

    Returns ``{"counts": DataFrame, "occupancy": DataFrame}``:

    - ``counts``: one row per :class:`VehicleType` with ``vehicle_type_id``,
      ``vehicle_type_name``, ``energy_source``, ``vehicle_count``.
    - ``occupancy``: one row per (VehicleType, Depot) with
      ``vehicle_type_id``, ``vehicle_type_name``, ``energy_source``,
      ``depot_id``, ``depot_name``, ``station_id``, ``size_factor``,
      ``peak_occupancy``. ``peak_occupancy`` is ``max(occupancy_total)`` from
      :func:`eflips.eval.output.prepare.power_and_occupancy` over all areas
      that target this VT in that depot and have at least one
      ``EventType.STANDBY_DEPARTURE`` event. ``station_id`` is the depot's
      :class:`~eflips.model.Station` id (so occupancy can be aligned with the
      transition-planner depot ids), and ``size_factor`` is the vehicle type's
      12 m-equivalent slot weight (``length / 12``; diesel VTs inherit their
      electric counterpart's length). In the co-simulation diesel VTs *do*
      get depot areas, so both electric and diesel rows are present.
    """

    def __init__(self, code_version: str = "1", cache_enabled: bool = True) -> None:
        super().__init__(code_version=code_version, cache_enabled=cache_enabled)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.temporal_resolution": """
    Temporal resolution in seconds passed to power_and_occupancy when computing
    peak depot occupancy. Default: 60.
                """.strip(),
        }

    def analyze(
        self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]
    ) -> Dict[str, pd.DataFrame]:
        temporal_resolution = params.get(f"{self.__class__.__name__}.temporal_resolution", 60)

        vehicle_counts: Dict[int, int] = {
            vt_id: count
            for vt_id, count in session.query(Vehicle.vehicle_type_id, func.count(Vehicle.id))
            .group_by(Vehicle.vehicle_type_id)
            .all()
        }

        vehicle_types = session.query(VehicleType).all()
        depot_info: Dict[int, Any] = {
            d_id: (name, station_id)
            for d_id, name, station_id in session.query(
                Depot.id, Depot.name, Depot.station_id
            ).all()
        }

        # Size weight in 12 m-equivalent slots per vehicle type. Diesel vehicle
        # types carry no length, so fall back to their electric counterpart's
        # length (strip the "Diesel " prefix) to stay comparable to the
        # size-weighted depot-charger slots reported by the transition planner.
        electric_length_by_short: Dict[str, float] = {
            vt.name_short: vt.length
            for vt in vehicle_types
            if vt.length is not None and vt.name_short is not None
        }

        def _size_factor(vt: VehicleType) -> float:
            length = vt.length
            if (
                length is None
                and vt.name_short
                and vt.name_short.startswith(CreateDieselVehicleTypes.DIESEL_PREFIX)
            ):
                length = electric_length_by_short.get(
                    vt.name_short[len(CreateDieselVehicleTypes.DIESEL_PREFIX) :]
                )
            return (length / 12.0) if length else 1.0

        counts_rows: List[Dict[str, Any]] = []
        occupancy_rows: List[Dict[str, Any]] = []

        for vt in vehicle_types:
            counts_rows.append(
                {
                    "vehicle_type_id": vt.id,
                    "vehicle_type_name": vt.name,
                    "energy_source": vt.energy_source,
                    "vehicle_count": int(vehicle_counts.get(vt.id, 0)),
                }
            )

            areas_by_depot: Dict[int, List[int]] = defaultdict(list)
            for area_id, depot_id in (
                session.query(Area.id, Area.depot_id)
                .join(Event, Event.area_id == Area.id)
                .filter(
                    Area.vehicle_type_id == vt.id,
                    Event.event_type == EventType.STANDBY_DEPARTURE,
                )
                .distinct()
                .all()
            ):
                areas_by_depot[depot_id].append(area_id)

            for depot_id, area_ids in areas_by_depot.items():
                occupancy_df = eval_output_prepare.power_and_occupancy(
                    area_ids, session, temporal_resolution
                )
                depot_name, station_id = depot_info.get(depot_id, (None, None))
                occupancy_rows.append(
                    {
                        "vehicle_type_id": vt.id,
                        "vehicle_type_name": vt.name,
                        "energy_source": vt.energy_source,
                        "depot_id": depot_id,
                        "depot_name": depot_name,
                        "station_id": station_id,
                        "size_factor": _size_factor(vt),
                        "peak_occupancy": float(occupancy_df["occupancy_total"].max()),
                    }
                )

        return {
            "counts": pd.DataFrame(counts_rows),
            "occupancy": pd.DataFrame(occupancy_rows),
        }


class RevertBlocksToElectric(Modifier):
    """Reassign a set of blocks from their diesel vehicle type back to electric.

    Inverse of :class:`~eflips.x.steps.modifiers.general_utilities.VehicleTypeBlockAssignment`.
    For each target rotation whose vehicle type is a diesel counterpart
    (``"Diesel {name_short}"``), the rotation is reassigned to the matching electric
    vehicle type -- the one whose ``name_short`` is the stripped ``"{name_short}"``.
    Both the electric and the diesel vehicle types are expected to coexist in the
    scenario (as they do in the database produced by the diesel simulation).

    Used by the multi-stage co-simulation: the diesel database has *every* rotation
    on a diesel vehicle type, so before a stage is simulated the rotations already
    electrified that year (everything except the stage's unelectrified blocks) are
    reverted to electric, leaving only that year's diesel blocks as diesel.

    **Empty-trip replacement.** In the diesel database every rotation's deadhead
    (EMPTY) trips point at a *diesel* depot. Reverting the vehicle type alone leaves
    those deadheads unchanged, so an electrified rotation would still start/end at a
    diesel depot and the new electric depots -- which exist only in the electric
    database -- would never receive a rotation (``DepotGenerator`` derives depots
    from each rotation's first-departure / last-arrival station). When
    ``electric_db`` is provided, each reverted rotation's EMPTY trips are therefore
    deleted and re-created from the *same rotation id* in the electric database, so
    the electrified rotations carry their electric depot assignment (including new
    electric depots). Passenger trips are left untouched. When ``electric_db`` is
    ``None`` only the vehicle types are reverted.

    Rotations already on an electric vehicle type are skipped for the vehicle-type
    revert; a rotation whose diesel vehicle type has no electric counterpart raises
    :class:`ValueError`.

    .. note::
       Cross-database station/route resolution relies on the diesel and electric
       databases sharing the same base-network ids (the diesel simulation is
       branched from the same electric scenario). Base stations/routes are matched
       by id (guarded by name); only genuinely new electric depot stations and their
       deadhead routes are created in the diesel database.
    """

    DIESEL_PREFIX = CreateDieselVehicleTypes.DIESEL_PREFIX

    def __init__(self, code_version: str = "v1.1.0", **kwargs: Any):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            f"{cls.__name__}.block_ids": """
            Optional list of Rotation (block) ids to revert to their electric
            vehicle-type counterpart. When omitted or None, ALL diesel rotations in
            the scenario are reverted; an explicit list reverts only those; an empty
            list is a no-op.
            Default: None (revert all diesel rotations)
            Type: Optional[List[int]]
            """,
            f"{cls.__name__}.electric_db": """
            Optional path to the electric database (the output of
            run_transition_planner) whose rotations carry their final electric depot
            deadheads, including new electric depots. When provided, each reverted
            rotation's EMPTY (deadhead) trips are replaced with the same rotation's
            EMPTY trips from this database (matched by rotation id) so electrified
            rotations get their electric depot assignment. When None, only vehicle
            types are reverted and deadhead trips are left unchanged.
            Default: None
            Type: Optional[pathlib.Path]
            """,
        }

    def _electric_types_by_short(
        self, session: Session, scenario: Scenario
    ) -> Dict[str, VehicleType]:
        """Map each electric VT's ``name_short`` to the VehicleType itself.

        Electric vehicle types are those whose ``name_short`` does *not* start with
        the ``"Diesel "`` prefix.
        """
        electric_types: Dict[str, VehicleType] = {}
        for vt in session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id).all():
            if vt.name_short and not vt.name_short.startswith(self.DIESEL_PREFIX):
                electric_types[vt.name_short] = vt
        return electric_types

    @staticmethod
    def _copy_geom(geom: Any) -> Any:
        """Copy a geometry value across databases, preserving 3D coordinates."""
        if geom is None:
            return None
        return from_shape(to_shape(geom), srid=4326)

    def _resolve_station(
        self,
        session: Session,
        scenario: Scenario,
        elec_station: Station,
        cache: Dict[int, Station],
    ) -> Station:
        """Resolve an electric-db station to the diesel db, creating it if new.

        Base-network stations are matched by id (guarded by name so an id collision
        with an unrelated diesel station falls back to a name lookup); otherwise a
        matching station is looked up by name; otherwise a new station is created
        (this is the case for new electric depots).
        """
        if elec_station.id in cache:
            return cache[elec_station.id]

        station = (
            session.query(Station)
            .filter(Station.id == elec_station.id, Station.scenario_id == scenario.id)
            .one_or_none()
        )
        # Guard against an id that maps to a different station in the diesel db.
        if (
            station is not None
            and elec_station.name is not None
            and station.name != elec_station.name
        ):
            station = None
        if station is None and elec_station.name is not None:
            station = (
                session.query(Station)
                .filter(Station.name == elec_station.name, Station.scenario_id == scenario.id)
                .first()
            )
        if station is None:
            # Copy the full electrification column set together -- Station has a
            # check constraint requiring amount_charging_places / power_per_charger /
            # power_total / charge_type / voltage_level to all be set (or all NULL)
            # consistently with is_electrified.
            charging_point_type_id = elec_station.charging_point_type_id
            if charging_point_type_id is not None and (
                session.query(ChargingPointType)
                .filter(ChargingPointType.id == charging_point_type_id)
                .count()
                == 0
            ):
                self.logger.warning(
                    "ChargingPointType %s referenced by station '%s' is absent in the "
                    "diesel db; setting it to None.",
                    charging_point_type_id,
                    elec_station.name,
                )
                charging_point_type_id = None
            station = Station(
                name=elec_station.name,
                name_short=elec_station.name_short,
                scenario_id=scenario.id,
                geom=self._copy_geom(elec_station.geom),
                is_electrified=bool(elec_station.is_electrified),
                is_electrifiable=bool(elec_station.is_electrifiable),
                amount_charging_places=elec_station.amount_charging_places,
                power_per_charger=elec_station.power_per_charger,
                power_total=elec_station.power_total,
                charge_type=elec_station.charge_type,
                voltage_level=elec_station.voltage_level,
                charging_point_type_id=charging_point_type_id,
            )
            session.add(station)
            session.flush()
            self.logger.warning(
                "Created new depot station '%s' (diesel-db id %s) from the electric db.",
                elec_station.name,
                station.id,
            )

        cache[elec_station.id] = station
        return station

    def _resolve_route(
        self,
        session: Session,
        scenario: Scenario,
        elec_route: Route,
        dep: Station,
        arr: Station,
        station_cache: Dict[int, Station],
        route_cache: Dict[Any, Route],
    ) -> Route:
        """Resolve/create a deadhead route between two diesel-db stations."""
        key = (dep.id, arr.id)
        if key in route_cache:
            return route_cache[key]

        route = (
            session.query(Route)
            .filter(
                Route.scenario_id == scenario.id,
                Route.departure_station_id == dep.id,
                Route.arrival_station_id == arr.id,
            )
            .first()
        )
        if route is None:
            route = Route(
                scenario_id=scenario.id,
                departure_station=dep,
                arrival_station=arr,
                name=elec_route.name,
                distance=elec_route.distance,
                line_id=elec_route.line_id,
                geom=self._copy_geom(elec_route.geom),
            )
            assoc = []
            for a in sorted(elec_route.assoc_route_stations, key=lambda x: x.elapsed_distance):
                assoc.append(
                    AssocRouteStation(
                        scenario_id=scenario.id,
                        route=route,
                        station=self._resolve_station(session, scenario, a.station, station_cache),
                        elapsed_distance=a.elapsed_distance,
                    )
                )
            route.assoc_route_stations = assoc
            session.add(route)
            session.add_all(assoc)
            session.flush()

        route_cache[key] = route
        return route

    def _replace_empty_trips(
        self,
        session: Session,
        scenario: Scenario,
        rotation_ids: List[int],
        electric_db: Path,
    ) -> None:
        """Replace each rotation's EMPTY trips with those from the electric db."""
        electric_url = f"sqlite:////{Path(electric_db).absolute().as_posix()}"
        electric_engine = eflips.model.create_engine(electric_url)
        electric_session = Session(electric_engine)

        station_cache: Dict[int, Station] = {}
        route_cache: Dict[Any, Route] = {}
        replaced_trips = 0
        try:
            for rid in rotation_ids:
                elec_rotation = (
                    electric_session.query(Rotation).filter(Rotation.id == rid).one_or_none()
                )
                if elec_rotation is None:
                    raise ValueError(
                        f"Rotation {rid} not found in electric database {electric_db}. "
                        "The electric and diesel databases must share rotation ids."
                    )
                elec_empty_trips = [
                    t for t in elec_rotation.trips if t.trip_type == TripType.EMPTY
                ]

                diesel_rotation = session.query(Rotation).filter(Rotation.id == rid).one()
                for trip in list(diesel_rotation.trips):
                    if trip.trip_type != TripType.EMPTY:
                        continue
                    # Events were already removed by DeleteDepotEvents; delete any
                    # that remain (standalone use) before removing the trip.
                    session.query(Event).filter(Event.trip_id == trip.id).delete()
                    for st in list(trip.stop_times):
                        session.delete(st)
                    session.delete(trip)
                session.flush()

                for et in elec_empty_trips:
                    dep = self._resolve_station(
                        session, scenario, et.route.departure_station, station_cache
                    )
                    arr = self._resolve_station(
                        session, scenario, et.route.arrival_station, station_cache
                    )
                    route = self._resolve_route(
                        session, scenario, et.route, dep, arr, station_cache, route_cache
                    )
                    new_trip = Trip(
                        scenario_id=scenario.id,
                        route=route,
                        rotation_id=rid,
                        trip_type=TripType.EMPTY,
                        departure_time=et.departure_time,
                        arrival_time=et.arrival_time,
                        loaded_mass=et.loaded_mass,
                    )
                    new_stop_times = [
                        StopTime(
                            scenario_id=scenario.id,
                            trip=new_trip,
                            station=self._resolve_station(
                                session, scenario, st.station, station_cache
                            ),
                            arrival_time=st.arrival_time,
                            dwell_duration=st.dwell_duration,
                        )
                        for st in sorted(et.stop_times, key=lambda x: x.arrival_time)
                    ]
                    new_trip.stop_times = new_stop_times
                    session.add(new_trip)
                    session.add_all(new_stop_times)
                    replaced_trips += 1
                session.flush()
        finally:
            electric_session.close()
            electric_engine.dispose()

        self.logger.info(
            "Replaced empty trips for %d rotation(s) (%d empty trips) from the electric db.",
            len(rotation_ids),
            replaced_trips,
        )

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """Reassign the requested rotations back to electric, replacing deadheads."""
        scenario = session.query(Scenario).one()

        electric_types = self._electric_types_by_short(session, scenario)

        block_ids = params.get(f"{self.__class__.__name__}.block_ids", None)
        query = session.query(Rotation).filter(Rotation.scenario_id == scenario.id)
        if block_ids is None:
            rotations = query.all()
            self.logger.info("Reverting all %d rotation(s) to electric.", len(rotations))
        else:
            rotations = query.filter(Rotation.id.in_(block_ids)).all()
            self.logger.info(
                "Reverting %d of %d requested rotation(s) to electric.",
                len(rotations),
                len(block_ids),
            )

        reverted = 0
        for rotation in rotations:
            current = rotation.vehicle_type
            if not (current.name_short and current.name_short.startswith(self.DIESEL_PREFIX)):
                continue  # already electric
            source_short = current.name_short[len(self.DIESEL_PREFIX) :]
            if source_short not in electric_types:
                raise ValueError(
                    f"No electric counterpart for vehicle type '{current.name_short}'. "
                    f"Available: {sorted(electric_types.keys())}"
                )
            rotation.vehicle_type = electric_types[source_short]
            reverted += 1

        session.flush()
        self.logger.info("Reverted %d rotation(s) to electric vehicle types.", reverted)

        electric_db = params.get(f"{self.__class__.__name__}.electric_db", None)
        if electric_db is None:
            return
        self._replace_empty_trips(session, scenario, [r.id for r in rotations], Path(electric_db))


def build_diesel_depot_config_by_year(
    session: sqlalchemy.orm.session.Session,
    electric_slots_by_year: Dict[int, Dict[int, float]],
    initial_depot_capacities: Dict[int, int],
    round_up: bool = True,
) -> Dict[int, List[Dict[str, Any]]]:
    """Build a per-year :class:`PartialDepotAssignment` ``depot_config`` for diesel.

    For each operational year the diesel-available capacity of an *existing* depot
    is its initial diesel slot count minus the depot-charger slots already occupied
    by the electric fleet (electrified + under construction). The electric slots are
    the cumulative, size-weighted charger footprint ready by the start of the next
    year (see
    :meth:`eflips.transition.transition_planning.TransitionPlannerModel.get_depot_electric_slots_by_year`).

    Only depots present in ``initial_depot_capacities`` (the existing depots measured
    from the diesel simulation) are emitted. New, electric-only depots are never
    included -- diesel buses are not parked there -- and electric slots reported for
    them are ignored.

    :param session: session on the diesel database (which holds both the electric
        and the diesel vehicle types). Used to resolve the diesel vehicle-type ids
        allowed at every depot.
    :param electric_slots_by_year: ``{operational_year: {station_id: electric_slots}}``,
        the ``"depot_electric_slots_by_year_map"`` entry of the transition-planner
        result. ``station_id`` keys not present default to zero electric slots.
    :param initial_depot_capacities: ``{station_id: capacity}`` from
        :class:`~eflips.transition.parameter_registry.DieselFleetParams`; defines the
        set of existing depots and their baseline diesel capacity (12 m slots).
    :param round_up: when True (default) the electric footprint is reserved with
        :func:`math.ceil`, so diesel never overflows into an electrified or
        under-construction slot. When False it is truncated (``int``).
    :return: ``{operational_year: [depot_config_dict, ...]}`` ready to hand to
        :class:`PartialDepotAssignment` as its ``depot_config`` parameter. Each dict
        is ``{"depot_station": station_id, "capacity": remaining_diesel_slots,
        "vehicle_type": [diesel_vt_id, ...]}``.
    """
    diesel_vt_ids = [
        vt_id
        for (vt_id,) in session.query(VehicleType.id).filter(
            VehicleType.energy_source == EnergySource.DIESEL
        )
    ]

    depot_config_by_year: Dict[int, List[Dict[str, Any]]] = {}
    for year, slots_by_station in electric_slots_by_year.items():
        depot_config: List[Dict[str, Any]] = []
        for station_id, initial_capacity in initial_depot_capacities.items():
            electric_slots = slots_by_station.get(station_id, 0.0)
            reserved = math.ceil(electric_slots) if round_up else int(electric_slots)
            depot_config.append(
                {
                    "depot_station": station_id,
                    "capacity": max(0, initial_capacity - reserved),
                    "vehicle_type": diesel_vt_ids,
                }
            )
        depot_config_by_year[int(year)] = depot_config

    return depot_config_by_year


@task
def run_hybrid_fleet_simulation(
    context: PipelineContext,
    steps: List[PipelineStep],
    plot_output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Run simulation steps serially, collect VehicleTypeStatistics, then generate plots.

    The final step must be :class:`VehicleTypeStatistics`; its result
    (``{"counts": DataFrame, "occupancy": DataFrame}``) is returned so the
    caller can aggregate per-stage statistics for cross-stage plotting.
    """
    vt_stats_result: Dict[str, pd.DataFrame] = {}
    for step in steps:
        if isinstance(step, VehicleTypeStatistics):
            vt_stats_result = step.execute(context=context)
        else:
            step.execute(context=context)

    generate_simple_plots(
        context=context,
        output_dir=plot_output_dir,
    )

    return vt_stats_result


def generate_simple_plots(
    context: PipelineContext,
    output_dir: Path,
) -> None:
    """Generate plots for the transition plan results."""
    # Placeholder for actual plot generation logic based on results
    # For example, you could generate a plot showing the unelectrified blocks and depot locations

    logger.info(f"Starting analysis flow, outputs will be saved to: {output_dir}")

    # Define output directories
    simple_dir = output_dir / "simple"

    simple_dir.mkdir(parents=True, exist_ok=True)

    # Define simple analyzers by category
    pre_simulation_analyzers = [
        RotationInfoAnalyzer,
        GeographicTripPlotAnalyzer,
    ]

    post_simulation_analyzers = [
        DepartureArrivalSocAnalyzer,
        DepotEventAnalyzer,
        SpecificEnergyConsumptionAnalyzer,
    ]

    analyzers_to_run = pre_simulation_analyzers + post_simulation_analyzers

    all_futures = []
    for analyzer_class in analyzers_to_run:
        output_file = simple_dir / f"{analyzer_class.__name__}.html"
        future = execute_simple_analyzer.submit(analyzer_class, context, output_file)  # type: ignore[type-abstract]
        all_futures.append(future)

    # Execute InteractiveMapAnalyzer with depot and station plot directories
    logger.info("Submitting InteractiveMapAnalyzer...")
    map_output_file = output_dir / "InteractiveMapAnalyzer.html"

    logger.info("All analysis tasks submitted. Waiting for completion...")
    wait(all_futures)
    logger.info("Analysis flow complete. All outputs saved.")


@flow(
    task_runner=ProcessPoolTaskRunner(max_workers=4),
)
def simulate_multi_stage_electrification(
    unelectrified_blocks: Dict[int, List[int]],
    electric_slots_by_year: Dict[int, Dict[int, float]],
    initial_depot_capacities: Dict[int, int],
    workdir: Path,
    input_db: Path,
    electric_db: Path,
    base_url: Optional[str] = None,
    log_level: str = "INFO",
    csv_dir: Optional[Path] = None,
    cleanup_intermediate_dbs: bool = False,
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Co-simulate the fleet year by year as depot capacity is taken by electric.

    Runs one stage per transition year, starting from the diesel database (every
    rotation on a diesel vehicle type, plus the electric vehicle types). Each stage
    reverts the rotations already electrified that year back to electric -- so only
    the year's unelectrified blocks remain diesel -- and reassigns just those diesel
    blocks to depots whose diesel capacity has been reduced by the electric
    depot-charger footprint (electrified + under construction). Electric and diesel
    are then simulated together.

    Per stage (serial):
    ``DeleteDepotEvents -> RevertBlocksToElectric -> PartialDepotAssignment ->
    DepotGenerator -> Simulation -> VehicleTypeStatistics -> Plots``. Stages run in
    parallel via ProcessPoolTaskRunner.

    Only years with a non-empty diesel block set *and* a per-year depot config are
    simulated (operational years ``1 .. T-1``; by year ``T`` the fleet is fully
    electric, so there is no diesel stage).

    Args:
        unelectrified_blocks: ``{year: [rotation_id, ...]}`` diesel blocks per year
            (``TransitionPlanner`` result ``"unelectrified_blocks"``).
        electric_slots_by_year: ``{year: {station_id: electric_slots}}`` cumulative
            size-weighted electric charger footprint per depot per year
            (``TransitionPlanner`` result ``"depot_electric_slots_by_year_map"``).
        initial_depot_capacities: ``{station_id: capacity}`` from
            :class:`~eflips.transition.parameter_registry.DieselFleetParams`.
        workdir: Base working directory for the per-stage databases.
        input_db: Path to the diesel database (post diesel simulation). Its rotation
            and station ids must match those in ``unelectrified_blocks`` /
            ``electric_slots_by_year`` (i.e. the diesel sim branched from the same
            electric scenario).
        electric_db: Path to the electric database (``run_transition_planner``
            output) whose rotations carry their final electric depot deadheads
            (including new electric depots). Each stage's electrified rotations take
            their EMPTY (deadhead) trips from this database, matched by rotation id,
            via :class:`RevertBlocksToElectric`.
        base_url: OpenRouteService base URL for :class:`PartialDepotAssignment`.
            When None the step falls back to the ``OPENROUTESERVICE_BASE_URL``
            environment variable.
        log_level: Logging level for the pipeline steps.
        csv_dir: Directory the per-stage statistics (``per_stage_occupancy.csv``,
            ``per_stage_counts.csv``), the depot-name lookup and the initial depot
            capacities are written to, so the slot-distribution plots can be
            regenerated without re-running the simulation. Defaults to
            ``workdir / "csv"`` (the same directory the transition-planner CSVs
            are written to).
        cleanup_intermediate_dbs: When True, delete each stage's intermediate
            ``step_001`` .. ``step_004`` databases once every stage has finished
            without error and the plotting CSVs are written, keeping only the
            final ``step_005_Simulation.db`` per stage. These four scratch copies
            per stage (~88 MB/stage) are only consumed within the stage pipeline;
            nothing downstream of this flow reads them (the plots come from the
            CSVs). Deleting them forgoes Prefect cache reuse, so a subsequent run
            re-simulates the affected stages. Defaults to False.

    Returns:
        Dict keyed by year, each value the :class:`VehicleTypeStatistics` result
        (``{"counts": DataFrame, "occupancy": DataFrame}``) for that stage.
    """
    # Per-year diesel depot capacities (initial minus the electric footprint),
    # built once from a read-only session on the diesel database.
    config_context = PipelineContext(work_dir=workdir, current_db=input_db, params={})
    with config_context.get_session() as session:
        depot_config_by_year = build_diesel_depot_config_by_year(
            session, electric_slots_by_year, initial_depot_capacities
        )

    from eflips.x.flows.analysis_flow import query_all_ids

    all_rotation_ids = query_all_ids(config_context, Rotation)

    parallel_flows: List[Any] = []
    stage_ids: List[int] = []

    for i, stage_blocks in unelectrified_blocks.items():
        # Skip years with no diesel buses or no depot config (e.g. the fully
        # electric final year): PartialDepotAssignment also rejects empty rotations.
        if not stage_blocks or i not in depot_config_by_year:
            continue

        run_id = f"stage_{i}"
        stage_workdir = workdir / run_id
        os.makedirs(stage_workdir, exist_ok=True)

        electrified_ids = [b_id for b_id in all_rotation_ids if b_id not in stage_blocks]

        # Create fresh params for this stage
        params: Dict[str, Any] = {
            "log_level": log_level,
            # Revert everything already electrified this year back to electric; only
            # this year's unelectrified blocks stay diesel. Electrified rotations
            # also take their electric depot deadheads from the electric database.
            "RevertBlocksToElectric.block_ids": electrified_ids,
            "RevertBlocksToElectric.electric_db": electric_db,
            # Reassign the diesel blocks under the reduced depot capacity.
            "PartialDepotAssignment.rotation_ids": stage_blocks,
            "PartialDepotAssignment.depot_config": depot_config_by_year[i],
            "DepotGenerator.generate_optimal_depot": False,
            "Simulation.smart_charging": SmartChargingStrategy.NONE,
            "GeographicTripPlotAnalyzer.rotation_ids": electrified_ids,
            "GeographicTripPlotAnalyzer.plot_charging_station": True,
            "GeographicTripPlotAnalyzer.plot_depot_charger_count": True,
        }
        # Only set base_url when provided; otherwise PartialDepotAssignment reads it
        # from the OPENROUTESERVICE_BASE_URL environment variable.
        if base_url is not None:
            params["PartialDepotAssignment.base_url"] = base_url

        # Create context for this stage
        sub_context = PipelineContext(
            work_dir=stage_workdir,
            current_db=input_db,
            params=params,
        )

        # Build pipeline steps
        steps: List[PipelineStep] = [
            DeleteDepotEvents(),
            RevertBlocksToElectric(),
            PartialDepotAssignment(),
            DepotGenerator(),
            Simulation(),
            VehicleTypeStatistics(),
        ]

        # Submit stage for parallel execution
        parallel_flows.append(
            run_hybrid_fleet_simulation.submit(
                context=sub_context,
                steps=steps,
                plot_output_dir=stage_workdir / "plots",
            )
        )
        stage_ids.append(i)

    print("Waiting for simulations to complete...")
    per_stage_results: Dict[int, Dict[str, pd.DataFrame]] = {}
    for stage_id, pf in zip(stage_ids, parallel_flows):
        per_stage_results[stage_id] = pf.result()
    print("All stages completed.")

    # Persist the per-stage statistics and the plotting inputs so the slot
    # distribution plots can be regenerated without re-running the simulation.
    _save_multi_stage_csvs(
        per_stage_results,
        initial_depot_capacities,
        Path(csv_dir) if csv_dir is not None else workdir / "csv",
    )

    # Reaching here means every stage's ``.result()`` returned (no error) and the
    # plotting CSVs are on disk, so the intermediate per-stage databases are safe
    # to drop when requested.
    if cleanup_intermediate_dbs:
        _cleanup_intermediate_stage_dbs([workdir / f"stage_{i}" for i in stage_ids])

    return per_stage_results


def _save_multi_stage_csvs(
    per_stage_results: Dict[int, Dict[str, pd.DataFrame]],
    initial_depot_capacities: Dict[int, int],
    csv_dir: Path,
) -> None:
    """Write per-stage statistics + inputs used by the slot-distribution plots.

    Emits into ``csv_dir``:

    - ``initial_depot_capacities.csv`` — ``depot_id, capacity`` (the diesel
      baseline slots, so :func:`plot_slot_distributions` runs standalone).
    - ``per_stage_occupancy.csv`` — every stage's ``VehicleTypeStatistics``
      occupancy rows with a ``year`` (stage) and an ``is_diesel`` column.
    - ``per_stage_counts.csv`` — the same for the vehicle-count rows.
    - ``depot_names.csv`` — ``station_id, depot_name`` lookup for plot labels.
    """
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"depot_id": d, "capacity": c} for d, c in initial_depot_capacities.items()]
    ).to_csv(csv_dir / "initial_depot_capacities.csv", index=False)

    occ_frames: List[pd.DataFrame] = []
    cnt_frames: List[pd.DataFrame] = []
    for year, res in per_stage_results.items():
        occ = res.get("occupancy")
        if occ is not None and not occ.empty:
            occ_frames.append(occ.assign(year=year))
        cnt = res.get("counts")
        if cnt is not None and not cnt.empty:
            cnt_frames.append(cnt.assign(year=year))

    if occ_frames:
        occ_all = pd.concat(occ_frames, ignore_index=True)
        occ_all["is_diesel"] = occ_all["energy_source"] == EnergySource.DIESEL
        occ_all.to_csv(csv_dir / "per_stage_occupancy.csv", index=False)
        if "station_id" in occ_all.columns:
            (
                occ_all[["station_id", "depot_name"]]
                .dropna()
                .drop_duplicates()
                .to_csv(csv_dir / "depot_names.csv", index=False)
            )
    if cnt_frames:
        cnt_all = pd.concat(cnt_frames, ignore_index=True)
        cnt_all["is_diesel"] = cnt_all["energy_source"] == EnergySource.DIESEL
        cnt_all.to_csv(csv_dir / "per_stage_counts.csv", index=False)


def _cleanup_intermediate_stage_dbs(stage_workdirs: List[Path]) -> None:
    """Delete each stage's intermediate databases, keeping ``*_Simulation.db``.

    Every stage copies the database once per :class:`Modifier` step
    (``step_001`` .. ``step_004``), leaving four ~22 MB scratch copies per stage
    alongside the ~22 MB simulated result. Only the final ``*_Simulation.db`` has
    downstream value; the earlier copies exist solely to feed the next step and
    Prefect's cache. Call only after the flow has finished without error and the
    plotting CSVs are written. Errors deleting a file are logged, not raised.
    """
    logger = logging.getLogger(__name__)
    freed = 0
    removed = 0
    for stage_workdir in stage_workdirs:
        for db_file in sorted(Path(stage_workdir).glob("step_*.db")):
            # Keep the final simulated database; drop the earlier scratch copies.
            if db_file.name.endswith("_Simulation.db"):
                continue
            try:
                size = db_file.stat().st_size
                db_file.unlink()
                freed += size
                removed += 1
            except OSError as exc:
                logger.warning("Could not delete %s: %s", db_file, exc)
    if removed:
        logger.info(
            "Cleaned up %d intermediate stage databases (%.1f MB freed).",
            removed,
            freed / 1e6,
        )


DIESEL_HATCH = "///"


def _vehicle_type_color_map(vt_names: List[str]) -> Dict[str, Any]:
    """Stable color per vehicle type, shared across both plots."""
    cmap = plt.get_cmap("tab20")
    return {name: cmap(i % cmap.N) for i, name in enumerate(vt_names)}


def _hatch_legend_handles() -> List[mpatches.Patch]:
    return [
        mpatches.Patch(facecolor="white", edgecolor="black", label="Electric"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=DIESEL_HATCH, label="Diesel"),
    ]


def plot_multi_stage_vehicle_type_statistics(
    per_stage_results: Dict[int, Dict[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Render cross-stage VehicleTypeStatistics plots.

    Writes two PNGs into ``output_dir``:

    - ``vehicle_count_per_stage.png`` — stacked bars of vehicle count per
      vehicle type, one bar per stage. Diesel types are hatched.
    - ``peak_occupancy_per_stage_depot.png`` — grouped (by depot) + stacked
      (by vehicle type) bars of peak depot occupancy, one group per stage.
      Diesel types are hatched (typically absent — diesel has no depot
      events).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_ids = sorted(per_stage_results.keys())
    counts = pd.concat(
        [per_stage_results[s]["counts"].assign(stage=s) for s in stage_ids],
        ignore_index=True,
    )
    occupancy = pd.concat(
        [per_stage_results[s]["occupancy"].assign(stage=s) for s in stage_ids],
        ignore_index=True,
    )

    # Shared across both plots: stable color + diesel lookup per VT name.
    all_vt_names = sorted(set(counts["vehicle_type_name"]).union(occupancy["vehicle_type_name"]))
    vt_colors = _vehicle_type_color_map(all_vt_names)
    diesel_by_vt = {
        name: energy == EnergySource.DIESEL
        for name, energy in pd.concat(
            [
                counts[["vehicle_type_name", "energy_source"]],
                occupancy[["vehicle_type_name", "energy_source"]],
            ]
        )
        .drop_duplicates()
        .itertuples(index=False)
    }

    _plot_vehicle_count_per_stage(counts, stage_ids, vt_colors, diesel_by_vt, output_dir)
    _plot_peak_occupancy_per_stage_depot(occupancy, stage_ids, vt_colors, diesel_by_vt, output_dir)


def _plot_vehicle_count_per_stage(
    counts: pd.DataFrame,
    stage_ids: List[int],
    vt_colors: Dict[str, Any],
    diesel_by_vt: Dict[str, bool],
    output_dir: Path,
) -> None:
    pivot = (
        counts.pivot_table(
            index="stage",
            columns="vehicle_type_name",
            values="vehicle_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(stage_ids)
        .fillna(0)
    )

    fig, ax = plt.subplots(layout="constrained")
    x = np.arange(len(stage_ids))
    bottom = np.zeros(len(stage_ids))
    for vt_name in pivot.columns:
        heights = pivot[vt_name].to_numpy()
        ax.bar(
            x,
            heights,
            bottom=bottom,
            color=vt_colors[vt_name],
            edgecolor="black",
            hatch=DIESEL_HATCH if diesel_by_vt.get(vt_name) else None,
            label=vt_name,
        )
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in stage_ids])
    ax.set_xlabel("Stage")
    ax.set_ylabel("Vehicle count")
    ax.set_title("Vehicle count per type per stage")

    type_legend = ax.legend(title="Vehicle type", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.add_artist(type_legend)
    ax.legend(
        handles=_hatch_legend_handles(),
        title="Energy source",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
    )

    fig.savefig(output_dir / "vehicle_count_per_stage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_peak_occupancy_per_stage_depot(
    occupancy: pd.DataFrame,
    stage_ids: List[int],
    vt_colors: Dict[str, Any],
    diesel_by_vt: Dict[str, bool],
    output_dir: Path,
) -> None:
    if occupancy.empty:
        logger.warning("No peak occupancy data; skipping peak occupancy plot.")
        return

    depots = sorted(occupancy["depot_name"].dropna().unique().tolist())
    pivot = occupancy.pivot_table(
        index=["stage", "depot_name"],
        columns="vehicle_type_name",
        values="peak_occupancy",
        aggfunc="max",
        fill_value=0,
    )

    x = np.arange(len(stage_ids))

    for depot in depots:
        fig, ax = plt.subplots(layout="constrained")
        bottom = np.zeros(len(stage_ids))
        for vt_name in pivot.columns:
            heights = np.array(
                [
                    (pivot.loc[(s, depot), vt_name] if (s, depot) in pivot.index else 0.0)
                    for s in stage_ids
                ]
            )
            if not heights.any():
                continue
            ax.bar(
                x,
                heights,
                bottom=bottom,
                width=0.6,
                color=vt_colors[vt_name],
                edgecolor="black",
                hatch=DIESEL_HATCH if diesel_by_vt.get(vt_name) else None,
                label=vt_name,
            )
            bottom += heights

        ax.set_xticks(x)
        ax.set_xticklabels([f"Stage {s}" for s in stage_ids])
        ax.set_xlabel("Stage")
        ax.set_ylabel("Peak occupancy")
        ax.set_title(f"Peak depot occupancy — {depot}")

        type_handles = [
            mpatches.Patch(
                facecolor=vt_colors[vt],
                edgecolor="black",
                hatch=DIESEL_HATCH if diesel_by_vt.get(vt) else None,
                label=vt,
            )
            for vt in pivot.columns
        ]
        type_legend = ax.legend(
            handles=type_handles,
            title="Vehicle type",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
        )
        ax.add_artist(type_legend)
        ax.legend(
            handles=_hatch_legend_handles(),
            title="Energy source",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
        )

        safe_name = depot.replace(" ", "_").replace("/", "-")
        fig.savefig(
            output_dir / f"peak_occupancy_per_stage_{safe_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Depot slot-distribution plots (planned vs. simulated)
#
# Both plots stack, per depot per operational year, the depot's slots
# (12 m-equivalents) into three segments:
#   * in operation (electric)  -- electric depot chargers already ready/in use
#   * under construction        -- electric chargers being built that year
#   * in operation (diesel)     -- capacity/occupancy still served by diesel
#
# The electric operation / under-construction split is derived from the
# accumulated depot_electric_slots_by_year.csv (E(i) = cumulative slots ready by
# ready_year i+1): in operation = E(i-1) (ready at the start of year i), under
# construction = E(i) - E(i-1) (being built that year, ready next year).
# "Planned" diesel = initial capacity - E(i); "real" electric/diesel come from
# the per-stage peak occupancy, with the same under-construction segment from
# the transition planner (the simulation does not build chargers). Both read
# purely from ``csv_dir`` so they can be regenerated without re-running.
# --------------------------------------------------------------------------- #

_SLOT_SEGMENTS = ["in_operation_electric", "under_construction", "in_operation_diesel"]
_SLOT_STYLE: Dict[str, Dict[str, Any]] = {
    "in_operation_electric": {
        "color": "#2ca02c",
        "hatch": None,
        "label": "In operation — electric",
    },
    "under_construction": {
        "color": "#ff7f0e",
        "hatch": "///",
        "label": "Under construction",
    },
    "in_operation_diesel": {
        "color": "#8c8c8c",
        "hatch": None,
        "label": "In operation — diesel",
    },
}


def _load_depot_names(csv_dir: Path) -> Dict[int, str]:
    """Load a ``station_id -> depot_name`` label lookup, empty if unavailable."""
    path = Path(csv_dir) / "depot_names.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        int(row.station_id): str(row.depot_name)
        for row in df.itertuples(index=False)
        if not pd.isna(row.station_id)
    }


def _load_initial_capacities(
    csv_dir: Path, initial_depot_capacities: Optional[Dict[int, int]]
) -> Dict[int, float]:
    """Resolve initial depot capacities from the argument or the saved CSV."""
    if initial_depot_capacities is not None:
        return {int(k): float(v) for k, v in initial_depot_capacities.items()}
    path = Path(csv_dir) / "initial_depot_capacities.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {int(row.depot_id): float(row.capacity) for row in df.itertuples(index=False)}


def _electric_split_by_depot_year(csv_dir: Path) -> pd.DataFrame:
    """Split the accumulated electric depot-charger slots into operation / build.

    Derived solely from ``depot_electric_slots_by_year.csv``, whose (already
    size-weighted) cumulative ``electric_slots`` ``E(i)`` is indexed by
    ``operational_year`` ``i`` and becomes ready at ``ready_year = i + 1``.
    Per depot per operational year ``i``:

    - ``reserved`` = ``E(i)`` — electrified + under construction; this is the
      footprint the diesel depot-config builder subtracts from the initial
      capacity, so ``in_operation_electric + under_construction`` sums to it.
    - ``in_operation_electric`` = ``E(i - 1)`` — slots already ready at the
      start of year ``i`` (``E(0) = 0``).
    - ``under_construction`` = ``E(i) - E(i - 1)`` — slots being built during
      year ``i`` (ready at the start of year ``i + 1``).

    Returns columns ``depot_id, year, reserved, in_operation_electric,
    under_construction``.
    """
    cols = ["depot_id", "year", "reserved", "in_operation_electric", "under_construction"]
    path = Path(csv_dir) / "depot_electric_slots_by_year.csv"
    if not path.exists():
        return pd.DataFrame(columns=cols)
    electric = pd.read_csv(path)
    split = (
        electric.groupby(["depot_id", "operational_year"])["electric_slots"]
        .sum()
        .rename("reserved")
        .reset_index()
        .rename(columns={"operational_year": "year"})
        .sort_values(["depot_id", "year"])
    )
    prev = split.groupby("depot_id")["reserved"].shift(1).fillna(0.0)
    split["in_operation_electric"] = prev
    split["under_construction"] = (split["reserved"] - prev).clip(lower=0.0)
    return split[cols]


def _planned_slot_frame(
    csv_dir: Path, initial_depot_capacities: Optional[Dict[int, int]]
) -> pd.DataFrame:
    """Build the planned per-depot per-year slot decomposition.

    The electric ``in operation`` / ``under construction`` split and the total
    reserved footprint (electrified + under construction) come from the
    accumulated ``depot_electric_slots_by_year.csv`` via
    :func:`_electric_split_by_depot_year`; diesel gets whatever is left of the
    depot's initial capacity (``initial - reserved``). Depots with no initial
    capacity (new electric-only depots) get zero diesel slots.
    """
    split = _electric_split_by_depot_year(csv_dir)
    caps = _load_initial_capacities(csv_dir, initial_depot_capacities)

    years = sorted(split["year"].unique())
    depot_ids = sorted(set(split["depot_id"]).union(caps.keys()))
    grid = pd.MultiIndex.from_product([depot_ids, years], names=["depot_id", "year"]).to_frame(
        index=False
    )

    df = grid.merge(split, on=["depot_id", "year"], how="left")
    for col in ("reserved", "in_operation_electric", "under_construction"):
        df[col] = df[col].fillna(0.0)
    df["capacity"] = df["depot_id"].map(caps)
    df["in_operation_diesel"] = (df["capacity"] - df["reserved"]).clip(lower=0.0).fillna(0.0)
    return df[["depot_id", "year", *_SLOT_SEGMENTS]]


def _real_slot_frame(csv_dir: Path) -> pd.DataFrame:
    """Build the simulated per-depot per-year slot decomposition.

    Electric/diesel come from the per-stage peak occupancy (converted from
    vehicle counts to 12 m-equivalent slots via ``size_factor``); the
    under-construction segment is taken from the transition planner
    (``depot_electric_slots_by_year.csv`` via
    :func:`_electric_split_by_depot_year`), since the simulation does not build
    chargers.
    """
    occ = pd.read_csv(Path(csv_dir) / "per_stage_occupancy.csv")
    occ = occ[occ["station_id"].notna()].copy()
    occ["depot_id"] = occ["station_id"].astype(int)
    occ["slots"] = occ["peak_occupancy"] * occ["size_factor"]
    if "is_diesel" in occ.columns:
        occ["is_diesel"] = occ["is_diesel"].astype(bool)
    else:
        occ["is_diesel"] = occ["energy_source"].astype(str).str.upper().str.contains("DIESEL")

    elec = (
        occ[~occ["is_diesel"]]
        .groupby(["depot_id", "year"])["slots"]
        .sum()
        .rename("in_operation_electric")
    )
    dies = (
        occ[occ["is_diesel"]]
        .groupby(["depot_id", "year"])["slots"]
        .sum()
        .rename("in_operation_diesel")
    )
    df = pd.concat([elec, dies], axis=1)
    for col in ("in_operation_electric", "in_operation_diesel"):
        if col not in df.columns:
            df[col] = 0.0
    df = df.fillna(0.0).reset_index()

    uc = _electric_split_by_depot_year(csv_dir)[["depot_id", "year", "under_construction"]]
    df = df.merge(uc, on=["depot_id", "year"], how="left")
    df["under_construction"] = df["under_construction"].fillna(0.0)
    return df[["depot_id", "year", *_SLOT_SEGMENTS]]


def _draw_slot_distribution(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    depot_names: Optional[Dict[int, str]] = None,
    ylabel: str = "Depot slots (12 m equiv.)",
) -> None:
    """Render one stacked-bar subplot per depot (x = operational year)."""
    depot_names = depot_names or {}
    depot_ids = sorted(df["depot_id"].unique())
    all_years = sorted(df["year"].unique())
    n = len(depot_ids)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(11, 3.0 * n + 0.8),
        layout="constrained",
        sharex=True,
        squeeze=False,
    )
    x = np.array(all_years)
    for ax, depot_id in zip(axes[:, 0], depot_ids):
        sub = df[df["depot_id"] == depot_id].set_index("year").reindex(all_years).fillna(0.0)
        bottom = np.zeros(len(x))
        for seg in _SLOT_SEGMENTS:
            style = _SLOT_STYLE[seg]
            heights = sub[seg].to_numpy()
            ax.bar(
                x,
                heights,
                bottom=bottom,
                color=style["color"],
                edgecolor="black",
                linewidth=0.3,
                hatch=style["hatch"],
                label=style["label"],
            )
            bottom += heights
        name = depot_names.get(int(depot_id), f"Depot {depot_id}")
        ax.set_title(f"{name} (station {depot_id})", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.margins(x=0.02)

    axes[-1, 0].set_xlabel("Operational year")
    axes[-1, 0].set_xticks(all_years)

    handles = [
        mpatches.Patch(
            facecolor=_SLOT_STYLE[seg]["color"],
            edgecolor="black",
            hatch=_SLOT_STYLE[seg]["hatch"],
            label=_SLOT_STYLE[seg]["label"],
        )
        for seg in _SLOT_SEGMENTS
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.suptitle(title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_slot_distributions(
    csv_dir: Path,
    output_dir: Path,
    initial_depot_capacities: Optional[Dict[int, int]] = None,
) -> None:
    """Render the planned and simulated depot slot-distribution plots.

    Reads everything from ``csv_dir`` so it can run standalone from already
    saved data (no simulation needed):

    - planned: ``depot_electric_slots_by_year.csv`` (both the electric
      operation / under-construction split and the reserved footprint) and the
      initial depot capacities (from ``initial_depot_capacities`` or, when None,
      ``initial_depot_capacities.csv``).
    - simulated: ``per_stage_occupancy.csv`` (written by
      :func:`simulate_multi_stage_electrification`) plus the under-construction
      slots from the same ``depot_electric_slots_by_year.csv``. Skipped with a
      warning if the per-stage file is absent.

    Writes ``planned_slot_distribution.{png,csv}`` and, when per-stage data is
    present, ``real_slot_distribution.{png,csv}`` into ``output_dir``.
    """
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    depot_names = _load_depot_names(csv_dir)

    planned = _planned_slot_frame(csv_dir, initial_depot_capacities)
    planned.to_csv(output_dir / "planned_slot_distribution.csv", index=False)
    _draw_slot_distribution(
        planned,
        output_dir / "planned_slot_distribution.png",
        title="Planned depot slot distribution (transition plan)",
        depot_names=depot_names,
    )

    if (csv_dir / "per_stage_occupancy.csv").exists():
        real = _real_slot_frame(csv_dir)
        real.to_csv(output_dir / "real_slot_distribution.csv", index=False)
        _draw_slot_distribution(
            real,
            output_dir / "real_slot_distribution.png",
            title="Simulated depot slot distribution (per-stage peak occupancy)",
            depot_names=depot_names,
        )
    else:
        logger.warning(
            "per_stage_occupancy.csv not found in %s; skipping the simulated "
            "slot-distribution plot.",
            csv_dir,
        )
