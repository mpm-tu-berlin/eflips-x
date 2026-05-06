"""
GTFS-specific utility modifiers for the eflips-x pipeline.

This module contains modifiers shared across GTFS-based flows.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import sqlalchemy.orm
from eflips.model import (
    ConsumptionLut as DbConsumptionLut,
    EnergySource,
    Rotation,
    Scenario,
    Trip,
    VehicleClass,
    VehicleType,
)

from eflips.x.framework import Modifier
from eflips.x.steps.modifiers.consumption_luts import (
    CONSUMPTION_LUT_DIR,
    ConsumptionLut,
    load_consumption_lut_df,
)

logger = logging.getLogger(__name__)


class _Unset:
    """Sentinel: 'do not pass this kwarg, let the model use its server default'."""


_UNSET = _Unset()


# Per-VehicleType field defaults. Each can be overridden in `params` as either
# a scalar (broadcast to all VTs) or a Dict[str, value] keyed by VT name.
# Values of `_UNSET` are skipped during VehicleType construction so the model's
# server defaults take effect.
_FIELD_DEFAULTS: Dict[str, Any] = {
    "name_short": None,
    "energy_source": EnergySource.BATTERY_ELECTRIC,
    "battery_capacity": 360.0,
    "battery_capacity_reserve": _UNSET,
    "charging_curve": [[0.0, 450.0], [1.0, 450.0]],
    "v2g_curve": None,
    "charging_efficiency": _UNSET,
    "opportunity_charging_capable": True,
    "minimum_charging_power": _UNSET,
    "length": None,
    "width": None,
    "height": None,
    "empty_mass": None,
    "allowed_mass": None,
    "tco_parameters": _UNSET,
    "consumption": 1.5,
}


class ConfigureVehicleTypes(Modifier):
    """Owns VehicleType configuration for GTFS-driven flows.

    Creates a fresh set of VehicleTypes from per-property parameters, drops
    any pre-existing VTs, reassigns every rotation to the new "default" VT
    (the first name in ``vehicle_type_names``), and optionally splits
    rotations so individually listed trips can be pinned to a specific VT.

    A ``consumption`` value may be a ``ConsumptionLut`` enum member; in that
    case the VehicleType's scalar consumption is left NULL and a
    ``VehicleClass`` plus ``eflips.model.ConsumptionLut`` are created from the
    bundled CSV.
    """

    def __init__(self, code_version: str = "2", **kwargs: Any) -> None:
        # Hash all bundled LUTs so editing one of them invalidates the cache.
        lut_files: List[Union[str, Path]] = sorted(CONSUMPTION_LUT_DIR.glob("*.csv"))
        super().__init__(additional_files=lut_files, code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        prefix = cls.__name__
        scalar_or_dict = (
            "Scalar (broadcast to every VT) or Dict[str, value] keyed by "
            "VehicleType.name. When a dict, its keys must equal "
            "vehicle_type_names exactly."
        )
        docs: Dict[str, str] = {
            f"{prefix}.vehicle_type_names": (
                "Required. List[str] of VehicleType names to create. The first "
                f"name is the default VT for rotations not pinned by "
                f"{prefix}.trip_to_vehicle_type."
            ),
            f"{prefix}.rotation_opportunity_charging": (
                "Bool. Applied to every Rotation.allow_opportunity_charging. " "Default: True."
            ),
            f"{prefix}.trip_to_vehicle_type": (
                "Optional Dict[int (Trip.id), str (VehicleType.name)]. Trips "
                "listed here are peeled off into single-trip rotations bound "
                "to the named VT."
            ),
        }
        for field, default in _FIELD_DEFAULTS.items():
            shown_default = (
                "(model server default)" if isinstance(default, _Unset) else repr(default)
            )
            docs[f"{prefix}.{field}"] = (
                f"{scalar_or_dict} Maps to VehicleType.{field}. " f"Default: {shown_default}."
            )
        docs[f"{prefix}.consumption"] += (
            " May also be a ConsumptionLut enum member; in that case a "
            "VehicleClass + ConsumptionLut row is created and "
            "VehicleType.consumption is left NULL."
        )
        return docs

    def modify(self, session: sqlalchemy.orm.Session, params: Dict[str, Any]) -> None:
        prefix = self.__class__.__name__

        names = params.get(f"{prefix}.vehicle_type_names")
        if not names:
            raise ValueError(f"{prefix}.vehicle_type_names is required (List[str] of VT names).")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate names in {prefix}.vehicle_type_names: {names}")
        name_set: Set[str] = set(names)

        per_vt = self._resolve_per_vt(prefix, names, name_set, params)

        rotation_opp_charging: bool = params.get(f"{prefix}.rotation_opportunity_charging", True)
        trip_to_vt: Optional[Dict[int, str]] = params.get(f"{prefix}.trip_to_vehicle_type")

        scenario_id = self._unique_scenario_id(session)

        new_vts: Dict[str, VehicleType] = {}
        for name in names:
            new_vts[name] = self._build_vehicle_type(
                session, scenario_id=scenario_id, name=name, fields=per_vt[name]
            )
        session.flush()
        default_vt = new_vts[names[0]]

        for rot in session.query(Rotation).all():
            rot.vehicle_type = default_vt
            rot.allow_opportunity_charging = rotation_opp_charging

        if trip_to_vt:
            unknown = set(trip_to_vt.values()) - name_set
            if unknown:
                raise ValueError(
                    f"{prefix}.trip_to_vehicle_type references unknown VT names: "
                    f"{sorted(unknown)}"
                )
            self._apply_trip_overrides(
                session,
                trip_to_vt=trip_to_vt,
                new_vts=new_vts,
                rotation_opp_charging=rotation_opp_charging,
                scenario_id=scenario_id,
            )

        session.flush()

        # Drop any pre-existing VTs (those not created by us).
        kept_ids = {vt.id for vt in new_vts.values()}
        for old in session.query(VehicleType).filter(~VehicleType.id.in_(kept_ids)).all():
            # Disconnect M2M to VehicleClass so the assoc rows are removed
            # before the VT itself; pre-existing classes themselves stay.
            old.vehicle_classes.clear()
            session.delete(old)
        session.flush()

        logger.info(
            "ConfigureVehicleTypes: created %d VT(s) %s; reassigned all rotations " "to %r%s.",
            len(new_vts),
            list(new_vts.keys()),
            default_vt.name,
            f"; {len(trip_to_vt)} trip override(s) applied" if trip_to_vt else "",
        )

    # ------------------------------------------------------------------ helpers

    def _resolve_per_vt(
        self,
        prefix: str,
        names: List[str],
        name_set: Set[str],
        params: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        per_vt: Dict[str, Dict[str, Any]] = {n: {} for n in names}
        for field, default in _FIELD_DEFAULTS.items():
            key = f"{prefix}.{field}"
            value = params.get(key, default)
            if isinstance(value, dict):
                if set(value.keys()) != name_set:
                    raise ValueError(
                        f"{key} dict keys {sorted(value.keys())} must equal "
                        f"vehicle_type_names {sorted(name_set)} exactly."
                    )
                for name in names:
                    per_vt[name][field] = value[name]
            else:
                for name in names:
                    per_vt[name][field] = value
        return per_vt

    @staticmethod
    def _unique_scenario_id(session: sqlalchemy.orm.Session) -> int:
        ids = [s.id for s in session.query(Scenario).all()]
        if len(ids) != 1:
            raise ValueError(f"Expected exactly one Scenario in the DB, got {len(ids)}.")
        return ids[0]

    def _build_vehicle_type(
        self,
        session: sqlalchemy.orm.Session,
        scenario_id: int,
        name: str,
        fields: Dict[str, Any],
    ) -> VehicleType:
        consumption_value = fields["consumption"]
        is_lut = isinstance(consumption_value, ConsumptionLut)

        vt_kwargs: Dict[str, Any] = {
            k: v for k, v in fields.items() if k != "consumption" and not isinstance(v, _Unset)
        }
        vt = VehicleType(
            scenario_id=scenario_id,
            name=name,
            consumption=None if is_lut else consumption_value,
            **vt_kwargs,
        )
        session.add(vt)
        if is_lut:
            # Attach the VehicleClass + ConsumptionLut before flushing so the
            # model's xor-check listener sees a class on first insert.
            self._attach_consumption_lut(
                session,
                scenario_id=scenario_id,
                vehicle_type=vt,
                lut_member=consumption_value,
            )
        session.flush()
        return vt

    def _attach_consumption_lut(
        self,
        session: sqlalchemy.orm.Session,
        scenario_id: int,
        vehicle_type: VehicleType,
        lut_member: ConsumptionLut,
    ) -> None:
        vehicle_class = VehicleClass(
            scenario_id=scenario_id,
            name=f"Consumption class for {vehicle_type.name}",
        )
        session.add(vehicle_class)
        vehicle_class.vehicle_types.append(vehicle_type)

        df = load_consumption_lut_df(lut_member)
        scenario = session.get(Scenario, scenario_id)
        lut = DbConsumptionLut.df_to_consumption_obj(
            df, scenario_or_id=scenario, vehicle_class_or_id=vehicle_class
        )
        lut.name = f"{lut_member.name} for {vehicle_type.name}"
        session.add(lut)

    def _apply_trip_overrides(
        self,
        session: sqlalchemy.orm.Session,
        trip_to_vt: Dict[int, str],
        new_vts: Dict[str, VehicleType],
        rotation_opp_charging: bool,
        scenario_id: int,
    ) -> None:
        for trip_id, vt_name in trip_to_vt.items():
            trip = session.get(Trip, trip_id)
            if trip is None:
                raise ValueError(f"trip_to_vehicle_type references unknown Trip.id {trip_id}")
            target_vt = new_vts[vt_name]
            old_rot = trip.rotation
            if len(old_rot.trips) > 1:
                new_rot = Rotation(
                    scenario_id=scenario_id,
                    vehicle_type=target_vt,
                    allow_opportunity_charging=rotation_opp_charging,
                    name=f"{old_rot.name} [trip {trip_id}]",
                )
                session.add(new_rot)
                trip.rotation = new_rot
            else:
                old_rot.vehicle_type = target_vt
