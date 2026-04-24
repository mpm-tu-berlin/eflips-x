"""
GTFS-specific utility modifiers for the eflips-x pipeline.

This module contains modifiers shared across GTFS-based flows.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import sqlalchemy.orm
from eflips.model import Rotation, Route, Trip, TripType, VehicleType

from eflips.x.framework import Modifier

logger = logging.getLogger(__name__)

# Defaults matching the original Potsdam/SWU flows
DEFAULT_BATTERY_CAPACITY = 360.0  # kWh
DEFAULT_CONSUMPTION = 1.5  # kWh/km
DEFAULT_CHARGING_CURVE = [[0.0, 450.0], [1.0, 450.0]]  # constant 450 kW

# Fields that are only applied if explicitly present in params. These map 1:1
# to attributes on eflips.model.VehicleType.
_OPTIONAL_VEHICLE_TYPE_FIELDS = (
    "name",
    "name_short",
    "battery_capacity_reserve",
    "charging_efficiency",
    "opportunity_charging_capable",
    "minimum_charging_power",
    "length",
    "width",
    "height",
    "empty_mass",
    "allowed_mass",
    "v2g_curve",
    "tco_parameters",
)


class ConfigureVehicleTypes(Modifier):
    """Parameterized modifier to configure vehicle types for GTFS-based flows.

    Sets battery capacity, energy consumption, opportunity charging capability,
    and charging curves for all vehicle types. Also enables opportunity charging
    for all rotations. Additional VehicleType attributes (name, name_short,
    dimensions, masses, etc.) can be overridden via optional params.
    """

    def __init__(self, code_version: str = "2", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            "ConfigureVehicleTypes.battery_capacity": "Battery capacity in kWh. Default: 360.0",
            "ConfigureVehicleTypes.consumption": "Energy consumption in kWh/km. Default: 1.5",
            "ConfigureVehicleTypes.charging_curve": (
                "Charging curve as list of [soc, power_kw] pairs. "
                "Default: [[0.0, 450.0], [1.0, 450.0]]"
            ),
            "ConfigureVehicleTypes.name": "Optional: overwrite VehicleType.name on every type.",
            "ConfigureVehicleTypes.name_short": (
                "Optional: overwrite VehicleType.name_short on every type. Useful for matching "
                "depot_config entries against a known short name."
            ),
            "ConfigureVehicleTypes.battery_capacity_reserve": (
                "Optional: battery capacity reserve below 0 kWh, in kWh."
            ),
            "ConfigureVehicleTypes.charging_efficiency": (
                "Optional: charging efficiency (0 < x <= 1)."
            ),
            "ConfigureVehicleTypes.opportunity_charging_capable": (
                "Optional: override opportunity-charging-capable flag. Defaults to True."
            ),
            "ConfigureVehicleTypes.minimum_charging_power": (
                "Optional: minimum charging power in kW."
            ),
            "ConfigureVehicleTypes.length": "Optional: vehicle length in meters.",
            "ConfigureVehicleTypes.width": "Optional: vehicle width in meters.",
            "ConfigureVehicleTypes.height": "Optional: vehicle height in meters.",
            "ConfigureVehicleTypes.empty_mass": "Optional: empty mass in kg.",
            "ConfigureVehicleTypes.allowed_mass": "Optional: allowed payload mass in kg.",
            "ConfigureVehicleTypes.v2g_curve": "Optional: vehicle-to-grid curve.",
            "ConfigureVehicleTypes.tco_parameters": "Optional: TCO parameters dict.",
        }

    def modify(self, session: sqlalchemy.orm.Session, params: Dict[str, Any]) -> None:
        battery_capacity: float = params.get(
            "ConfigureVehicleTypes.battery_capacity", DEFAULT_BATTERY_CAPACITY
        )
        consumption: float = params.get("ConfigureVehicleTypes.consumption", DEFAULT_CONSUMPTION)
        charging_curve: List[List[float]] = params.get(
            "ConfigureVehicleTypes.charging_curve", DEFAULT_CHARGING_CURVE
        )

        optional_overrides: Dict[str, Any] = {
            field: params[f"ConfigureVehicleTypes.{field}"]
            for field in _OPTIONAL_VEHICLE_TYPE_FIELDS
            if f"ConfigureVehicleTypes.{field}" in params
        }

        vehicle_types = session.query(VehicleType).all()
        if not vehicle_types:
            logger.warning("No vehicle types found in database!")
        for vt in vehicle_types:
            vt.battery_capacity = battery_capacity
            vt.consumption = consumption
            vt.opportunity_charging_capable = True
            vt.charging_curve = charging_curve
            for field, value in optional_overrides.items():
                setattr(vt, field, value)
            logger.info(
                f"Set vehicle type '{vt.name}': "
                f"battery_capacity={battery_capacity} kWh, "
                f"consumption={consumption} kWh/km"
            )

        session.query(Rotation).update({"allow_opportunity_charging": True})
        session.flush()


# Fields cloned from the base VehicleType unless overridden via params on the
# long-distance modifier. Mirrors the surface area of ConfigureVehicleTypes.
_LD_INHERITABLE_VEHICLE_TYPE_FIELDS = (
    "battery_type_id",
    "battery_capacity_reserve",
    "charging_efficiency",
    "opportunity_charging_capable",
    "minimum_charging_power",
    "length",
    "width",
    "height",
    "empty_mass",
    "allowed_mass",
    "v2g_curve",
    "tco_parameters",
    "energy_source",
)


class LongDistanceVehicleType(Modifier):
    """Create a long-distance VehicleType and reassign trips on long routes to it.

    Looks up the single VehicleType produced by ConfigureVehicleTypes (raises if
    there isn't exactly one), clones it with overrides for battery capacity,
    consumption, and charging curve, then walks every passenger trip whose Route
    is longer than ``long_distance_vehicle_threshold`` km. Affected trips are
    moved to a freshly created shadow Rotation per original rotation, bound to
    the new long-distance VehicleType. Original rotations keep their short
    trips; rotations that end up empty are deleted.

    Must run after ``ConfigureVehicleTypes`` and before ``VehicleScheduling`` —
    VehicleScheduling groups trips by ``Rotation.vehicle_type_id`` and would
    not see the new vehicle type otherwise.
    """

    def __init__(self, code_version: str = "v1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            "LongDistanceVehicleType.long_distance_vehicle_threshold": (
                "Route length threshold in km. Trips on routes longer than this "
                "are moved to the long-distance vehicle type. Default: 61.0"
            ),
            "LongDistanceVehicleType.name": (
                "Name of the new long-distance VehicleType. Default: "
                "'<base_name> Long Distance'"
            ),
            "LongDistanceVehicleType.name_short": (
                "Short name of the new long-distance VehicleType. Default: "
                "'<base_name_short>_LD'"
            ),
            "LongDistanceVehicleType.battery_capacity": (
                "Battery capacity in kWh. Default: 500.0"
            ),
            "LongDistanceVehicleType.consumption": ("Energy consumption in kWh/km. Default: 1.2"),
            "LongDistanceVehicleType.charging_curve": (
                "Charging curve as list of [soc, power_kw] pairs. "
                "Default: [[0.0, 450.0], [1.0, 450.0]]"
            ),
        }

    def modify(self, session: sqlalchemy.orm.Session, params: Dict[str, Any]) -> None:
        threshold_km: float = params.get(
            "LongDistanceVehicleType.long_distance_vehicle_threshold", 61.0
        )
        battery_capacity: float = params.get("LongDistanceVehicleType.battery_capacity", 500.0)
        consumption: float = params.get("LongDistanceVehicleType.consumption", 1.2)
        charging_curve: List[List[float]] = params.get(
            "LongDistanceVehicleType.charging_curve", DEFAULT_CHARGING_CURVE
        )

        if threshold_km <= 0:
            raise ValueError(f"long_distance_vehicle_threshold must be > 0, got {threshold_km}")
        if battery_capacity <= 0:
            raise ValueError(f"battery_capacity must be > 0, got {battery_capacity}")
        if consumption <= 0:
            raise ValueError(f"consumption must be > 0, got {consumption}")

        base_vt = session.query(VehicleType).one()

        new_vt = VehicleType(
            scenario_id=base_vt.scenario_id,
            name=params.get("LongDistanceVehicleType.name", f"{base_vt.name} Long Distance"),
            name_short=params.get(
                "LongDistanceVehicleType.name_short",
                f"{base_vt.name_short}_LD" if base_vt.name_short else None,
            ),
            battery_capacity=battery_capacity,
            consumption=consumption,
            charging_curve=charging_curve,
        )
        for field in _LD_INHERITABLE_VEHICLE_TYPE_FIELDS:
            setattr(new_vt, field, getattr(base_vt, field))
        session.add(new_vt)
        session.flush()
        logger.info(
            f"Created long-distance VehicleType '{new_vt.name}' "
            f"(battery {battery_capacity} kWh, consumption {consumption} kWh/km)"
        )

        threshold_m = threshold_km * 1000.0
        long_trips: List[Trip] = (
            session.query(Trip)
            .join(Route, Trip.route_id == Route.id)
            .filter(Route.distance > threshold_m)
            .filter(Trip.trip_type == TripType.PASSENGER)
            .all()
        )

        if not long_trips:
            logger.info(
                f"No passenger trips on routes longer than {threshold_km} km — "
                f"no rotations reassigned."
            )
            return

        trips_by_rotation: Dict[int, List[Trip]] = defaultdict(list)
        for trip in long_trips:
            trips_by_rotation[trip.rotation_id].append(trip)

        affected_route_ids = {t.route_id for t in long_trips}

        for original_rotation_id, trips in trips_by_rotation.items():
            original = session.query(Rotation).filter(Rotation.id == original_rotation_id).one()
            shadow = Rotation(
                scenario_id=original.scenario_id,
                vehicle_type_id=new_vt.id,
                allow_opportunity_charging=original.allow_opportunity_charging,
                name=f"{original.name} [LD]" if original.name else None,
            )
            session.add(shadow)
            session.flush()
            for trip in trips:
                trip.rotation_id = shadow.id

        session.flush()

        # Delete any original rotations that are now empty (all their trips were long).
        emptied = (
            session.query(Rotation)
            .filter(Rotation.id.in_(trips_by_rotation.keys()))
            .filter(~Rotation.trips.any())
            .all()
        )
        for r in emptied:
            session.delete(r)
        if emptied:
            logger.info(f"Deleted {len(emptied)} now-empty original rotation(s).")

        session.flush()

        logger.info(
            f"Created {len(trips_by_rotation)} long-distance rotations and reassigned "
            f"{len(long_trips)} trips on {len(affected_route_ids)} routes "
            f"(>{threshold_km} km)."
        )
