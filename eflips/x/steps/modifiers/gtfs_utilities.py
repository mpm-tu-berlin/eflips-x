"""
GTFS-specific utility modifiers for the eflips-x pipeline.

This module contains modifiers shared across GTFS-based flows.
"""

import logging
from typing import Any, Dict, List

import sqlalchemy.orm
from eflips.model import VehicleType, Rotation

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
