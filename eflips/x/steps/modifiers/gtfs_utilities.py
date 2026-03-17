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


class ConfigureVehicleTypes(Modifier):
    """Parameterized modifier to configure vehicle types for GTFS-based flows.

    Sets battery capacity, energy consumption, opportunity charging capability,
    and charging curves for all vehicle types. Also enables opportunity charging
    for all rotations.
    """

    def __init__(self, code_version: str = "1", **kwargs: Any) -> None:
        super().__init__(code_version=code_version, **kwargs)

    @classmethod
    def document_params(cls) -> Dict[str, str]:
        return {
            "ConfigureVehicleTypes.battery_capacity": ("Battery capacity in kWh. Default: 360.0"),
            "ConfigureVehicleTypes.consumption": ("Energy consumption in kWh/km. Default: 1.5"),
            "ConfigureVehicleTypes.charging_curve": (
                "Charging curve as list of [soc, power_kw] pairs. "
                "Default: [[0.0, 450.0], [1.0, 450.0]]"
            ),
        }

    def modify(self, session: sqlalchemy.orm.Session, params: Dict[str, Any]) -> None:
        battery_capacity: float = params.get(
            "ConfigureVehicleTypes.battery_capacity", DEFAULT_BATTERY_CAPACITY
        )
        consumption: float = params.get("ConfigureVehicleTypes.consumption", DEFAULT_CONSUMPTION)
        charging_curve: List[List[float]] = params.get(
            "ConfigureVehicleTypes.charging_curve", DEFAULT_CHARGING_CURVE
        )

        vehicle_types = session.query(VehicleType).all()
        if not vehicle_types:
            logger.warning("No vehicle types found in database!")
        for vt in vehicle_types:
            vt.battery_capacity = battery_capacity
            vt.consumption = consumption
            vt.opportunity_charging_capable = True
            vt.charging_curve = charging_curve
            logger.info(
                f"Set vehicle type '{vt.name}': "
                f"battery_capacity={battery_capacity} kWh, "
                f"consumption={consumption} kWh/km"
            )

        session.query(Rotation).update({"allow_opportunity_charging": True})
        session.flush()
