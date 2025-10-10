"""
BVG-specific tools for modifying imported BVGXML datasets.

This module contains modifiers that are specific to the BVG (Berlin public transport) dataset,
such as removing unused vehicle types and rotations, and replacing them with standardized
electric vehicle types.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List

import eflips.model
from eflips.model import Rotation, VehicleType
from sqlalchemy import not_
from sqlalchemy.orm import Session

from eflips.x.framework import Modifier


class RemoveUnusedVehicleTypes(Modifier):
    """
    Remove unused vehicle types from a just-imported BVGXML dataset.

    A just-imported BVGXML dataset contains some dummy data that does not seem to refer to actual
    operations. This modifier removes all vehicle types and rotations that are not actually used
    in the dataset. It also creates generic electric vehicle types for all vehicles we want to keep.

    The modifier performs the following operations:
    1. Removes all rotations with vehicle types not in the conversion mapping
    2. Creates new standardized electric vehicle types
    3. Maps old vehicle types to new standardized types
    4. Deletes old vehicle types
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_vehicle_types(scenario_id: int) -> List[VehicleType]:
        """Get the default BVG vehicle types."""
        return [
            VehicleType(
                name="Ebusco 3.0 12 large battery",
                scenario_id=scenario_id,
                name_short="EN",
                battery_capacity=500.0,
                battery_capacity_reserve=0.0,
                charging_curve=[[0, 450], [1, 450]],
                opportunity_charging_capable=True,
                minimum_charging_power=10,
                length=12.0,
                width=2.55,
                height=3.19,
                empty_mass=12000,
                allowed_mass=12000 + (70 * 68),  # 70 passengers, 68 kg each
                consumption=None,
            ),
            VehicleType(
                name="Solaris Urbino 18 large battery",
                scenario_id=scenario_id,
                name_short="GN",
                battery_capacity=640.0,
                battery_capacity_reserve=0.0,
                charging_curve=[[0, 450], [1, 450]],
                opportunity_charging_capable=True,
                minimum_charging_power=10,
                length=18.0,
                width=2.55,
                height=3.19,
                empty_mass=19000,
                allowed_mass=19000 + (100 * 68),  # 100 passengers, 68 kg each
                consumption=None,
            ),
            VehicleType(
                name="Alexander Dennis Enviro500EV large battery",
                scenario_id=scenario_id,
                name_short="DD",
                battery_capacity=472,
                battery_capacity_reserve=0,
                charging_curve=[[0, 450], [1, 450]],
                opportunity_charging_capable=True,
                minimum_charging_power=10,
                length=12.0,
                width=2.55,
                height=4.3,
                empty_mass=19000,
                allowed_mass=19000 + (112 * 68),  # 112 passengers, 68 kg each
                consumption=None,
            ),
        ]

    @staticmethod
    def _get_default_conversion_mapping() -> Dict[str, List[str]]:
        """Get the default vehicle type conversion mapping."""
        return {
            "GN": ["GN", "GEG", "GEG-200"],
            "EN": ["EED-120", "EED-160", "EED-320", "EN", "MN"],
            "DD": ["DL", "D"],
        }

    def document_params(self) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary documenting the parameters of this modifier.
        """
        return {
            f"{self.__class__.__name__}.new_vehicle_types": """
List of `VehicleType` instances to create in the database. Each instance should have a unique
`name_short` property that doesn't conflict with existing vehicle types in the database.

**Default:** Three vehicle types (EN: single decker, GN: articulated bus, DD: double decker)
with BVG-specific configurations.
            """.strip(),
            f"{self.__class__.__name__}.vehicle_type_conversion": """
Dictionary mapping new vehicle type short names (keys) to lists of old vehicle type short names
(values). The keys must match the `name_short` values of the new vehicle types. The values must
cover all existing vehicle types in the database that should be kept.

**Format:** `Dict[str, List[str]]` where keys are new type short names and values are lists of
old type short names to convert.

**Default:**
```python
{
    "GN": ["GN", "GEG", "GEG-200"],
    "EN": ["EED-120", "EED-160", "EED-320", "EN", "MN"],
    "DD": ["DL", "D"]
}
```
            """.strip(),
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> Path:
        """
        Modify the database by removing unused vehicle types and creating standardized ones.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters containing optional new_vehicle_types and vehicle_type_conversion

        Returns:
        --------
        Path
            Path to the modified database (returned via context)
        """
        # Make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")
        scenario_id = scenarios[0].id

        # Get existing vehicle types in the database
        existing_vehicle_types = (
            session.query(VehicleType).filter(VehicleType.scenario_id == scenario_id).all()
        )
        existing_short_names = {vt.name_short for vt in existing_vehicle_types}

        # Get parameters with defaults
        param_key_new = f"{self.__class__.__name__}.new_vehicle_types"
        param_key_conversion = f"{self.__class__.__name__}.vehicle_type_conversion"

        new_vehicle_types: List[VehicleType] = params.get(
            param_key_new, self._get_default_vehicle_types(scenario_id)
        )
        vehicle_type_conversion: Dict[str, List[str]] = params.get(
            param_key_conversion, self._get_default_conversion_mapping()
        )

        # Emit warning if using defaults
        if param_key_new not in params:
            warnings.warn(
                f"Parameter '{param_key_new}' not provided, using default BVG vehicle types",
                UserWarning,
            )
        if param_key_conversion not in params:
            warnings.warn(
                f"Parameter '{param_key_conversion}' not provided, using default BVG conversion mapping",
                UserWarning,
            )

        # Validation: Check that new vehicle type short names are unique
        new_short_names = [vt.name_short for vt in new_vehicle_types]
        if len(new_short_names) != len(set(new_short_names)):
            raise ValueError(
                f"New vehicle types have duplicate name_short values: {new_short_names}"
            )

        # Validation: Check that new vehicle type short names don't conflict with existing ones
        # (excluding the ones we're about to convert)
        all_old_names_to_convert = set()
        for old_names in vehicle_type_conversion.values():
            all_old_names_to_convert.update(old_names)

        conflicting_names = set(new_short_names) & (
            existing_short_names - all_old_names_to_convert
        )
        if conflicting_names:
            raise ValueError(
                f"New vehicle type short names conflict with existing types that are not being converted: {conflicting_names}"
            )

        # Validation: Check that conversion mapping keys match new vehicle type short names
        conversion_keys = set(vehicle_type_conversion.keys())
        new_names_set = set(new_short_names)
        if conversion_keys != new_names_set:
            missing_in_conversion = new_names_set - conversion_keys
            extra_in_conversion = conversion_keys - new_names_set
            error_msg = "Mismatch between new vehicle types and conversion mapping keys."
            if missing_in_conversion:
                error_msg += f" Missing in conversion: {missing_in_conversion}."
            if extra_in_conversion:
                error_msg += f" Extra in conversion: {extra_in_conversion}."
            raise ValueError(error_msg)

        # Validation: Check that conversion mapping values cover all existing vehicle types
        # that we want to keep (i.e., they should be in the database)
        for old_names in vehicle_type_conversion.values():
            for old_name in old_names:
                if old_name not in existing_short_names:
                    self.logger.warning(
                        f"Vehicle type '{old_name}' in conversion mapping does not exist in database"
                    )

        # Collect all vehicle types we want to keep (those in the conversion mapping)
        vehicle_types_to_keep = set()
        for old_names in vehicle_type_conversion.values():
            vehicle_types_to_keep.update(old_names)

        # Remove all rotations with vehicle types not in the keep list
        rotations_to_remove = (
            session.query(Rotation)
            .join(VehicleType)
            .filter(not_(VehicleType.name_short.in_(vehicle_types_to_keep)))
            .all()
        )

        for rotation in rotations_to_remove:
            self.logger.debug(
                f"Removing rotation {rotation.name}, vehicle type {rotation.vehicle_type.name_short}, "
                f"start {rotation.trips[0].route.departure_station.name}, "
                f"end {rotation.trips[-1].route.arrival_station.name}"
            )
            for trip in rotation.trips:
                for stop_time in trip.stop_times:
                    session.delete(stop_time)
                session.delete(trip)
            session.delete(rotation)
        session.flush()

        # Set scenario_id for all new vehicle types and add them to the session
        created_vehicle_types = []
        for vt in new_vehicle_types:
            vt.scenario_id = scenario_id
            session.add(vt)
            created_vehicle_types.append(vt)
        session.flush()

        # Build a mapping from vehicle type instances to their short names for lookup
        new_vt_by_short_name = {vt.name_short: vt for vt in created_vehicle_types}

        # Map old vehicle types to new standardized types
        for new_short_name, old_short_names in vehicle_type_conversion.items():
            new_vehicle_type = new_vt_by_short_name[new_short_name]
            old_rotations = (
                session.query(Rotation)
                .filter(Rotation.scenario_id == scenario_id)
                .join(VehicleType)
                .filter(VehicleType.name_short.in_(old_short_names))
                .all()
            )
            for rotation in old_rotations:
                rotation.vehicle_type = new_vehicle_type

        # Delete the old vehicle types
        new_vt_ids = {vt.id for vt in created_vehicle_types}
        old_vehicle_types = (
            session.query(VehicleType)
            .filter(VehicleType.scenario_id == scenario_id)
            .filter(not_(VehicleType.id.in_(new_vt_ids)))
            .all()
        )
        for vehicle_type in old_vehicle_types:
            session.delete(vehicle_type)

        self.logger.info(
            f"Removed unused vehicle types and created {len(created_vehicle_types)} standardized types"
        )

        return None  # Modifier doesn't return a specific result, just modifies the database
