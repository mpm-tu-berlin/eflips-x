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
from eflips.model import Rotation, VehicleType, Station, Route, AssocRouteStation, StopTime
from fuzzywuzzy import fuzz
from geoalchemy2.shape import to_shape
from sqlalchemy import not_, func
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


class RemoveUnusedRotations(Modifier):
    """
    Remove unused rotations from a just-imported BVGXML dataset.

    A just-imported BVGXML dataset contains some dummy data that does not seem to refer to actual
    operations. This modifier removes all rotations that do not start and end at stations from a
    specified list of depot/garage stations.

    The modifier performs the following operations:
    1. Identifies depot stations based on their short names
    2. Removes all rotations that do not start and end at the same depot station
    3. Removes all rotations that do not start at one of the specified depot stations
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_depot_short_names() -> List[str]:
        """Get the default list of depot station short names to keep."""
        return [
            "BTRB",
            "BF B",
            "BF C",
            "BFI",
            "BF I",
            "BHKO",
            "BHLI",
            "BF L",
            "BHMA",
            "BF M",
            "BF S",
            "BF MDA",
        ]

    def document_params(self) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary documenting the parameters of this modifier.
        """
        return {
            f"{self.__class__.__name__}.depot_station_short_names": """
List of station short names that represent depots/garages where rotations should start and end.
Only rotations that start and end at the same station from this list will be kept.

**Format:** `List[str]` containing station short names.

**Default:**
```python
[
    "BTRB", "BF B", "BF C", "BFI", "BF I", "BHKO",
    "BHLI", "BF L", "BHMA", "BF M", "BF S", "BF MDA"
]
```

**Note:** The default values are specific to BVG (Berlin) depots.
            """.strip(),
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Modify the database by removing rotations that don't start/end at specified depots.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters containing optional depot_station_short_names

        Returns:
        --------
        None
            This modifier modifies the database in place and doesn't return a specific result
        """
        # Make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")
        scenario_id = scenarios[0].id

        # Get parameters with defaults
        param_key = f"{self.__class__.__name__}.depot_station_short_names"
        depot_short_names: List[str] = params.get(param_key, self._get_default_depot_short_names())

        # Emit warning if using defaults
        if param_key not in params:
            warnings.warn(
                f"Parameter '{param_key}' not provided, using default BVG depot station names",
                UserWarning,
            )

        # Validation: Check that the list is not empty
        if not depot_short_names:
            raise ValueError("depot_station_short_names cannot be empty")

        # Get the station IDs for the depots we want to keep
        station_ids_to_keep = (
            session.query(Station.id)
            .filter(
                Station.scenario_id == scenario_id,
                Station.name_short.in_(depot_short_names),
            )
            .all()
        )
        station_ids_to_keep = [station_id for station_id, in station_ids_to_keep]

        if not station_ids_to_keep:
            self.logger.warning(
                f"No stations found with short names: {depot_short_names}. "
                "No rotations will be kept."
            )

        self.logger.info(
            f"Keeping rotations that start and end at {len(station_ids_to_keep)} depot stations"
        )

        # Process all rotations and remove those that don't meet the criteria
        all_rotations = session.query(Rotation).filter(Rotation.scenario_id == scenario_id).all()
        rotations_removed = 0

        for rotation in all_rotations:
            if not rotation.trips:
                self.logger.warning(f"Rotation {rotation.name} has no trips, removing it")
                session.delete(rotation)
                rotations_removed += 1
                continue

            first_station_id = rotation.trips[0].route.departure_station_id
            last_station_id = rotation.trips[-1].route.arrival_station_id

            # Remove rotation if it doesn't start and end at the same depot station
            # or if the depot is not in our keep list
            if first_station_id != last_station_id or first_station_id not in station_ids_to_keep:
                self.logger.debug(
                    f"Removing rotation {rotation.name}, "
                    f"vehicle type {rotation.vehicle_type.name_short}, "
                    f"start {rotation.trips[0].route.departure_station.name}, "
                    f"end {rotation.trips[-1].route.arrival_station.name}"
                )
                for trip in rotation.trips:
                    for stop_time in trip.stop_times:
                        session.delete(stop_time)
                    session.delete(trip)
                session.delete(rotation)
                rotations_removed += 1

        self.logger.info(f"Removed {rotations_removed} unused rotations")

        return None


class MergeStations(Modifier):
    """
    Merge nearby stations in a BVGXML dataset.

    A BVGXML dataset contains quite some stations that are different Station objects in the database,
    but from an electrification perspective refer to the same thing. Especially when doing vehicle
    scheduling and not allowing "deadheading" between nearby stations, this can lead to problems.
    This modifier merges stations that are:
    1. Within a specified distance of each other (geospatial proximity)
    2. Have similar names (fuzzy name matching above a threshold)

    The modifier performs the following operations:
    1. Identifies stations that are used in routes (departure or arrival stations)
    2. Finds nearby stations within max_distance_meters
    3. Applies fuzzy name matching to verify they refer to the same location
    4. Groups stations to merge together
    5. Keeps the station with the shortest name
    6. Updates all references (routes, trips, stoptimes) to point to the kept station
    7. Deletes the merged stations
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_max_distance() -> float:
        """Get the default maximum distance in meters for considering stations as nearby."""
        return 100.0

    @staticmethod
    def _get_default_match_percentage() -> float:
        """Get the default fuzzy name matching percentage threshold."""
        return 80.0

    def document_params(self) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary documenting the parameters of this modifier.
        """
        return {
            f"{self.__class__.__name__}.max_distance_meters": """
Maximum distance in meters between two stations to consider them as potentially the same station.
Stations beyond this distance will never be merged, regardless of name similarity.

**Format:** `float` (distance in meters)

**Default:** 100.0

**Note:** This uses geospatial distance calculation (ST_Distance).
            """.strip(),
            f"{self.__class__.__name__}.match_percentage": """
Minimum fuzzy name matching percentage required to merge two nearby stations.
Uses the Levenshtein distance ratio (0-100) to compare station names.

**Format:** `float` (percentage 0-100)

**Default:** 80.0

**Example:** "Berlin Hauptbahnhof" and "S+U Berlin Hauptbahnhof" would have a high match percentage.
            """.strip(),
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Modify the database by merging nearby stations with similar names.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters containing optional max_distance_meters and match_percentage

        Returns:
        --------
        None
            This modifier modifies the database in place and doesn't return a specific result
        """
        # Make sure there is just one scenario
        scenarios = session.query(eflips.model.Scenario).all()
        if len(scenarios) != 1:
            raise ValueError(f"Expected exactly one scenario, found {len(scenarios)}")

        # Get parameters with defaults
        param_key_distance = f"{self.__class__.__name__}.max_distance_meters"
        param_key_match = f"{self.__class__.__name__}.match_percentage"

        max_distance_meters: float = params.get(
            param_key_distance, self._get_default_max_distance()
        )
        match_percentage: float = params.get(param_key_match, self._get_default_match_percentage())

        # Validation
        if max_distance_meters <= 0:
            raise ValueError(f"max_distance_meters must be positive, got {max_distance_meters}")
        if not 0 <= match_percentage <= 100:
            raise ValueError(f"match_percentage must be between 0 and 100, got {match_percentage}")

        # Identify all the station IDs where routes start or end
        start_station_ids = session.query(Route.departure_station_id).distinct().all()
        end_station_ids = session.query(Route.arrival_station_id).distinct().all()
        station_ids_in_use = set(
            [station_id for station_id, in start_station_ids]
            + [station_id for station_id, in end_station_ids]
        )

        self.logger.info(f"Found {len(station_ids_in_use)} stations in use by routes")

        to_merge: List[List[Station]] = []
        stations_in_use = session.query(Station).filter(Station.id.in_(station_ids_in_use)).all()

        for station in stations_in_use:
            geom_wkb = to_shape(station.geom).wkb

            # Do a fancy geospatial query to find all stations within the given distance
            nearby_stations = (
                session.query(Station)
                .filter(Station.id != station.id)
                .filter(Station.id.in_(station_ids_in_use))
                .filter(
                    func.ST_Distance(Station.geom, func.ST_GeomFromWKB(geom_wkb), 1)
                    <= max_distance_meters
                )
                .all()
            )

            if len(nearby_stations) > 0:
                # Also check if they're named similarly using fuzzy matching
                for nearby_station in nearby_stations:
                    orig_station_name = station.name
                    nearby_station_name = nearby_station.name
                    percentage = fuzz.ratio(orig_station_name, nearby_station_name)
                    if percentage >= match_percentage:
                        # See if one of the stations is already in the to_merge list
                        found = False
                        for merge_group in to_merge:
                            if station in merge_group or nearby_station in merge_group:
                                if station not in merge_group:
                                    merge_group.append(station)
                                if nearby_station not in merge_group:
                                    merge_group.append(nearby_station)
                                found = True
                                break
                        if not found:
                            to_merge.append([station, nearby_station])
                        self.logger.debug(
                            f"Found nearby stations to merge: {station.name} and "
                            f"{nearby_station.name} ({percentage}%)"
                        )

        self.logger.info(f"Found {len(to_merge)} groups of stations to merge")

        stations_merged = 0
        for merge_group in to_merge:
            # Log the stations to merge
            self.logger.debug(f"Merging stations: {[station.name for station in merge_group]}")
            # Pick the station with the shortest name as the one to keep
            station_to_keep = min(merge_group, key=lambda s: len(s.name))
            self.logger.debug(f"Keeping station: {station_to_keep.name}")
            stations_to_remove = [s for s in merge_group if s != station_to_keep]

            for other_station in stations_to_remove:
                other_station_geom = other_station.geom
                with session.no_autoflush:
                    # Update all routes, trips, and stoptimes containing the station
                    # to point to the kept station instead
                    session.query(Route).filter(
                        Route.departure_station_id == other_station.id
                    ).update({"departure_station_id": station_to_keep.id})

                    session.query(Route).filter(
                        Route.arrival_station_id == other_station.id
                    ).update({"arrival_station_id": station_to_keep.id})

                    session.query(AssocRouteStation).filter(
                        AssocRouteStation.station_id == other_station.id
                    ).update({"station_id": station_to_keep.id, "location": other_station_geom})

                    session.query(StopTime).filter(StopTime.station_id == other_station.id).update(
                        {"station_id": station_to_keep.id}
                    )

                session.flush()
                session.delete(other_station)
                stations_merged += 1

        self.logger.info(f"Merged {stations_merged} stations into {len(to_merge)} groups")

        return None
