"""
BVG-specific tools for modifying imported BVGXML datasets.

This module contains modifiers that are specific to the BVG (Berlin public transport) dataset,
such as removing unused vehicle types and rotations, and replacing them with standardized
electric vehicle types.
"""

import logging
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import eflips.model
import numpy as np
import pandas as pd
import sqlalchemy
from eflips.model import (
    Rotation,
    VehicleType,
    Station,
    Route,
    AssocRouteStation,
    StopTime,
    Trip,
    VehicleClass,
    ConsumptionLut,
    Scenario,
)
from fuzzywuzzy import fuzz  # type: ignore[import-untyped]
from geoalchemy2.shape import to_shape
from sqlalchemy import not_, func
from sqlalchemy.orm import Session, joinedload

from eflips.x.framework import Modifier


def depots_for_bvg(
    session: sqlalchemy.orm.session.Session,
) -> List[Dict[str, Union[int, Tuple[float, float], List[int], str]]]:
    """
    Get depot configuration for BVG (Berlin) scenarios.

    This function returns a list of depot configurations specific to the BVG (Berlin public
    transport company), including existing depot stations and planned new depots with their
    capacities and allowed vehicle types.

    Parameters:
    -----------
    scenario : Scenario
        The scenario to get depot information for

    Returns:
    --------
    List[Dict[str, Union[int, Tuple[float, float], List[int]]]]
        List of depot configurations. Each dictionary contains:
        - "depot_station": Either a station ID (int) or (lon, lat) tuple for new depots
        - "capacity": Depot capacity in 12m bus equivalents
        - "vehicle_type": List of vehicle type IDs allowed at this depot
        - "name": Depot name (only for new depots that don't exist in database)

    Notes:
    ------
    The depot capacities and vehicle type restrictions are based on BVG planning data:
    - "Abstellfläche Mariendorf": No charging infrastructure (capacity 0)
    - "Betriebshof Spandau": 240 capacity, all vehicle types (EN, GN, DD)
    - "Betriebshof Indira-Gandhi-Straße": 320 capacity, all vehicle types
    - "Betriebshof Britz": 160 capacity, all vehicle types
    - "Betriebshof Cicerostraße": 229 capacity, all vehicle types
    - "Betriebshof Müllerstraße": 175 capacity, all vehicle types
    - "Betriebshof Lichtenberg": 140 capacity, only articulated buses (GN)
    - "Betriebshof Köpenicker Landstraße": 220 capacity, EN and GN (new depot)
    - "Betriebshof Rummelsburger Landstraße": 80 capacity, only GN (new depot)
    - "Betriebshof Säntisstraße": 250 capacity, EN and GN (new depot)
    - "Betriebshof Alt Friedrichsfelde": No charging infrastructure (capacity 0)
    """
    # Put the new capacities into a variable
    #
    # - "Abstellfläche Mariendorf" will not be equipped with charging infrastructure, therefore it cannot serve as a depot for electrified buses
    # - There will be a new depot "Köpenicker Landstraße" at the coordinates 52.4654085,13.4964867 with a capacity of 200 12m buses
    # - There will be a new depot "Rummelsburger Landstraße" at the coordinates "52.4714167,13.5053889" with a capacity of 60 12m buses
    # - There will be a new depot "Säntisstraße" at the coordinates "52.416735,13.3844563" with a capacity of 230 12m buses
    # - The capacity of the existing depot "Spandau" will be 220 12m buses
    # - The capacity of the existing depot "Indira-Gandhi-Straße" will be 300 12m buses
    # - The capacity of the existing depot "Britz" weill be 140 12m buses
    # - The capacity of the existing depot "Cicerostraße" will be 209 12m buses
    # - The capacity of the existing depot "Müllerstraße" will be 155 12m buses
    # - The capacity of the existing depot "Lichtenberg" will be 120 12m buses
    # - "Alt Friedrichsfelde" will not be equipped with charging infrastructure, therefore it cannot serve as a depot for electrified buses
    #
    # Allowed vehicle types are also specified for each depot. The are following vehicle types in total:
    # - "EN" for 12m electric buses
    # - "GN" for 18m articulated buses
    # - "DD" for 12m double-decker buses
    #
    # And the vehicle types that can be used at each depot are as follows:
    # - "Abstellfläche Mariendorf": None
    # - "Betriebshof Spandau": EN, GN, DD
    # - "Betriebshof Indira-Gandhi-Straße": EN, GN, DD
    # - "Betriebshof Britz": EN, GN, DD
    # - "Betriebshof Cicerostraße": EN, GN, DD
    # - "Betriebshof Müllerstraße": EN, GN, DD
    # - "Betriebshof Lichtenberg": GN
    # - "Betriebshof Köpenicker Landstraße": EN, GN
    # - "Betriebshof Rummelsburger Landstraße": GN
    # - "Betriebshof Säntisstraße": EN, GN
    # - "Betriebshof Alt Friedrichsfelde": None
    #
    # The new capacities should be specified as a dictionary containing the following keys:
    # - "depot_station": Either the ID of the existing station or a (lon, lat) tuple for a depot that does not yet exist in the database
    # - "capacity": The new capacity of the depot, in 12m buses
    # - "vehicle_type": A list of vehicle type ids that can be used at this depot
    # - "name": The name of the depot (only for new depots)

    assert session.query(Scenario).count() == 1, "Expected exactly one scenario"

    scenario = session.query(Scenario).one()

    depot_list: List[Dict[str, Union[int, Tuple[float, float], List[int], str]]] = []
    all_vehicle_type_id_query = (
        session.query(VehicleType).filter(VehicleType.scenario == scenario).all()
    )
    all_vehicle_type_ids = [x.id for x in all_vehicle_type_id_query]

    # "Abstellfläche Mariendorf" will have a capacity of zero
    station_id_query = (
        session.query(Station)
        .filter(Station.name_short == "BF MDA")
        .filter(Station.scenario == scenario)
        .one()
    )
    station_id: int = station_id_query.id

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 0,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Spandau will hava a capacity of 220
    station_id_query = (
        session.query(Station)
        .filter(Station.name_short == "BF S")
        .filter(Station.scenario == scenario)
        .one()
    )
    station_id = station_id_query.id
    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 240,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Indira-Gandhi-Straße" will have a capacity of 300
    station_id_query = (
        session.query(Station)
        .filter(Station.name_short == "BFI")
        .filter(Station.scenario == scenario)
        .one()
    )
    station_id = station_id_query.id

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 320,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Britz" will have a capacity of 140
    station_id_query = (
        session.query(Station)
        .filter(Station.name_short == "BTRB")
        .filter(Station.scenario == scenario)
        .one()
    )
    station_id = station_id_query.id

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 160,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Cicerostraße" will have a capacity of 209
    station_id_query = (
        session.query(Station)
        .filter(Station.name_short == "BF C")
        .filter(Station.scenario == scenario)
        .one()
    )
    station_id = station_id_query.id

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 229,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Müllerstraße" will have a capacity of 155
    station_id_query = (
        session.query(Station)
        .filter(Station.name_short == "BF M")
        .filter(Station.scenario == scenario)
        .one()
    )
    station_id = station_id_query.id

    vehicle_types = ["EN", "GN", "DD"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]

    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 175,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Lichtenberg" will have a capacity of 120
    station_id_query = (
        session.query(Station)
        .filter(Station.name_short == "BHLI")
        .filter(Station.scenario == scenario)
        .one()
    )
    station_id = station_id_query.id

    vehicle_types = ["GN"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]
    depot_list.append(
        {
            "depot_station": station_id,
            "capacity": 140,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Köpenicker Landstraße" will have a capacity of 200

    vehicle_types = ["EN", "GN"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]
    depot_list.append(
        {
            "depot_station": (13.4964867, 52.4654085),
            "name": "Betriebshof Köpenicker Landstraße",
            "capacity": 220,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Rummelsburger Landstraße" will have a capacity of 60
    vehicle_types = ["GN"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]
    depot_list.append(
        {
            "depot_station": (13.5053889, 52.4714167),
            "name": "Betriebshof Rummelsburger Landstraße",
            "capacity": 80,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Säntisstraße" will have a capacity of 230
    vehicle_types = ["EN", "GN"]
    vehicle_type_query = (
        session.query(VehicleType)
        .filter(VehicleType.name_short.in_(vehicle_types))
        .filter(VehicleType.scenario == scenario)
        .all()
    )
    vehicle_type_ids = [x.id for x in vehicle_type_query]
    depot_list.append(
        {
            "depot_station": (13.3844563, 52.416735),
            "name": "Betriebshof Säntisstraße",
            "capacity": 250,
            "vehicle_type": vehicle_type_ids,
        }
    )

    # "Betriebshof Alt Friedrichsfelde" will have a capacity of 0
    depot_list.append(
        {
            "depot_station": (13.5401389, 52.5123056),
            "name": "Betriebshof Alt Friedrichsfelde",
            "capacity": 0,
            "vehicle_type": all_vehicle_type_ids,
        }
    )

    return depot_list


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
            f"{self.__class__.__name__}.override_consumption_lut": """
This should be a Dict[str, float | Path], with the key being the short name of the (new)
vehicle type to override. If the value is a float, a default LUT based on Ji(2022) will 
be created using eflips-model and it's consumption will be scaled by multiplying with the
given float. If the value is a Path, it should point to an Excel file (see
`data/input/consumption_lut_gn.xlsx` for an example) that contains the consumption LUT
for the vehicle type.
""".strip(),
        }

    def add_consumption_lut_for_vehicle_type(
        self,
        session: Session,
        vehicle_type: VehicleType,
        multiplier: float = 1.0,
        path: Optional[Path] = None,
    ) -> None:
        """
        This method creates the correpsonding vehicle class and consumption LUT for the given vehicle type.
        If a path is given, it will load the consumption LUT from the given Excel file.

        :param session: An open SQLAlchemy session
        :param vehicle_type: The vehicle type to create the consumption LUT for
        :param multiplier: an optional multiplier to scale the consumption LUT. Default is 1.0 (no scaling)
        :param path: an optional path to an Excel file containing the consumption LUT. if set this will override the multiplier.

        :return: None. The LUT is added to the database via the session.
        """
        logger = logging.getLogger(__name__)

        # Create a vehicle class for the vehicle type
        vehicle_class = VehicleClass(
            scenario_id=vehicle_type.scenario_id,
            name=f"Consumption LUT for {vehicle_type.name_short}",
            vehicle_types=[vehicle_type],
        )
        session.add(vehicle_class)
        session.flush()  # To get assign the IDs

        # Create a LUT for the vehicle class
        # It may be scaked or updated below.
        consumption_lut = ConsumptionLut.from_vehicle_type(vehicle_type, vehicle_class)
        session.add(consumption_lut)

        if path is not None:
            with open(path, "rb") as f:
                consumption_lut_file = pd.read_excel(f)

            # The LUT is a 2D table. The first column is the average speed.
            # The first row contains the temperatures.
            # Turn it into a multi-indexed dataframe
            emp_temperatures = np.array(consumption_lut_file.columns[1:]).astype(np.float64)
            emp_speeds = np.array(consumption_lut_file.iloc[:, 0]).astype(np.float64)
            emp_data = np.array(consumption_lut_file.iloc[:, 1:]).astype(np.float64)

            new_coordinates = []
            new_values = []

            # Update the LUT with the empirical data
            incline = 0.0
            level_of_loading = 0.5
            for i, temperature in enumerate(emp_temperatures):
                for j, speed in enumerate(emp_speeds):
                    # Interpolate the empirical data to the coordinates
                    consumption = emp_data[i, j]
                    if not np.isnan(consumption):
                        new_coordinates.append((incline, temperature, level_of_loading, speed))
                        new_values.append(consumption)
            consumption_lut.data_points = [
                [float(value) for value in coord] for coord in new_coordinates
            ]
            consumption_lut.values = [float(value) for value in new_values]
            logger.info(
                f"Loaded consumption LUT for vehicle type {vehicle_type.name_short} from {path}"
            )
        elif multiplier != 1.0:
            consumption_lut.values = [
                float(value * multiplier) for value in consumption_lut.values
            ]
            logger.info(
                f"Scaled consumption LUT for vehicle type {vehicle_type.name_short} by {multiplier}"
            )

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
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

            # Check if we need to override the consumption LUT for this vehicle type
            param_key_lut = f"{self.__class__.__name__}.override_consumption_lut"
            override_lut: Dict[str, Any] = params.get(param_key_lut, {})
            if vt.name_short in override_lut:
                lut_value = override_lut[vt.name_short]
                if isinstance(lut_value, (float, int)):
                    self.add_consumption_lut_for_vehicle_type(
                        session, vt, multiplier=float(lut_value)
                    )
                elif isinstance(lut_value, (str, Path)):
                    self.add_consumption_lut_for_vehicle_type(session, vt, path=Path(lut_value))
                else:
                    raise ValueError(
                        f"Invalid value for {param_key_lut}['{vt.name_short}']: {lut_value}. "
                        "Must be float or Path."
                    )
            else:
                # Add a default consumption LUT without scaling
                self.add_consumption_lut_for_vehicle_type(session, vt)

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
            geom_wkb = to_shape(station.geom).wkb  # type: ignore[arg-type]

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


class ReduceToNDaysNDepots(Modifier):
    """
    Reduce a dataset to a configurable number of days and depots.

    This modifier is useful for creating smaller test datasets from larger ones. It identifies
    the days with the most trips and the depots with the fewest rotations, then removes all
    rotations that don't match these criteria.

    The modifier performs the following operations:
    1. Identifies the N days with the most trips
    2. Identifies the M depots (start/end stations) with the fewest rotations
    3. Removes all rotations that don't start on one of the selected days
    4. Removes all rotations that don't start at one of the selected depots
    """

    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _get_default_num_days() -> int:
        """Get the default number of days to keep."""
        return 1

    @staticmethod
    def _get_default_num_depots() -> int:
        """Get the default number of depots to keep."""
        return 2

    def document_params(self) -> Dict[str, str]:
        """
        Document the parameters of this modifier.

        Returns:
        --------
        Dict[str, str]
            Dictionary documenting the parameters of this modifier.
        """
        return {
            f"{self.__class__.__name__}.num_days": """
Number of days with the most trips to keep in the dataset. The modifier will identify the days
with the highest trip counts and remove all rotations that don't start on one of these days.

**Format:** `int` (positive integer)

**Default:** 1

**Note:** Days with fewer trips (which might be just overflow into the wee hours of the following
day) are intentionally avoided by selecting days with the MOST trips.
            """.strip(),
            f"{self.__class__.__name__}.num_depots": """
Number of depots with the fewest rotations to keep in the dataset. The modifier will identify
depot stations (where rotations start/end) with the fewest rotations and keep only those.

**Format:** `int` (positive integer)

**Default:** 2

**Note:** Depots are identified by the stations where rotations start. The modifier keeps depots
with the fewest rotations to create smaller, more manageable test datasets.
            """.strip(),
        }

    def modify(self, session: Session, params: Dict[str, Any]) -> None:
        """
        Modify the database by reducing it to N days and M depots.

        Parameters:
        -----------
        session : Session
            SQLAlchemy session connected to the database to modify
        params : Dict[str, Any]
            Pipeline parameters containing optional num_days and num_depots

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
        param_key_days = f"{self.__class__.__name__}.num_days"
        param_key_depots = f"{self.__class__.__name__}.num_depots"

        num_days: int = params.get(param_key_days, self._get_default_num_days())
        num_depots: int = params.get(param_key_depots, self._get_default_num_depots())

        # Validation
        if num_days <= 0:
            raise ValueError(f"num_days must be positive, got {num_days}")
        if num_depots <= 0:
            raise ValueError(f"num_depots must be positive, got {num_depots}")

        # Find the days with the most trips
        # We can't use the days with the fewest trips since those might be just overflow
        # into the wee hours of the following day
        # Count trips by date
        from datetime import date

        day_count_dict: Dict[date, int] = defaultdict(int)
        all_trips = (
            session.query(Trip)
            .filter(Trip.rotation.has(Rotation.scenario_id == scenario_id))
            .all()
        )

        for trip in all_trips:
            trip_date = trip.departure_time.date()
            day_count_dict[trip_date] += 1

        if not day_count_dict:
            self.logger.warning("No trips found in database, nothing to reduce")
            return None

        # Convert to list of tuples and sort by trip count descending
        day_counts = [(day, count) for day, count in day_count_dict.items()]
        days_sorted = sorted(day_counts, key=lambda x: x[1], reverse=True)
        days_to_keep = [day for day, count in days_sorted[:num_days]]

        self.logger.info(
            f"Keeping {len(days_to_keep)} day(s) with most trips: "
            f"{', '.join(f'{day} ({count} trips)' for day, count in days_sorted[:num_days])}"
        )

        # Identify all the places where rotations start or end and count them
        all_start_end_stations: Dict[Station, int] = defaultdict(int)
        all_rotations_on_kept_days = (
            session.query(Rotation)
            .join(Trip)
            .filter(func.date(Trip.departure_time).in_(days_to_keep))
            .filter(Rotation.scenario_id == scenario_id)
            .options(joinedload(Rotation.trips).joinedload(Trip.route))
            .all()
        )

        for rotation in all_rotations_on_kept_days:
            if len(rotation.trips) == 0:
                raise ValueError(f"Rotation {rotation.name} has no trips")
            all_start_end_stations[rotation.trips[0].route.departure_station] += 1

        # Keep only the N depots with the fewest rotations starting or ending there
        if num_depots > len(all_start_end_stations):
            self.logger.warning(
                f"Requested {num_depots} depots but only {len(all_start_end_stations)} found. "
                f"Keeping all {len(all_start_end_stations)} depots."
            )
            depots_to_keep = list(all_start_end_stations.items())
        else:
            depots_to_keep = sorted(all_start_end_stations.items(), key=lambda x: x[1])[
                :num_depots
            ]

        depot_station_ids_to_keep = [depot[0].id for depot in depots_to_keep]

        self.logger.info(
            f"Keeping {len(depots_to_keep)} depot(s) with fewest rotations: "
            f"{', '.join(f'{depot[0].name} ({depot[1]} rotations)' for depot in depots_to_keep)}"
        )

        # Process all rotations and remove those that don't meet the criteria
        all_rotations_on_kept_days = (
            session.query(Rotation)
            .filter(Rotation.scenario_id == scenario_id)
            .options(joinedload(Rotation.trips).joinedload(Trip.route))
            .all()
        )

        to_delete: List[eflips.model.Base] = []
        rotations_kept = 0

        for rotation in all_rotations_on_kept_days:
            if len(rotation.trips) == 0:
                raise ValueError(f"Rotation {rotation.name} has no trips")

            first_station_id = rotation.trips[0].route.departure_station_id
            rotation_date = rotation.trips[0].departure_time.date()

            # Remove rotation if it doesn't start at a depot we're keeping or on a day we're keeping
            if (
                first_station_id not in depot_station_ids_to_keep
                or rotation_date not in days_to_keep
            ):
                self.logger.debug(
                    f"Removing rotation {rotation.name}, "
                    f"vehicle type {rotation.vehicle_type.name_short}, "
                    f"start {rotation.trips[0].route.departure_station.name}, "
                    f"end {rotation.trips[-1].route.arrival_station.name}, "
                    f"date {rotation_date}"
                )
                for trip in rotation.trips:
                    for stop_time in trip.stop_times:
                        to_delete.append(stop_time)
                    to_delete.append(trip)
                to_delete.append(rotation)
            else:
                rotations_kept += 1

        # Delete all marked objects
        for obj in to_delete:
            session.delete(obj)

        self.logger.info(
            f"Reduced dataset to {num_days} day(s) and {num_depots} depot(s). "
            f"Kept {rotations_kept} rotations, removed {len([obj for obj in to_delete if isinstance(obj, Rotation)])} rotations."
        )

        return None
