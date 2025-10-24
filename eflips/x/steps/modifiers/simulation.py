"""

The Simulation Modifiers run a simulation. This may also include depot optimization.

"""

import logging
from pathlib import Path
from typing import Any, Dict

import eflips.model
import sqlalchemy.orm.session
from eflips.depot.api import (
    generate_consumption_result,
    simple_consumption_simulation,
    group_rotations_by_start_end_stop,
    generate_depot_layout,
    generate_depot_optimal_size,
    generate_optimal_depot_layout,
    DepotConfigurationWish,
    simulate_scenario,
)
from eflips.model import (
    Scenario,
    Station,
)
from sqlalchemy.exc import MultipleResultsFound

from eflips.x.framework import Modifier


class DepotGenerator(Modifier):
    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    @property
    def default_charging_power_kw(self) -> float:
        return 90.0

    @property
    def default_standard_block_length(self) -> int:
        return 6

    def document_params(self) -> Dict[str, str]:
        return {
            f"{self.__class__.__name__}.depot_wishes": """
A list of `DepotConfigurationWish` objects defining the desired depots to be generated. If set, it must
contain one object for each depot (station where rotations start or end) in the scenario. If not set,
depot generation will be done automatically based on existing stations.

Default: None
            """.strip(),
            f"{self.__class__.__name__}.generate_optimal_depots": """
*Only used if `depot_wishes` is not set.* If true, depots will be generated automatically based on an 
optimization procedure that tries to minimize the depot size. If False, depots will be generated using
an all-direct layout and much larger than needed (however, this is much faster to compute).
            """.strip(),
            f"{self.__class__.__name__}.charging_power_kw": """
*Only used if `depot_wishes` is not set.* The charging power (in kW) to be used for depot chargers.

Default: 90
            """.strip(),
            f"{self.__class__.__name__}.standard_block_length": """
*Only used if `depot_wishes` is not set and `generate_optimal_depots` is True.*

The amount of buses parked behing each other in one row in the depot layout optimization. This
influences the shape of the depot. A higher value will lead to less areas, but more blocking of 
vehicles as they park behind each other.
            
Default: 6
            """.strip(),
        }

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Path:
        """
        Generates depots based on the provided parameters.
        :param session: An open SQLAlchemy session.
        :param params:
        :return:
        """
        # Extract parameters
        depot_wishes = params.get(f"{self.__class__.__name__}.depot_wishes", None)
        generate_optimal_depots = params.get(
            f"{self.__class__.__name__}.generate_optimal_depots", False
        )
        charging_power_kw = params.get(
            f"{self.__class__.__name__}.charging_power_kw", self.default_charging_power_kw
        )
        standard_block_length = params.get(
            f"{self.__class__.__name__}.standard_block_length", self.default_standard_block_length
        )

        # Get the scenario
        try:
            scenario = session.query(Scenario).one_or_none()
            if scenario is None:
                raise ValueError("No scenario found in the database.")
        except MultipleResultsFound:
            count = session.query(Scenario).count()
            raise ValueError(f"Expected exactly one scenario in the database, found {count}.")

        # Validate parameters
        if depot_wishes is not None:
            # Validate that depot_wishes is a list
            if not isinstance(depot_wishes, list):
                raise TypeError(
                    f"depot_wishes must be a list of DepotConfigurationWish objects, got {type(depot_wishes).__name__}"
                )

            # Validate that all elements are DepotConfigurationWish objects
            for i, wish in enumerate(depot_wishes):
                if not isinstance(wish, DepotConfigurationWish):
                    raise TypeError(
                        f"depot_wishes[{i}] must be a DepotConfigurationWish object, got {type(wish).__name__}"
                    )

            # Validate depot_wishes against actual depots in the scenario
            # Group rotations by start/end station to get all depot stations
            rotation_groups = group_rotations_by_start_end_stop(scenario.id, session)

            # Extract unique depot stations (stations where rotations start or end)
            depot_stations = set()
            for (start_station, end_station), _ in rotation_groups.items():
                depot_stations.add(start_station)
                depot_stations.add(end_station)

            # Create a set of station IDs from depot_wishes
            wish_station_ids = {wish.station_id for wish in depot_wishes}
            depot_station_ids = {station.id for station in depot_stations}

            # Check if depot_wishes covers all depot stations
            missing_stations = depot_station_ids - wish_station_ids
            extra_stations = wish_station_ids - depot_station_ids

            if missing_stations:
                missing_names = [
                    session.query(Station).filter(Station.id == sid).one().name
                    for sid in missing_stations
                ]
                raise ValueError(
                    f"depot_wishes is missing configuration for the following depot stations: {', '.join(missing_names)}. "
                    f"Each depot station (where rotations start or end) must have a corresponding DepotConfigurationWish."
                )

            if extra_stations:
                extra_names = [
                    session.query(Station).filter(Station.id == sid).one().name
                    for sid in extra_stations
                ]
                raise ValueError(
                    f"depot_wishes contains configuration for stations that are not depot stations: {', '.join(extra_names)}. "
                    f"Only stations where rotations start or end should be included."
                )

            # Check for contradictory parameters when depot_wishes is set
            if generate_optimal_depots:
                self.logger.warning(
                    "Parameter 'generate_optimal_depots' is ignored when 'depot_wishes' is set. "
                    "The depot layout will be generated based on the provided depot configuration wishes."
                )

            # Generate depots using depot wishes
            self.logger.info(
                f"Generating depots using {len(depot_wishes)} depot configuration wishes."
            )
            generate_optimal_depot_layout(
                depot_config_wishes=depot_wishes,
                scenario=scenario,
                database_url=None,  # Use the current session
                delete_existing_depot=True,
            )

        else:
            # depot_wishes is not set, use automatic generation
            if generate_optimal_depots:
                # Use optimization-based generation
                self.logger.info(
                    f"Generating depots with optimization (standard_block_length={standard_block_length}, "
                    f"charging_power={charging_power_kw} kW)."
                )
                generate_depot_optimal_size(
                    scenario=scenario,
                    standard_block_length=standard_block_length,
                    charging_power=charging_power_kw,
                    database_url=None,  # Use the current session
                    delete_existing_depot=True,
                )
            else:
                # Use simple generation
                self.logger.info(
                    f"Generating depots with simple layout (charging_power={charging_power_kw} kW)."
                )
                generate_depot_layout(
                    scenario=scenario,
                    charging_power=charging_power_kw,
                    database_url=None,  # Use the current session
                    delete_existing_depot=True,
                )

        self.logger.info("Depot generation completed successfully.")
        return Path(".")


class Simulation(Modifier):
    def __init__(self, code_version: str = "v1.0.0", **kwargs):
        super().__init__(code_version=code_version, **kwargs)
        self.logger = logging.getLogger(__name__)

    def document_params(self) -> Dict[str, str]:
        return {
            f"{self.__class__.__name__}.repetition_period": """
The repetition period (in days) for the simulation. This defines after how many days the schedule repeats.
It should be a timedelta object representing whole days (e.g., timedelta(days=7) for a weekly repetition).

Default: auto-detected based on the scenario.
            """.strip(),
            f"{self.__class__.__name__}.smart_charging": """
Set a eflips.depot.api.SmartChargingStrategy to be used during the simulation. If not set, no smart charging
will be applied (SmartChargingStrategy.NONE).
Default: SmartChargingStrategy.NONE
            """.strip(),
            f"{self.__class__.__name__}.ignore_unstable_simulation": """
If True, the simulation will not raise an exception if it becomes unstable.
            
Default: False
            """.strip(),
            f"{self.__class__.__name__}.ignore_delayed_trips": """
If True, the simulation will ignore delayed trips instead of raising an exception.

Default: False
            """.strip(),
        }

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> Path:
        """
        Runs a simulation based on the provided parameters.
        :param session: An open SQLAlchemy session.
        :param params:
        :return:
        """

        # Extract parameters
        repetition_period = params.get(f"{self.__class__.__name__}.repetition_period", None)
        smart_charging = params.get(
            f"{self.__class__.__name__}.smart_charging",
            eflips.depot.api.SmartChargingStrategy.NONE,
        )
        ignore_unstable_simulation = params.get(
            f"{self.__class__.__name__}.ignore_unstable_simulation", False
        )
        ignore_delayed_trips = params.get(f"{self.__class__.__name__}.ignore_delayed_trips", False)

        scenario = session.query(Scenario).one()

        ##### Step 1: Consumption simulation
        consumption_results = generate_consumption_result(scenario)
        simple_consumption_simulation(
            scenario, initialize_vehicles=True, consumption_result=consumption_results
        )

        ##### Step 2: Run the simulation
        simulate_scenario(
            scenario=scenario,
            repetition_period=repetition_period,
            smart_charging_strategy=smart_charging,
            ignore_unstable_simulation=ignore_unstable_simulation,
            ignore_delayed_trips=ignore_delayed_trips,
        )

        ##### Step 3: Consumption simulation
        consumption_results = generate_consumption_result(scenario)
        simple_consumption_simulation(
            scenario, initialize_vehicles=False, consumption_result=consumption_results
        )
