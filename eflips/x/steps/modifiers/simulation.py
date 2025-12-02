"""

The Simulation Modifiers run a simulation. This may also include depot optimization.

"""

import logging
from typing import Any, Dict

import sqlalchemy.orm.session
from eflips.depot.api import (  # type: ignore[import-untyped]
    generate_consumption_result,
    simple_consumption_simulation,
    group_rotations_by_start_end_stop,
    generate_depot_layout,
    generate_depot_optimal_size,
    generate_optimal_depot_layout,
    DepotConfigurationWish,
    simulate_scenario,
    SmartChargingStrategy,
)
from eflips.model import (
    Scenario,
    Station,
)
from sqlalchemy.exc import MultipleResultsFound

from eflips.x.framework import Modifier


class DepotGenerator(Modifier):
    """
    Generates depot infrastructure (Depot objects, Areas, and Processes) for a scenario.

    This modifier identifies depot stations (stations where vehicle rotations start or end) and
    creates the necessary depot infrastructure for each one. It supports three modes of operation:

    1. **Simple Layout (default)**: Fast generation using an all-direct layout (larger depots).
    2. **Optimal Layout**: Slower optimization-based generation that minimizes depot size.
    3. **Custom Configuration**: User-defined depot layouts via DepotConfigurationWish objects.

    The modifier will delete any existing depot infrastructure before creating new depots.

    Requirements:
    - Exactly one Scenario must exist in the database
    - Vehicle rotations must already be defined
    - For optimal layout mode: VehicleType objects must have dimensions (length, width, height)
    """

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
**MODE 3: Custom Configuration**

A list of `DepotConfigurationWish` objects from eflips.depot.api, one for each depot station
in the scenario. A depot station is any station where at least one vehicle rotation starts or ends.

When set, this parameter takes precedence over all other parameters (generate_optimal_depots,
charging_power_kw, and standard_block_length will be ignored).

**Requirements:**
- Must be a list (not a single object)
- Must contain exactly one DepotConfigurationWish per depot station
- Each DepotConfigurationWish must reference a valid depot station by station_id
- Cannot include stations that are not depot stations

**DepotConfigurationWish structure:**
- station_id: int - ID of the depot station
- auto_generate: bool - If True, depot layout is auto-generated; if False, must provide explicit areas
- default_power: float - Default charging power (required if auto_generate=True)
- standard_block_length: int - Block length for parking rows (required if auto_generate=True)
- areas: list[AreaInformation] - Explicit area definitions (required if auto_generate=False)
- cleaning_slots, cleaning_duration, shunting_slots, shunting_duration: Optional parameters

**Example:**
```python
from eflips.depot.api import DepotConfigurationWish

depot_wishes = [
    DepotConfigurationWish(
        station_id=1,
        auto_generate=True,
        default_power=120.0,
        standard_block_length=6
    ),
    DepotConfigurationWish(
        station_id=5,
        auto_generate=True,
        default_power=150.0,
        standard_block_length=8
    )
]
```

Default: None (uses automatic generation instead)
            """.strip(),
            f"{self.__class__.__name__}.generate_optimal_depots": """
**MODE 1 vs MODE 2: Automatic Generation Mode Selection**

*Only used if `depot_wishes` is NOT set.*

Controls which automatic depot generation mode to use:

- **False (default)**: MODE 1 - Simple Layout Generation
  - Fastest method
  - Creates an all-direct layout (every parking spot has direct access)
  - Results in larger depots with more space than needed
  - Good for quick testing or when space is not a constraint

- **True**: MODE 2 - Optimal Layout Generation
  - Slower optimization-based method
  - Minimizes depot size by optimizing parking arrangements
  - Uses standard_block_length to configure parking rows
  - Requires VehicleType objects to have dimensions (length, width, height)
  - Better for realistic depot sizing

Default: False
            """.strip(),
            f"{self.__class__.__name__}.charging_power_kw": """
**Charging power for automatic generation modes**

*Only used if `depot_wishes` is NOT set.*

The charging power (in kW) to be used for depot chargers when depots are automatically generated
(both simple and optimal modes). This value is applied to all generated depot charging infrastructure.

**Note:** If using depot_wishes with auto_generate=True, specify power in the DepotConfigurationWish
default_power parameter instead.

Default: 90.0 kW
            """.strip(),
            f"{self.__class__.__name__}.standard_block_length": """
**Block length for optimal layout generation**

*Only used if `depot_wishes` is NOT set AND `generate_optimal_depots` is True (MODE 2).*

The number of buses parked behind each other in one row during depot layout optimization.
This parameter influences the shape and efficiency of the depot:

- **Lower values (e.g., 4)**: More parking rows, fewer vehicles per row, less blocking, more flexible
- **Higher values (e.g., 10)**: Fewer parking rows, more vehicles per row, more blocking, more compact

Trade-offs:
- Higher values create more compact depots but increase the chance of vehicles blocking each other
- Lower values reduce blocking but require more total depot area

**Note:** If using depot_wishes with auto_generate=True, specify this in the DepotConfigurationWish
standard_block_length parameter instead.

Default: 6
            """.strip(),
        }

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
        """
        Generates depot infrastructure for all depot stations in the scenario.

        This method performs the following steps:
        1. Identifies all depot stations (stations where vehicle rotations start or end)
        2. Deletes any existing depot infrastructure (Depot objects, Areas, Processes)
        3. Generates new depot infrastructure based on the selected mode:

           - MODE 1 (Simple): Fast all-direct layout
           - MODE 2 (Optimal): Optimization-based layout
           - MODE 3 (Custom): User-specified DepotConfigurationWish objects

        Args:
            session: An open SQLAlchemy session connected to the database containing the scenario.
                    Must contain exactly one Scenario with defined vehicle rotations.
            params: Dictionary of parameters controlling depot generation:
                - DepotGenerator.depot_wishes: Optional list of DepotConfigurationWish objects
                - DepotGenerator.generate_optimal_depots: Boolean for MODE 1 vs MODE 2 (default: False)
                - DepotGenerator.charging_power_kw: Charging power in kW (default: 90.0)
                - DepotGenerator.standard_block_length: Block length for MODE 2 (default: 6)

        Returns:
            Nothing.

        Raises:
            ValueError: If no scenario exists, multiple scenarios exist, depot_wishes validation fails,
                       or depot stations are missing/extra in depot_wishes
            TypeError: If depot_wishes is not a list or contains non-DepotConfigurationWish objects

        Examples:

            MODE 1: Simple layout (default)::

                modifier.modify(session, {"DepotGenerator.charging_power_kw": 100.0})

            MODE 2: Optimal layout::

                modifier.modify(session, {
                    "DepotGenerator.generate_optimal_depots": True,
                    "DepotGenerator.charging_power_kw": 150.0,
                    "DepotGenerator.standard_block_length": 8
                })

            MODE 3: Custom configuration::

                from eflips.depot.api import DepotConfigurationWish
                wishes = [DepotConfigurationWish(station_id=1, auto_generate=True,
                                                default_power=120.0, standard_block_length=6)]
                modifier.modify(session, {"DepotGenerator.depot_wishes": wishes})
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

    def modify(self, session: sqlalchemy.orm.session.Session, params: Dict[str, Any]) -> None:
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
            SmartChargingStrategy.NONE,
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
