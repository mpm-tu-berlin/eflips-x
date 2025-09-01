import logging
from collections import Counter

import sqlalchemy
from eflips.depot.api import simple_consumption_simulation, generate_consumption_result
from eflips.model import (
    Scenario,
    Rotation,
    Trip,
    Event,
    EventType,
    Vehicle,
    Station,
    ChargeType,
    VoltageLevel,
)
from sqlalchemy import not_
from sqlalchemy.orm import Session

from eflips.x.util_legacy import clear_previous_simulation_results


def remove_terminus_charging_from_okay_rotations(
    scenario: Scenario,
    session: sqlalchemy.orm.session.Session,
) -> None:
    """
    Run the consumption model for the scenario and for those rotations where the consumption is okay, remove the
    ability to charge at the terminus.

    :param scenario: The scenario to run the consumption model for.
    :param session: An open database session.
    :return: Nothing. Database is updated in place.
    """
    logger = logging.getLogger(__name__)

    # Run the consumption model
    clear_previous_simulation_results(scenario, session)
    consumption_results = generate_consumption_result(scenario)

    # `create_consumption_results` may have detached our scenario object from the session
    session.add(scenario)
    session.flush()

    simple_consumption_simulation(
        initialize_vehicles=True,
        scenario=scenario,
        consumption_result=consumption_results,
    )

    # Get the minimum SoC for each rotations
    low_soc_rot_q = (
        session.query(Rotation.id)
        .join(Trip)
        .join(Event)
        .filter(Rotation.scenario_id == scenario.id)
        .filter(Event.event_type == EventType.DRIVING)
        .filter(Event.soc_end < 0)
        .distinct()
    )
    high_soc_rot_q = (
        session.query(Rotation)
        .filter(Rotation.scenario_id == scenario.id)
        .filter(not_(Rotation.id.in_(low_soc_rot_q)))
    )

    logger.info(
        f"{low_soc_rot_q.count()} rotations with low SoC, {high_soc_rot_q.count()} with high SoC, {session.query(Rotation).filter(Rotation.scenario_id == scenario.id).count()} total rotations"
    )

    # For the rotations with high SoC, remove the ability to charge at the terminus
    for rotation in high_soc_rot_q:
        rotation.allow_opportunity_charging = False
    session.flush()


def number_of_rotations_below_zero(scenario: Scenario, session: Session) -> int:
    """
    Counts the number of rotations in a scenario where the SOC at the depot is below 0%.
    :param scenario: The scenario to check.
    :param session: The database session.
    :return: The number of rotations with SOC below 0%.
    """
    rotations_q = (
        session.query(Rotation)
        .join(Trip)
        .join(Event)
        .filter(Rotation.scenario_id == scenario.id)
        .filter(Event.event_type == EventType.DRIVING)
        .filter(Event.soc_end < 0)
    )
    return rotations_q.count()


def clear_previous_simulation_results(scenario: Scenario, session: Session) -> None:
    session.query(Rotation).filter(Rotation.scenario_id == scenario.id).update(
        {"vehicle_id": None}
    )
    session.query(Event).filter(Event.scenario_id == scenario.id).delete()
    session.query(Vehicle).filter(Vehicle.scenario_id == scenario.id).delete()


def add_charging_station(scenario, session, power: float = 450) -> int | None:
    """
    Adds a charging station to the scenario. The heuristic for selecting the charging station is to add is to select
    the one where the rotations with negative SoC spend the most time. If no such charging station can be found, None
    is returned.

    If the station selected is already electrified, the next best station is selected.

    :param scenario: THE scenario to add the charging station to.
    :param session: An open database session.
    :param power: The power of the charging station to be added. Default is 450 kW.
    :return: Either the id of the charging station that was added, or None if no charging station could be added.
    """
    # First, we identify all the rotations containing a SoC < 0 event
    logger = logging.getLogger(__name__)

    rotations_with_low_soc = (
        session.query(Rotation)
        .join(Trip)
        .join(Event)
        .filter(Event.soc_end < 0)
        .filter(Event.event_type == EventType.DRIVING)
        .filter(Event.scenario == scenario)
        .options(sqlalchemy.orm.joinedload(Rotation.trips).joinedload(Trip.route))
        .distinct()
        .all()
    )

    # For these rotations, we find all the arrival statiosn but the last one. The last one is the depot.
    # We sum up the time spent at a break at each of these stations.
    total_break_time_by_station = Counter()
    for rotation in rotations_with_low_soc:
        for i in range(len(rotation.trips) - 1):
            trip = rotation.trips[i]
            total_break_time_by_station[trip.route.arrival_station_id] += int(
                (rotation.trips[i + 1].departure_time - trip.arrival_time).total_seconds()
            )

    # If all stations have a score iof 0, we terminate the optimization
    if all(v == 0 for v in total_break_time_by_station.values()):
        return None

    for most_popular_station_id, _ in total_break_time_by_station.most_common():
        station: Station = (
            session.query(Station).filter(Station.id == most_popular_station_id).one()
        )
        if station.is_electrified:
            logger.warning(
                f"Station {station.name} is already electrified. Choosing the next best station."
            )
            continue

        if logger.isEnabledFor(logging.DEBUG):
            station_name = (
                session.query(Station).filter(Station.id == most_popular_station_id).one().name
            )
            logger.debug(
                f"Station {most_popular_station_id} ({station_name}) was selected as the station where the most time is spent."
            )

        # Actually add the charging station in the database.

        station.is_electrified = True
        station.amount_charging_places = 100
        station.power_per_charger = power
        station.power_total = station.amount_charging_places * station.power_per_charger
        station.charge_type = ChargeType.OPPORTUNITY
        station.voltage_level = VoltageLevel.MV

        return most_popular_station_id
