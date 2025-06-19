import logging

import sqlalchemy
from eflips.depot.api import simple_consumption_simulation
from eflips.model import Scenario, Rotation, Trip, Event, EventType
from sqlalchemy import not_

from util import create_consumption_results, clear_previous_simulation_results


def remove_terminus_charging_from_okay_rotations(
    scenario: Scenario,
    session: sqlalchemy.orm.session.Session,
    database_url: str,
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
    consumption_results = create_consumption_results(
        scenario, session, database_url=database_url
    )

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
