import os

from eflips.model import (
    Scenario,
    Trip,
    TripType,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload

if __name__ == "__main__":
    if "DATABASE_URL" not in os.environ:
        raise ValueError("Please set the DATABASE_URL environment variable.")
    DATABASE_URL = os.environ["DATABASE_URL"]
    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    scenario = session.query(Scenario).first()

    all_deadhead_trips = (
        session.query(Trip)
        .filter(Trip.scenario_id == scenario.id)
        .filter(Trip.trip_type == TripType.EMPTY)
        .options(joinedload(Trip.route))
        .all()
    )

    total_deadhead_km = sum([trip.route.distance for trip in all_deadhead_trips]) / 1000

    print(f"Total deadhead km: {total_deadhead_km}")
