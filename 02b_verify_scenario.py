#! /usr/bin/env python3

"""
Using system calls, drop the existing data (after asking for use confirmation) and import the data from
`00_bvg_schedule_input.sql`
"""

import os

from eflips.model import *
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    scenarios = session.query(Scenario).all()
    for scenario in scenarios:
        # Print the number of remaining routes, lines, and stations and trips
        print(
            f"Number of remaining routes: {session.query(Route).filter(Route.scenario == scenario).count()}"
        )
        print(
            f"Number of remaining lines: {session.query(Line).filter(Line.scenario == scenario).count()}"
        )
        print(
            f"Number of remaining stations: {session.query(Station).filter(Station.scenario == scenario).count()}"
        )
        print(
            f"Number of remaining trips: {session.query(Trip).filter(Trip.scenario == scenario).count()}"
        )
    session.rollback()
    session.close()
