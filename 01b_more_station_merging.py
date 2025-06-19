#! /usr/bin/env python3

import os
from typing import List

import sqlalchemy.orm.session
from eflips.model import *
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def merge_stations(stations: List[Station], session: sqlalchemy.orm.session.Session):
    main_station = min(stations, key=lambda station: len(station.name))
    if stations[1].name_short == "BF I":
        main_station = stations[1]
    for other_station in stations:
        if other_station != main_station:
            other_station_geom = other_station.geom
            with session.no_autoflush:
                # Update all routes, trips, and stoptimes containing the next station to point to the first station instead
                session.query(Route).filter(
                    Route.departure_station_id == other_station.id
                ).update({"departure_station_id": main_station.id})
                session.query(Route).filter(
                    Route.arrival_station_id == other_station.id
                ).update({"arrival_station_id": main_station.id})

                session.query(AssocRouteStation).filter(
                    AssocRouteStation.station_id == other_station.id
                ).update(
                    {"station_id": main_station.id, "location": other_station_geom}
                )

                session.query(StopTime).filter(
                    StopTime.station_id == other_station.id
                ).update({"station_id": main_station.id})
            session.flush()
            session.delete(other_station)


if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]
    SCENARIO_ID = 1

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    station_short_names_to_merge = [["BFI", "BF I"], ["UPAA", "UPBA"]]

    for station_short_names in station_short_names_to_merge:
        stations = (
            session.query(Station)
            .filter(Station.name_short.in_(station_short_names))
            .filter(Station.scenario_id == SCENARIO_ID)
            .all()
        )
        merge_stations(stations, session)

    session.commit()
    session.close()
