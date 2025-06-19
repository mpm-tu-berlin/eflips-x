#!/usr/bin/env python3

"""
This script finds for each rotation
- the scenario
- the distance (considering only passenger trips)
- the duration (considering only passenger trips)
- the schedule efficiency (considering only passenger trips)
- fraction of trips on the most popular line for the rotation
"""

import os
from collections import Counter
from datetime import timedelta
from typing import Dict

import pandas as pd
from eflips.model import *
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload
from tqdm.auto import tqdm

from util import update_database_to_most_recent_schema


def info_for_rotation(
    rotation: Rotation, session: sqlalchemy.orm.session.Session
) -> Dict[str, float]:
    """
    Returns a dictionary with the scenario, distance, duration and schedule efficiency for a given rotation.
    :param rotation: A rotation
    :param session: An open EBO database session
    :return: A dictionary with the scenario, distance, duration and schedule efficiency
    """
    scenario_short_name = rotation.scenario.name_short
    distance = 0  # m
    for trip in rotation.trips:
        distance += trip.route.distance
    distance /= 1000  # km
    duration = sum(
        [
            trip.arrival_time - trip.departure_time
            for trip in rotation.trips
            if trip.trip_type == TripType.PASSENGER
        ],
        timedelta(),
    )

    first_trip_departure = min([trip.departure_time for trip in rotation.trips])
    last_trip_arrival = max([trip.arrival_time for trip in rotation.trips])
    schedule_efficiency = duration / (last_trip_arrival - first_trip_departure)

    # Duration for analysis is the between the first trip departure and the last trip arrival
    duration_for_analysis = last_trip_arrival - first_trip_departure

    lines_and_count = Counter()
    for trip in rotation.trips:
        if trip.trip_type == TripType.PASSENGER:
            lines_and_count[trip.route.line_id] += 1

    most_popular_line, count = lines_and_count.most_common(1)[0]
    most_popular_line_fraction = count / len(
        [trip for trip in rotation.trips if trip.trip_type == TripType.PASSENGER]
    )

    total_deadhead_m = sum(
        [
            trip.route.distance
            for trip in rotation.trips
            if trip.trip_type == TripType.EMPTY
        ]
    )

    return {
        "scenario": scenario_short_name,
        "distance": distance,
        "duration": duration_for_analysis.total_seconds() / 3600,
        "schedule_efficiency": schedule_efficiency,
        "most_popular_line_fraction": most_popular_line_fraction,
        "total_deadhead_km": total_deadhead_m / 1000,
    }


if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # Update the database schema
    update_database_to_most_recent_schema()

    all_rotations = (
        session.query(Rotation)
        .options(joinedload(Rotation.trips).joinedload(Trip.route))
        .options(joinedload(Rotation.scenario))
    )

    results = []
    for rotation in tqdm(all_rotations, total=all_rotations.count()):
        results.append(info_for_rotation(rotation, session))

    df = pd.DataFrame(results)

    # Group by scenario and calculate the mean
    df_grouped = df.groupby("scenario").mean()

    df_grouped.to_excel("09_rotation_info.xlsx")

    # Also show the maximum for df["duration"] for each scenario
    df_grouped["max_duration"] = df.groupby("scenario")["duration"].max()
    print("Max duration:")
    print(df_grouped)

    # Also do one with count
    df_grouped = df.groupby("scenario").count()
    print("Count:")
    print(df_grouped)

    # and with sum
    df_grouped = df.groupby("scenario").sum()
    print("Sum:")
    print(df_grouped)
