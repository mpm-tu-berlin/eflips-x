#! /usr/bin/env python3

import os
from typing import List, Dict

import pandas as pd
import seaborn as sns
from eflips.model import *
from eflips.model import TripType
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import plotutils

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]
    SCENARIO_ID = 1

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    short_names_of_depots = ["BF I", "BF M", "BF B", "BF S", "BF L", "BF C", "BF MDA"]

    # Provide some information about the remaining data. Create a dataframe showing the number of rotations, trips and
    # total kilometers for each vehicle type and originating depot
    vehicle_type_data: List[Dict[str, int | float]] = []
    for vehicle_type in (
        session.query(VehicleType).filter(VehicleType.scenario_id == SCENARIO_ID).all()
    ):
        for depot_short_name in short_names_of_depots:
            depot_station_name = (
                session.query(Station)
                .filter(Station.name_short == depot_short_name)
                .first()
                .name
            )
            depot_station_name = depot_station_name.removeprefix("Betriebshof ")
            depot_station_name = depot_station_name.removeprefix("Abstellfläche ")
            rotations = (
                session.query(Rotation)
                .join(Trip)
                .join(Route)
                .join(Station, Route.departure_station_id == Station.id)
                .join(VehicleType)
                .filter(
                    Rotation.scenario_id == SCENARIO_ID,
                    VehicleType.id == vehicle_type.id,
                    Station.name_short == depot_short_name,
                )
                .all()
            )
            trips = sum([len(rotation.trips) for rotation in rotations]) - 2
            total_distance = (
                sum(
                    [
                        sum(
                            [
                                (
                                    trip.route.distance
                                    if trip.trip_type == TripType.PASSENGER
                                    else 0
                                )
                                for trip in rotation.trips
                            ]
                        )
                        for rotation in rotations
                    ]
                )
                / 1000
            )
            vehicle_type_data.append(
                {
                    "Fahrzeugtyp": vehicle_type.name_short,
                    "depot": depot_station_name,
                    "Umläufe": len(rotations),
                    "trips": trips,
                    "Fahrzeugkilometer": total_distance,
                }
            )
    vehicle_type_df = pd.DataFrame(vehicle_type_data)

    # Sum up all the "Fahrzeugkilometer"
    print(f"Total Fahrzeugkilometer: {vehicle_type_df['Fahrzeugkilometer'].sum()}")

    vehicle_name_translation = {
        "EN": "Single Decker",
        "GN": "Articulated Bus",
        "DD": "Double Decker",
    }

    # Now, do two stacked bar plots, one for kilometers and one for trips
    # Stack each vehicle type on top of each other
    fig, ax = plt.subplots(
        1, 1, figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT)
    )

    # Replace the vehicle type names with the translated names
    vehicle_type_df["Fahrzeugtyp"] = vehicle_type_df["Fahrzeugtyp"].apply(
        lambda x: vehicle_name_translation[x]
    )
    vehicle_type_df["Fahrzeugkilometer"] *= 52 / 1000000

    # Change the color palette to seaborns default color palette, but start at the 3rd color
    palette = sns.color_palette("Set2")

    df2 = vehicle_type_df.pivot(
        index="depot", columns="Fahrzeugtyp", values="Fahrzeugkilometer"
    )

    # Order:Single Decker, Double Decker, Articulated Bus
    df2 = df2[["Single Decker", "Articulated Bus", "Double Decker"]]
    # Plot the data
    df2.plot(kind="bar", stacked=True, ax=ax, color=palette)
    df2.to_pickle("03a_vehicle_type_and_depot_plot.pkl")

    ax.set_title("")
    ax.set_ylabel(r"Revenue Mileage $\left[ \frac{km \times 10^6}{a} \right]$")

    # Remove xlabel, as the depots are already labeled on the x-axis
    ax.set_xlabel("")
    # 45 degree rotation for better readability
    plt.xticks(rotation=45)

    # Name Legend
    plt.legend(title="", bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper left", ncols=3)

    plt.tight_layout()
    # plt.subplots_adjust(top=0.8)
    plt.savefig("03a_vehicle_type_and_depot_plot.pdf")
