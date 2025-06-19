#! /usr/bin/env python3

"""
This script runs the consumption simulation on the "individual trip" level. This is done using custom functions, as
as this allows much greater performance due to parallelism.
"""
import dataclasses
import functools
import multiprocessing
import os
import zoneinfo
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from tempfile import gettempdir
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import sqlalchemy.orm.session
from eflips.model import (
    VehicleType,
    VehicleClass,
    ConsumptionLut,
    Trip,
    Temperatures,
    Scenario,
    Rotation,
)
from matplotlib import pyplot as plt
from pyomo.common.dependencies import scipy
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, joinedload
from tqdm.auto import tqdm

import plotutils
from util import _progress_process_method

PASSENGER_MASS = 68  # kg
PASSENGER_COUNT = 17.6  # German-wide average


@dataclass
class ConsumptionInformation:
    """
    A dataclass to hold the information needed for the consumption simulation.
    """

    trip_id: int
    vehicle_type_name: str
    consumption_lut: ConsumptionLut | None  # the LUT for the vehicle class
    average_speed: float  # the average speed of the trip in km/h
    distance: float  # the distance of the trip in km
    temperature: float  # The ambient temperature in °C
    level_of_loading: float
    incline: float = 0.0  # The incline of the trip in 0.0-1.0
    consumption: float = None  # The consumption of the trip in kWh
    consumption_per_km: float = None  # The consumption per km in kWh

    def calculate(self):
        """
        Calculates the consumption for the trip. Returns a float in kWh.

        :return: The energy consumption in kWh. This is already the consumption for the whole trip.
        """

        # Make sure the consumption lut has 4 dimensions and the columns are in the correct order
        if self.consumption_lut.columns != [
            "incline",
            "t_amb",
            "level_of_loading",
            "mean_speed_kmh",
        ]:
            raise ValueError(
                "The consumption LUT must have the columns 'incline', 't_amb', 'level_of_loading', 'mean_speed_kmh'"
            )

        # Recover the scales along each of the four axes from the datapoints
        incline_scale = sorted(set([x[0] for x in self.consumption_lut.data_points]))
        temperature_scale = sorted(
            set([x[1] for x in self.consumption_lut.data_points])
        )
        level_of_loading_scale = sorted(
            set([x[2] for x in self.consumption_lut.data_points])
        )
        speed_scale = sorted(set([x[3] for x in self.consumption_lut.data_points]))

        # Create the 4d array
        consumption_lut = np.zeros(
            (
                len(incline_scale),
                len(temperature_scale),
                len(level_of_loading_scale),
                len(speed_scale),
            )
        )

        # Fill it with NaNs
        consumption_lut.fill(np.nan)

        for i, (incline, temperature, level_of_loading, speed) in enumerate(
            self.consumption_lut.data_points
        ):
            consumption_lut[
                incline_scale.index(incline),
                temperature_scale.index(temperature),
                level_of_loading_scale.index(level_of_loading),
                speed_scale.index(speed),
            ] = self.consumption_lut.values[i]

        # Interpolate the consumption
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (incline_scale, temperature_scale, level_of_loading_scale, speed_scale),
            consumption_lut,
            bounds_error=False,
            fill_value=None,
            method="linear",
        )
        consumption_per_km = interpolator(
            [self.incline, self.temperature, self.level_of_loading, self.average_speed]
        )[0]
        self.consumption = consumption_per_km * self.distance
        self.consumption_per_km = consumption_per_km
        self.consumption_lut = None  # To save memory


@functools.lru_cache
def temperatures_for_scenario(
    scenario_id: int, session: sqlalchemy.orm.session.Session
) -> Temperatures:
    """
    Returns the temperatures for a scenario.
    """
    return (
        session.query(Temperatures)
        .filter(Temperatures.scenario_id == scenario_id)
        .one()
    )


def temperature_for_trip(
    trip_id: int, session: sqlalchemy.orm.session.Session
) -> float:
    """
    Returns the temperature for a trip. Finds the temperature for the mid-point of the trip.

    :param trip_id: The ID of the trip
    :param session: The SQLAlchemy session
    :return: A temperature in °C
    """

    trip = session.query(Trip).filter(Trip.id == trip_id).one()
    temperatures = temperatures_for_scenario(trip.scenario_id, session)

    # Find the mid-point of the trip
    mid_time = trip.departure_time + (trip.arrival_time - trip.departure_time) / 2

    if temperatures.use_only_time:
        # The temperatures are only given by time. We change our mid-time to be the date of the temperatures
        mid_time = datetime.combine(temperatures.datetimes[0].date(), mid_time.time())

    mid_time = mid_time.timestamp()

    datetimes = [dt.timestamp() for dt in temperatures.datetimes]
    temperatures = temperatures.data

    temperature = np.interp(mid_time, datetimes, temperatures)
    return float(temperature)


def extract_trip_information(trip_id: int, progress_queue: Queue | None = None):
    """
    Extracts the information needed for the consumption simulation from a trip.
    """
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    with Session(engine) as session:
        trip = (
            session.query(Trip)
            .filter(Trip.id == trip_id)
            .options(joinedload(Trip.route))
            .options(
                joinedload(Trip.rotation)
                .joinedload(Rotation.vehicle_type)
                .joinedload(VehicleType.vehicle_classes)
                .joinedload(VehicleClass.consumption_lut)
            )
            .one()
        )
        # Check exactly one of the vehicle classes has a consumption LUT
        all_consumption_luts = [
            vehicle_class.consumption_lut
            for vehicle_class in trip.rotation.vehicle_type.vehicle_classes
        ]
        all_consumption_luts = [x for x in all_consumption_luts if x is not None]
        if len(all_consumption_luts) != 1:
            raise ValueError(
                f"Expected exactly one consumption LUT, got {len(all_consumption_luts)}"
            )
        consumption_lut = all_consumption_luts[0]
        # Disconnect the consumption LUT from the session to avoid loading the whole table
        session.expunge(consumption_lut)
        del all_consumption_luts

        total_distance = trip.route.distance / 1000  # km
        total_duration = (
            trip.arrival_time - trip.departure_time
        ).total_seconds() / 3600
        average_speed = total_distance / total_duration  # km/h

        temperature = temperature_for_trip(trip_id, session)

        payload_mass = PASSENGER_MASS * PASSENGER_COUNT
        full_payload = (
            trip.rotation.vehicle_type.allowed_mass
            - trip.rotation.vehicle_type.empty_mass
        )
        level_of_loading = payload_mass / full_payload

    engine.dispose()

    if progress_queue is not None:
        progress_queue.put(1)

    info = ConsumptionInformation(
        trip_id=trip_id,
        vehicle_type_name=trip.rotation.vehicle_type.name_short,
        consumption_lut=consumption_lut,
        average_speed=average_speed,
        distance=total_distance,
        temperature=temperature,
        level_of_loading=level_of_loading,
    )

    info.calculate()
    return info


def update_temperatures(
    scenario_id: int, key: str, session: sqlalchemy.orm.session.Session
) -> None:
    # Delete the old temperatures
    old_temperatures = (
        session.query(Temperatures)
        .filter(Temperatures.scenario_id == scenario_id)
        .one_or_none()
    )
    if old_temperatures is not None:
        session.delete(old_temperatures)

    # Update the temperatures of the scenario
    temperature_df = pd.read_excel("04_temperatures.xlsx")
    temperature_row = list(temperature_df[key])
    # this is a 24h profile. create a corresponding datetime in the correct timezone
    datetimes = []
    for i in range(24):
        datetimes.append(
            datetime(1970, 1, 1, i, 0, 0, 0, zoneinfo.ZoneInfo("Europe/Berlin"))
        )

    assert len(temperature_row) == len(datetimes)

    new_temperatures = Temperatures(
        scenario_id=scenario_id,
        name=key,
        use_only_time=True,
        datetimes=datetimes,
        data=temperature_row,
    )
    session.add(new_temperatures)


def add_consumption_luts(scenario_id: int, session: sqlalchemy.orm.session.Session):
    """
    Adds the consumption LUTs for a scenario.
    """

    passenger_counts = {
        "EN": 70,
        "GN": 100,
        "DD": 112,
    }

    for vehicle_type in (
        session.query(VehicleType).filter(VehicleType.scenario_id == scenario_id).all()
    ):
        passenger_count = passenger_counts[vehicle_type.name_short]
        vehicle_type.allowed_mass = (
            vehicle_type.empty_mass + passenger_count * PASSENGER_MASS
        )
        # Check if the vehicle type already has a vehicle class
        if any(
            vehicle_class.consumption_lut is not None
            for vehicle_class in vehicle_type.vehicle_classes
        ):
            continue

        vehicle_type.consumption = None

        # Create a vehicle class for the vehicle type
        vehicle_class = VehicleClass(
            scenario_id=vehicle_type.scenario_id,
            name=f"Consumption LUR for {vehicle_type.name_short}",
            vehicle_types=[vehicle_type],
        )
        session.add(vehicle_class)
        consumption_lut = ConsumptionLut.from_vehicle_type(vehicle_type, vehicle_class)
        session.add(consumption_lut)


def energy_consumption_for_key(key: str) -> pd.DataFrame:
    return_ = """
    Create a dataframe of energy consumption for a key.
    :param key: The key to use. Can be 'Month January' – 'Month December', 'Hottest Day (by average)', 'Coldest Day (by average)', 'DIN'
    :return: A pandas DataFrame with the energy consumption for each trip.
    """
    temp_dir = os.path.join(gettempdir(), "consumption_dfs")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f"{key}.pkl")
    if os.path.exists(temp_file):
        return pd.read_pickle(temp_file)
    else:
        engine = create_engine(DATABASE_URL)

        with Session(engine) as session:
            scenario_name = "OU"
            scenario = (
                session.query(Scenario)
                .filter(Scenario.name_short == scenario_name)
                .one()
            )
            update_temperatures(scenario.id, key, session)
            add_consumption_luts(scenario.id, session)

            # Commit() is needed cause we'll be using new sessions in the parallelized part
            session.commit()

            # Get all the trips for the scenario
            trip_q = session.query(Trip.id).filter(Trip.scenario_id == scenario.id)

            # This is my "parallelism with progress" pattern
            # Use my "parallelism with progress" boilerplate
            parallelism = True
            if parallelism:
                # Set up progress reporting
                progress_manager = multiprocessing.Manager()
                progress_queue = progress_manager.Queue()
                progress_process = multiprocessing.Process(
                    target=_progress_process_method,
                    args=(trip_q.count(), progress_queue),
                )
                progress_process.start()

                pool_args: List[Tuple[int, multiprocessing.Queue]] = []
                for trip in trip_q:
                    pool_args.append((trip[0], progress_queue))

                with multiprocessing.Pool() as p:
                    results = p.starmap(extract_trip_information, pool_args)

                # Remember to kill the progress process
                progress_process.kill()
            else:
                results = list()
                for trip in tqdm(trip_q.all(), smoothing=0):
                    results.append(extract_trip_information(trip[0], None))

            # Plot the results for now
            as_dicts = [dataclasses.asdict(x) for x in results]
            del results
            # Remove the memory-intensive consumption LUT
            for d in as_dicts:
                del d["consumption_lut"]

            df = pd.DataFrame(
                as_dicts,
            )

            vehicle_type_description = {
                "DD": "Double Decker (empirical model)",
                "EN": "Single Decker (empirical model)",
                "GN": "Articulated Bus (own measurements)",
            }
            # Update the vehicle type descriptions
            df["vehicle_type_name"] = df["vehicle_type_name"].map(
                vehicle_type_description
            )

            # Rename "vehicle_type" to "Vehicle Type"
            df = df.rename(columns={"vehicle_type_name": "Vehicle Type"})
            df.to_pickle(temp_file)
        engine.dispose()
        return df


if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    means_by_vehicle_type_and_key = list()
    row_indices = list()

    for key in (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "29.6°C",
        "-12°C",
    ):
        df = energy_consumption_for_key(key)
        if key == "-12°C":
            fig, ax = plt.subplots()
            fig.set_figwidth(plotutils.NORMAL_PLOT_WIDTH)
            fig.set_figheight(plotutils.NORMAL_PLOT_HEIGHT)
            sns.histplot(
                data=df,
                x="consumption_per_km",
                hue="Vehicle Type",
                element="step",
                ax=ax,
                legend=True,
                binwidth=0.05,
            )

            ax.set_xlabel(r"Consumption $\left[\frac{kWh}{km}\right]$")
            ax.set_ylabel("Count")
            plt.tight_layout()
            plt.savefig("04_consumption_per_km.pdf")
            plt.close()

        result = dict()
        row_indices.append(key)
        for vehicle_type in df["Vehicle Type"].unique():
            result[vehicle_type] = df[df["Vehicle Type"] == vehicle_type][
                "consumption_per_km"
            ].mean()
        means_by_vehicle_type_and_key.append(result)

    df = pd.DataFrame(means_by_vehicle_type_and_key, index=row_indices)

    # Scale the data
    if True:
        # Now, we want to find the scaling factor for the "EN" vehicle type. We have real by-quarter data for the "EN"
        # vehicle type.

        # summarise the monthly data to quarterly data
        quarters = [
            [
                "January",
                "February",
                "March",
            ],
            [
                "April",
                "May",
                "June",
            ],
            [
                "July",
                "August",
                "September",
            ],
            [
                "October",
                "November",
                "December",
            ],
        ]
        real_consumption_by_quarter = [
            1.6580115,
            1.3629038,
            1.3028950,
            1.5893908,
        ]
        model_consumption_by_quarter = list()
        for i, quarter in enumerate(quarters):
            months_in_quarter = quarters[i]
            consumptions_this_quarter = df.loc[months_in_quarter][
                "Single Decker (empirical model)"
            ]
            model_consumption_by_quarter.append(consumptions_this_quarter.mean())

        scaling_factors = np.array(real_consumption_by_quarter) / np.array(
            model_consumption_by_quarter
        )

        print(f"Scaling factors: {scaling_factors}")
        print(f"Mean scaling factor: {np.mean(scaling_factors)}")
        print(f"Standard deviation of scaling factors: {np.std(scaling_factors)}")

        # For the "EN" ("Single Decker") vehicle type and the "DD" ("Double Decker") vehicle type, we will scale both
        # the data in the dataframe and the "consumption" object in the database. We will not scale the "GN" ("Articulated Bus")
        # vehicle type.
        df["Single Decker (empirical model)"] *= np.mean(scaling_factors)
        df["Double Decker (empirical model)"] *= np.mean(scaling_factors)

        engine = create_engine(DATABASE_URL)
        with Session(engine) as session:
            for vehicle_type in ("EN", "DD"):
                vehicle_type_obj = (
                    session.query(VehicleType)
                    .filter(VehicleType.name_short == vehicle_type)
                    .all()
                )
                for vt in vehicle_type_obj:
                    assert len(vt.vehicle_classes) == 1
                    consumption_lut = vt.vehicle_classes[0].consumption_lut
                    np_values = np.array(consumption_lut.values) * float(
                        np.mean(scaling_factors)
                    )
                    py_values = [float(x) for x in np_values]
                    consumption_lut.values = py_values
            session.commit()
        engine.dispose()

    # Create a line plot
    fig, ax = plt.subplots(ncols=2, sharey=True, gridspec_kw={"width_ratios": [1, 2]})
    fig.set_figwidth(plotutils.NORMAL_PLOT_WIDTH)
    fig.set_figheight(plotutils.NORMAL_PLOT_HEIGHT)

    # Change the color palette to colorbrewer Set2
    palette = sns.color_palette("Set2")

    first_ax = ax[1]
    # Here, we do a line plot of the first 12 keys
    df.iloc[:12].plot(ax=first_ax, legend=True, color=palette)
    first_ax.set_xlabel("Month")
    # Rotate the x-axis labels 45 degrees, and do them every three months
    first_ax.set_xticks(range(12))
    first_ax.set_xticklabels(df.iloc[:12].index, rotation=45, ha="right")

    ax[0].set_ylabel(r"Mean consumption $\left[\frac{kWh}{km}\right]$")

    second_ax = ax[0]
    # Plot a bar plot of the last three keys
    df.iloc[12:].plot(kind="bar", ax=second_ax, legend=False, color=palette)
    # Rotate the tick labels 45 degrees and replace " " with "\n"
    second_ax.set_xticklabels(
        [x.replace(" ", "\n") for x in df.iloc[12:].index], rotation=45, ha="right"
    )

    plt.tight_layout()
    plt.savefig("04_consumption_per_km_by_key.pdf")
    plt.close()
