#! /usr/bin/env python3
import os
from math import sqrt

import numpy as np
import pandas as pd
from eflips.model import VehicleType, VehicleClass, ConsumptionLut, Scenario
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import plotutils

if __name__ == "__main__":
    if "DATABASE_URL" not in os.environ:
        raise ValueError("Please set the DATABASE_URL environment variable.")
    DATABASE_URL = os.environ["DATABASE_URL"]

    with open("consumption_lut.xlsx", "rb") as f:
        consumption_lut = pd.read_excel(f)
        # The LUT is a 2D table. The first column is the average speed.
        # The first row contains the temperatures.
        # Turn it into a multi-indexed dataframe
        emp_temperatures = np.array(consumption_lut.columns[1:]).astype(np.float64)
        emp_speeds = np.array(consumption_lut.iloc[:, 0]).astype(np.float64)
        emp_data = np.array(consumption_lut.iloc[:, 1:]).astype(np.float64)

    # For the second type, use eflips.model to calculate a LUT
    engine = create_engine(DATABASE_URL)
    session = Session(engine)
    try:
        vehicle_type = (
            session.query(VehicleType)
            .filter(VehicleType.name_short == "GN")
            .join(Scenario)
            .filter(Scenario.name_short == "OU")
            .one()
        )

        # Create a vehicle class for the vehicle type
        vehicle_class = VehicleClass(
            scenario_id=vehicle_type.scenario_id,
            name="Consumption LUT",
            vehicle_types=[vehicle_type],
        )
        session.add(vehicle_class)

        # Create a LUT for the vehicle class
        lut = ConsumptionLut.from_vehicle_type(vehicle_type, vehicle_class)
        session.add(lut)

        columns = lut.columns
        coordinates = lut.data_points
        data = lut.values

        # The arrangement here is a flattened 4D array. The dimensions are:
        # 'incline': incline. We take the ine where it's zero
        # 't_amb': ambient temperature. We take all values
        # 'leval_of_loading': level of loading. We take the value where it's 0.5
        # 'mean_speed_kmh': speed. We take all values

        # coordinates contains a list of list for each column value
        # data contains the consumption values

        # We need to turn this into a 2D table with the following columns:
        # 't_amb': ambient temperature
        # 'mean_speed_kmh': speed

        # Level of loading is derived from the mean pasenger count
        PASSENGER_MASS = 68  # kg
        PASSENGER_COUNT = 17.6  # German-wide average
        payload = PASSENGER_COUNT * PASSENGER_MASS
        vehicle_allowed_payload = vehicle_type.allowed_mass - vehicle_type.empty_mass
        level_of_loading = payload / vehicle_allowed_payload
        # Let's round it to the nearest 0.1
        level_of_loading = round(level_of_loading * 10) / 10

        coord_df = pd.DataFrame(coordinates, columns=columns)
        coord_df = coord_df[coord_df["level_of_loading"] == level_of_loading]
        coord_df = coord_df[coord_df["incline"] == 0]

        # Now we load the value from data and create the 2D array
        lut_data = np.zeros(
            dtype=np.float64, shape=(int(sqrt(len(coord_df))), int(sqrt(len(coord_df))))
        )
        lut_speeds = np.array(sorted((coord_df["mean_speed_kmh"]).unique()))
        lut_temperatures = np.array(sorted((coord_df["t_amb"]).unique()))

        for i, row in coord_df.iterrows():
            lut_data[
                np.where(lut_speeds == row["mean_speed_kmh"])[0][0],
                np.where(lut_temperatures == row["t_amb"])[0][0],
            ] = data[i]

        # We multiply the LUT Data here by the scaling factor (which is actually derived in step 4)
        SCALING_FACTOR = 2.07882047836436
        lut_data *= SCALING_FACTOR

        # We want to two separate plots, but with the same scales
        min_speed = min(lut_speeds.min(), emp_speeds.min())
        max_speed = max(lut_speeds.max(), emp_speeds.max())
        min_temp = min(lut_temperatures.min(), emp_temperatures.min())
        max_temp = max(lut_temperatures.max(), emp_temperatures.max())
        min_consumption = 0
        real_min_consumption = min(lut_data.min(), np.nanmin(emp_data))
        max_consumption = max(lut_data.max(), np.nanmax(emp_data))

        # Now, do a 3d Plot of the LUT
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            subplot_kw={"projection": "3d"},
            figsize=(plotutils.NORMAL_PLOT_WIDTH / 2, plotutils.NORMAL_PLOT_HEIGHT),
        )
        the_ax = ax
        X, Y = np.meshgrid(lut_speeds, lut_temperatures, indexing="ij")
        the_ax.plot_surface(
            X,
            Y,
            lut_data,
            cmap="viridis",
            vmin=real_min_consumption,
            vmax=max_consumption,
        )
        the_ax.set_xlabel(r"Speed $\left[ \frac{km}{h} \right]$")
        the_ax.set_ylabel("Temperature [°C]")
        the_ax.set_zlabel(r"Consumption  $\left[ \frac{kWh}{km} \right]$")
        # rotate the plot by 120 degrees
        the_ax.view_init(azim=45)

        # Set the ranges
        the_ax.set_xlim(min_speed, max_speed)
        the_ax.set_ylim(min_temp, max_temp)
        the_ax.set_zlim(min_consumption, max_consumption)

        # Set the layout so that the labels are not cut off
        plt.subplots_adjust(left=0.2, right=0.97, top=1.0, bottom=0.0)

        plt.savefig("03ba_consumption_lut_ji.pdf")
        plt.close()

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            subplot_kw={"projection": "3d"},
            figsize=(plotutils.NORMAL_PLOT_WIDTH / 2, plotutils.NORMAL_PLOT_HEIGHT),
        )

        the_ax = ax
        X, Y = np.meshgrid(emp_speeds, emp_temperatures, indexing="ij")
        the_ax.plot_surface(
            X,
            Y,
            emp_data,
            cmap="viridis",
            vmin=real_min_consumption,
            vmax=max_consumption,
        )
        the_ax.set_xlabel(r"Speed $\left[ \frac{km}{h} \right]$")
        the_ax.set_ylabel("Temperature [°C]")
        the_ax.set_zlabel(r"Consumption  $\left[ \frac{kWh}{km} \right]$")

        # rotate the plot by 120 degrees
        the_ax.view_init(azim=45)

        # Set the ranges
        the_ax.set_xlim(min_speed, max_speed)
        the_ax.set_ylim(min_temp, max_temp)
        the_ax.set_zlim(min_consumption, max_consumption)

        # Tight_layout without arguments squishes the plots together
        # Give them a bit more space to the left and right
        plt.subplots_adjust(left=0.2, right=0.97, top=1.0, bottom=0.0)
        plt.savefig("03bb_consumption_lut_emp.pdf")
        plt.close()
    finally:
        session.rollback()
        session.close()
