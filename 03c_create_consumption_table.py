#! /usr/bin/env python3

"""
This script loads the "consumption_lut.xlsx" containing the consumption for the 18 meter buses, turns it into an eFLIPS-
compatible consumption LUT and sets it as the consumption LUT for the vehicle type "GN" in all scenarios.
"""

import os

import numpy as np
import pandas as pd
from eflips.model import VehicleType, VehicleClass, ConsumptionLut
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

if __name__ == "__main__":
    if "DATABASE_URL" not in os.environ:
        raise ValueError("Please set the DATABASE_URL environment variable.")
    DATABASE_URL = os.environ["DATABASE_URL"]

    ## STEP 1: Create a model-based consumption LUT for all vehicle types in all scenarios
    engine = create_engine(DATABASE_URL)
    session = Session(engine)
    try:
        for vehicle_type in session.query(VehicleType):

            # Create a vehicle class for the vehicle type
            vehicle_class = VehicleClass(
                scenario_id=vehicle_type.scenario_id,
                name=f"Consumption LUT for {vehicle_type.name_short}",
                vehicle_types=[vehicle_type],
            )
            session.add(vehicle_class)

            # Create a LUT for the vehicle class
            lut = ConsumptionLut.from_vehicle_type(vehicle_type, vehicle_class)

            # Change the values to a python list of floats
            lut.values = [float(value) for value in lut.values]

            session.add(lut)
            session.flush()

        ### STEP 2: Load the empirical data and update the LUT for the vehicle type "GN" in all scenarios
        with open("consumption_lut.xlsx", "rb") as f:
            consumption_lut = pd.read_excel(f)
            # The LUT is a 2D table. The first column is the average speed.
            # The first row contains the temperatures.
            # Turn it into a multi-indexed dataframe
            emp_temperatures = np.array(consumption_lut.columns[1:]).astype(np.float64)
            emp_speeds = np.array(consumption_lut.iloc[:, 0]).astype(np.float64)
            emp_data = np.array(consumption_lut.iloc[:, 1:]).astype(np.float64)

        all_gn_buses = (
            session.query(VehicleType).filter(VehicleType.name_short == "GN").all()
        )
        for vehicle_type in all_gn_buses:
            consumption_lut = vehicle_type.vehicle_classes[0].consumption_lut
            new_coordinates = []
            new_values = []

            # Update the LUT with the empirical data
            incline = 0.0
            level_of_loading = 0.5
            for i, temperature in enumerate(emp_temperatures):
                for j, speed in enumerate(emp_speeds):
                    # Interpolate the empirical data to the coordinates
                    consumption = emp_data[i, j]
                    if not np.isnan(consumption):
                        new_coordinates.append(
                            (incline, temperature, level_of_loading, speed)
                        )
                        new_values.append(consumption)
            consumption_lut.data_points = [
                [float(value) for value in coord] for coord in new_coordinates
            ]
            consumption_lut.values = [float(value) for value in new_values]
    except:
        session.rollback()
        raise
    finally:
        session.commit()
