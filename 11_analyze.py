#! /usr/bin/env python3


import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from eflips.eval.output.prepare import power_and_occupancy
from eflips.model import *
from eflips.model import ConsistencyWarning
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

USE_SIMBA_CONSUMPTION = False
warnings.simplefilter("ignore", ConsistencyWarning)

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    SCENARIO_NAMES = ["OU", "DEP", "TERM"]

    # Create a dataframe of for each scenario and vehicle type
    # The info we care about is
    # - count of vehicles
    results_by_scenario_and_vehicle_type = []

    # Some of the info we aggregate over the scenarios
    # - total number of vehicles
    # - total number of electrified stations
    # - Total number of charging places at the termini (obtained thrpugh eflips-eval)
    # - Peak power consumption (obtained through eflips-eval)
    # - Average energy consumption (obtained through eflips-eval)
    # - Total number of rotations with SOC < 0
    results_by_scenario = []

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).one()
        )

        total_number_of_vehicles = (
            session.query(Vehicle).filter(Vehicle.scenario_id == scenario.id).count()
        )
        total_number_of_electrified_stations = (
            session.query(Station)
            .filter(Station.scenario_id == scenario.id, Station.is_electrified == True)
            .count()
        )

        # Get a power and occupation dataframe for each electrified station
        terminus_no_charging_places = 0
        terminus_peak_power = 0
        terminus_avg_power = []
        electrified_station_ids = (
            session.query(Station.id)
            .filter(Station.scenario_id == scenario.id, Station.is_electrified == True)
            .filter(Station.charge_type == ChargeType.OPPORTUNITY)
            .all()
        )
        # The simulation runs from local midnight on the first day to local midnight on the last day
        tz = pytz.timezone("Europe/Berlin")
        sim_start = tz.localize(datetime(2023, 7, 3))
        sim_end = sim_start + timedelta(days=7)

        for electrified_station_id in electrified_station_ids:
            power_and_occupancy_df = power_and_occupancy(
                area_id=None, session=session, station_id=electrified_station_id[0]
            )
            #     The columns are:
            #     - time: the time at which the data was recorded
            #     - power: the summed power consumption of the area(s) at the given time
            #     - occupancy: the summed occupancy of the area(s) at the given time
            terminus_peak_power = max(
                terminus_peak_power, power_and_occupancy_df["power"].max()
            )
            terminus_no_charging_places += power_and_occupancy_df[
                "occupancy_total"
            ].max()

            terminus_avg_power.append(
                power_and_occupancy_df[
                    (power_and_occupancy_df["time"] >= sim_start)
                    & (power_and_occupancy_df["time"] < sim_end)
                ]["power"].mean()
            )

        # Get power and occupancy for the depot
        depot_peak_power = 0
        depot_avg_power = []
        depot_ids = (
            session.query(Depot.id).filter(Depot.scenario_id == scenario.id).all()
        )

        total_capacity = 0
        for depot_id in depot_ids:

            areas = session.query(Area.id).filter(Area.depot_id == depot_id[0]).all()
            area_ids = [area.id for area in areas]
            power_and_occupancy_df = power_and_occupancy(
                area_id=area_ids, session=session
            )
            depot_peak_power = max(
                depot_peak_power, power_and_occupancy_df["power"].max()
            )
            depot_avg_power.append(
                power_and_occupancy_df[
                    (power_and_occupancy_df["time"] >= sim_start)
                    & (power_and_occupancy_df["time"] < sim_end)
                ]["power"].mean()
            )

            # If the area offers a charging process, find its utilization
            depot_capacity = 0
            for area_id in area_ids:
                area = session.query(Area).filter(Area.id == area_id).one()
                area: Area
                if "Charging" in [process.name for process in area.processes]:
                    match area.area_type:
                        case AreaType.DIRECT_ONESIDE | AreaType.DIRECT_TWOSIDE:
                            # Since the slots are filled sequentially, we can find the highest subloc_no of a charging event
                            # and assume that the area is full up to that subloc_no
                            highest_subloc_no = (
                                session.query(func.max(Event.subloc_no))
                                .filter(Event.area_id == area_id)
                                .scalar()
                            )
                            if highest_subloc_no is not None:
                                depot_capacity += highest_subloc_no
                        case AreaType.LINE:
                            # If there is any event in the area, it is full. add its capacity to the depot capacity
                            if (
                                session.query(Event)
                                .filter(Event.area_id == area_id)
                                .count()
                                > 0
                            ):
                                depot_capacity += area.capacity
                        case _:
                            raise ValueError(f"Unknown area type {area.area_type}")

            total_capacity += depot_capacity

        # Get the number of rotations with SOC < 0
        rotations_q = (
            session.query(Rotation)
            .join(Trip)
            .join(Event)
            .filter(Rotation.scenario_id == scenario.id)
            .filter(Event.event_type == EventType.DRIVING)
            .filter(Event.soc_end < 0)
        )
        number_of_rotations_below_zero = rotations_q.count()

        results_by_scenario.append(
            {
                "scenario": scenario.name,
                "total_number_of_vehicles": total_number_of_vehicles,
                "total_number_of_electrified_stations": total_number_of_electrified_stations,
                "terminus_no_charging_places": terminus_no_charging_places,
                "terminus_peak_power": terminus_peak_power,
                "terminus_avg_power": np.mean(terminus_avg_power),
                "depot_peak_power": depot_peak_power,
                "depot_avg_power": np.mean(depot_avg_power),
                "number_of_rotations_below_zero": number_of_rotations_below_zero,
                "total_charge_capacity": total_capacity,
            }
        )
    results_df = pd.DataFrame(results_by_scenario)
    results_df.to_excel("11_results_by_scenario.xlsx")
