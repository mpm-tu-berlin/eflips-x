import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from eflips.eval.output.prepare import power_and_occupancy
from eflips.model import *
from pyomo.common.dependencies import scipy
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    SCENARIO_NAMES = ["OU", "DEP", "TERM"]
    co2_totals = {}

    for scenario_short_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario)
            .filter(Scenario.name_short == scenario_short_name)
            .one()
        )

        all_areas = session.query(Area).filter(Area.scenario == scenario).all()
        all_electrified_stations = (
            session.query(Station)
            .filter(Station.scenario_id == scenario.id)
            .filter(Station.is_electrified == True)
            .all()
        )

        power_and_occupancy_depot = power_and_occupancy(
            area_id=[area.id for area in all_areas],
            session=session,
            sim_start_time=START_OF_SIMULATION,
            sim_end_time=END_OF_SIMULATION,
        )

        if len(all_electrified_stations) == 0:
            print(f"No electrified stations in scenario {scenario.name_short}")
            # Create a dummy dataframe with the same "time" as the depot one
            # and a zero "power" column
            power_and_occupancy_station = pd.DataFrame(
                {
                    "time": power_and_occupancy_depot["time"],
                    "power": [0] * len(power_and_occupancy_depot["time"]),
                }
            )
        else:
            power_and_occupancy_station = power_and_occupancy(
                area_id=None,
                station_id=[station.id for station in all_electrified_stations],
                session=session,
                sim_start_time=START_OF_SIMULATION,
                sim_end_time=END_OF_SIMULATION,
            )

        # Extend the station power dataframe to the same time as the depot
        depot_time = power_and_occupancy_depot["time"]
        station_power = power_and_occupancy_station["power"]
        station_power_interpolated = np.interp(
            depot_time, power_and_occupancy_station["time"], station_power
        )
        power_and_occupancy_station = pd.DataFrame(
            {
                "time": depot_time,
                "power": station_power_interpolated,
            }
        )

        # Load the CO2 intendity dataframe from
        CO2_INPUT_FILE = "co₂-emissionen_der_stromerzeugung.csv"
        co2_intensity = pd.read_csv(CO2_INPUT_FILE, usecols=[0, 2])
        co2_intensity.rename(
            columns={
                "date_id": "time",
                "CO₂-Emissionsfaktor des Strommix": "co2_intensity",
            },
            inplace=True,
        )
        # Change the values from implicit localtimezone to explicit Europe/Berlin
        tz = pytz.timezone("Europe/Berlin")
        for entry in co2_intensity["time"]:
            co2_intensity["time"] = co2_intensity["time"].replace(
                entry, tz.localize(datetime.strptime(entry, "%Y-%m-%dT%H:%M:%S"))
            )

        # For each entry, integrate the sum of the power consumption
        power_total = (
            power_and_occupancy_depot["power"] + power_and_occupancy_station["power"]
        )
        power_time = [p.timestamp() for p in power_and_occupancy_depot["time"]]
        cumulative_energy = scipy.integrate.cumulative_trapezoid(
            power_total, power_time, initial=0
        )  # unit is kW * s
        cumulative_energy = cumulative_energy / 3600  # unit is kW * h

        # Sample the cumulative energy at the same time as the CO2 intensity
        cumulative_energy_sampled = np.interp(
            [a.timestamp() for a in co2_intensity["time"]],
            power_time,
            cumulative_energy,
        )
        co2_intensity["energy_in_hour"] = np.diff(cumulative_energy_sampled, prepend=0)

        # Multiply the CO2 intensity with the cumulative energy
        co2_intensity["co2_emission"] = (
            co2_intensity["energy_in_hour"] * co2_intensity["co2_intensity"]
        )

        # The Co" total mass (in tons) is the sum of the CO2 emissions divided by 1e6
        co2_total_mass = np.sum(co2_intensity["co2_emission"]) / 1e6
        co2_totals[scenario_short_name] = co2_total_mass

        if scenario_short_name == "TERM":
            from matplotlib import pyplot as plt

            fig, ax = plt.subplots(nrows=2, figsize=(12, 6))
            ax[0].plot(
                power_and_occupancy_depot["time"],
                power_and_occupancy_depot["power"],
                label="Depot",
            )
            ax[0].plot(
                power_and_occupancy_station["time"],
                power_and_occupancy_station["power"],
                label="Endhaltestellen",
            )
            ax[0].set_xlabel("Datum")
            ax[0].set_ylabel("Leistung [kW]")
            ax[0].set_title(f"Leistungsaufnahme (Szenario Endhaltestellen)")
            ax[0].legend()

            # Ax[1] is the CO2 intensity
            ax[1].plot(
                co2_intensity["time"],
                co2_intensity["co2_intensity"],
                label="CO2 Emission factor [gCO2/kWh]",
            )
            ax[1].set_xlabel("Datum")
            ax[1].set_ylabel("CO₂-Emissionsfaktor [gCO2/kWh]")
            ax[1].set_title(f"CO₂-Emissionsfaktor des Strommix")
            plt.tight_layout()
            plt.savefig(f"11_power_consumption_{scenario.name_short}.png")
    print(co2_totals)
