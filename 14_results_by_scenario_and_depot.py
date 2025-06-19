import os
from datetime import datetime, timedelta
from typing import List, Any, Dict

import pandas as pd
import pytz
from eflips.eval.output.prepare import power_and_occupancy
from eflips.model import *
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

    # Create a dataframe of for each scenario and depot
    # The info we care about is
    # - capacities of cleaning areas
    # - capacities of line charging areas each vehicle type
    # - peak utilization of direct charging areas each vehicle type
    # - peak power consumption
    # - peak utilization of waiting areas
    # - distribution of the rotation-vehicle type of each depot

    writer = pd.ExcelWriter(
        f"14_results_by_scenario_and_depot.xlsx", engine="xlsxwriter"
    )
    for scenario_name_short in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario)
            .filter(Scenario.name_short == scenario_name_short)
            .one()
        )

        results: List[Dict[str, Any]] = []

        # Get the depot
        depots = session.query(Depot).filter(Depot.scenario_id == scenario.id).all()
        if len(depots) == 0:
            print(f"No depot found for scenario {scenario_name_short}")
            continue

        for depot in depots:
            # Get the capacities of the cleaning areas
            cleaning_area = (
                session.query(Area)
                .filter(Area.depot_id == depot.id)
                .join(AssocAreaProcess)
                .join(Process)
                .filter(Process.name.like("%Arrival Cleaning%"))
                .all()
            )

            if len(cleaning_area) == 1:
                cleaning_area_capacity = cleaning_area[0].capacity
            else:
                raise ValueError("There are more than one cleaning areas")
                cleaning_area_capacity_dict = {}
                for area in cleaning_area:
                    vehicle_type = (
                        area.vehicle_type.name_short if area.vehicle_type else "All"
                    )
                    capacity = area.capacity
                    cleaning_area_capacity_dict[vehicle_type] = capacity

                cleaning_area_capacity = str(cleaning_area_capacity_dict)

            # Get the capacities of the line charging areas

            dict_line_charging_area_capacity = {}
            line_charging_area = (
                session.query(Area)
                .filter(Area.depot_id == depot.id, Area.area_type == AreaType.LINE)
                .join(AssocAreaProcess)
                .join(Process)
                .filter(Process.name.like("%Charging%"))
                .all()
            )
            for area in line_charging_area:
                capacity = area.capacity
                vehicle_type = area.vehicle_type.name_short

                if vehicle_type in dict_line_charging_area_capacity:
                    dict_line_charging_area_capacity[vehicle_type] += capacity
                else:
                    dict_line_charging_area_capacity[vehicle_type] = capacity

            # Get the peak utilization of the direct charging areas

            dict_direct_charging_area_utilization = {}
            direct_charging_area = (
                session.query(Area)
                .filter(
                    Area.depot_id == depot.id, Area.area_type == AreaType.DIRECT_ONESIDE
                )
                .join(AssocAreaProcess)
                .join(Process)
                .filter(Process.name.like("%Charging"))
                .all()
            )

            for area in direct_charging_area:
                vehicle_type = area.vehicle_type.name_short
                try:
                    df_power_occupancy = power_and_occupancy(
                        area.id,
                        session,
                        sim_start_time=START_OF_SIMULATION,
                        sim_end_time=END_OF_SIMULATION,
                    )
                    peak_utilization = df_power_occupancy["occupancy_total"].max()
                except ValueError as e:
                    print("There is no direct charging areas")
                    peak_utilization = 0

                if vehicle_type in dict_direct_charging_area_utilization:
                    dict_direct_charging_area_utilization[vehicle_type] = max(
                        dict_direct_charging_area_utilization[vehicle_type],
                        peak_utilization,
                    )
                else:
                    dict_direct_charging_area_utilization[vehicle_type] = (
                        peak_utilization
                    )

            # Get the peak power consumption of the depot
            total_areas = depot.areas
            assert len(total_areas) > 0, "No areas found for depot"

            peak_power = power_and_occupancy(
                [area.id for area in total_areas],
                session,
                sim_start_time=START_OF_SIMULATION,
                sim_end_time=END_OF_SIMULATION
                - timedelta(
                    days=2
                ),  # We exclude the last two days, to work around a bug
            )["power"].max()

            # Get the peak utilization of the waiting areas
            for area in total_areas:
                if area.processes == []:
                    waiting_area = area
                    break

            try:

                peak_waiting_area_utilization = power_and_occupancy(
                    waiting_area.id,
                    session,
                    sim_start_time=START_OF_SIMULATION,
                    sim_end_time=END_OF_SIMULATION,
                )["occupancy_total"].max()

            except ValueError as e:
                print("No waiting events found")
                peak_waiting_area_utilization = 0

            # Distribution of the rotation-vehicle type of each depot

            rotations = scenario.rotations

            dict_rotation_vehicle_type = {}

            depot_station = depot.station

            for rotation in rotations:

                if rotation.trips[0].route.departure_station == depot_station:

                    vehicle_type = rotation.vehicle_type.name_short

                    if vehicle_type in dict_rotation_vehicle_type:
                        dict_rotation_vehicle_type[vehicle_type] += 1
                    else:
                        dict_rotation_vehicle_type[vehicle_type] = 1

            ######### Find the average charging power for each charging event #########

            # Get the charging events
            charging_events_depot = (
                session.query(Event)
                .filter(Event.scenario_id == scenario.id)
                .filter(Event.event_type == EventType.CHARGING_DEPOT)
                .join(Area)
                .filter(Area.depot_id == depot.id)
                .all()
            )

            POWER_BRACKETS = [0, 50, 100, 150, 200, 250]

            charging_event_durations = dict.fromkeys(POWER_BRACKETS, 0)
            for charging_event in charging_events_depot:
                charging_event_duration = (
                    charging_event.time_end - charging_event.time_start
                ).total_seconds()

                delta_soc = charging_event.soc_end - charging_event.soc_start
                energy = delta_soc * charging_event.vehicle_type.battery_capacity
                average_power = energy / (charging_event_duration / 3600)
                if average_power < 50:
                    charging_event_durations[0] += charging_event_duration
                elif average_power < 100:
                    charging_event_durations[50] += charging_event_duration
                elif average_power < 150:
                    charging_event_durations[100] += charging_event_duration
                elif average_power < 200:
                    charging_event_durations[150] += charging_event_duration
                elif average_power < 250:
                    charging_event_durations[200] += charging_event_duration
                else:
                    charging_event_durations[250] += charging_event_duration

            POWER_BRACKETS_OPP = [0, 90, 180, 270, 360]
            charging_event_durations_opp = dict.fromkeys(POWER_BRACKETS_OPP, 0)
            for charging_event in (
                session.query(Event)
                .filter(Event.scenario_id == scenario.id)
                .filter(Event.event_type == EventType.CHARGING_OPPORTUNITY)
                .all()
            ):
                charging_event_duration = (
                    charging_event.time_end - charging_event.time_start
                ).total_seconds() - 60  # The dead time is 60 seconds

                if charging_event_duration == 0:
                    continue

                delta_soc = charging_event.soc_end - charging_event.soc_start
                energy = delta_soc * charging_event.vehicle_type.battery_capacity
                average_power = energy / (charging_event_duration / 3600)
                if average_power < 90:
                    charging_event_durations_opp[0] += charging_event_duration
                elif average_power < 180:
                    charging_event_durations_opp[90] += charging_event_duration
                elif average_power < 270:
                    charging_event_durations_opp[180] += charging_event_duration
                elif average_power < 360:
                    charging_event_durations_opp[270] += charging_event_duration
                else:
                    charging_event_durations_opp[360] += charging_event_duration

            # Create a proper row for the dataframe
            # The dict values will be converted to multiple entries in the dataframe
            result_depot = {
                "Szenario": scenario.name,
                "Depot": depot.name,
                "Kapazität Reinigungsfläche": int(cleaning_area_capacity),
                "Spitzenauslastung Wartezone": int(peak_waiting_area_utilization),
                "Elektrische Anschlusskapazität": peak_power,
                "Depot-Laden 0-50 kW": charging_event_durations[0],
                "Depot-Laden 50-100 kW": charging_event_durations[50],
                "Depot-Laden 100-150 kW": charging_event_durations[100],
                "Depot-Laden 150-200 kW": charging_event_durations[150],
                "Depot-Laden 200-250 kW": charging_event_durations[200],
                "Depot-Laden >250 kW": charging_event_durations[250],
                "Opportunity-Laden 0-90 kW": charging_event_durations_opp[0],
                "Opportunity-Laden 90-180 kW": charging_event_durations_opp[90],
                "Opportunity-Laden 180-270 kW": charging_event_durations_opp[180],
                "Opportunity-Laden 270-360 kW": charging_event_durations_opp[270],
                "Opportunity-Laden >360 kW": charging_event_durations_opp[360],
            }
            for key, value in dict_rotation_vehicle_type.items():
                result_depot[f"Anzahl Umläufe {key}"] = value

            for key, value in dict_line_charging_area_capacity.items():
                if key == "DD" and f"Anzahl Umläufe {key}" not in result_depot:
                    result_depot[f"Kapazität Ladezone Linienabstellung {key}"] = 0
                else:
                    result_depot[f"Kapazität Ladezone Linienabstellung {key}"] = value

            for key, value in dict_direct_charging_area_utilization.items():
                result_depot[
                    f"Spitzenauslastung Ladezone Fischgrätenabstellung {key}"
                ] = value

            results.append(result_depot)

        df_result_depot = pd.DataFrame(results)

        # sort columns alphabetically
        df_result_depot = df_result_depot.reindex(
            sorted(df_result_depot.columns), axis=1
        )
        # Sort rows by "Depot"
        df_result_depot = df_result_depot.sort_values(by="Depot")

        df_result_depot.to_excel(writer, sheet_name=scenario.name_short)
    writer.close()
