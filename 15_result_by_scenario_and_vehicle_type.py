import os
from datetime import datetime, timedelta
from typing import List, Any, Dict

import numpy as np
import pandas as pd
import pytz
from eflips.model import *
from sqlalchemy import create_engine, or_, and_
from sqlalchemy.orm import Session

tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)


def calculate_total_energy_charged(
    events: List[Event], simulation_start, simulation_end
) -> float:
    total_soc_charged = 0
    for event in events:
        if event.timeseries is None:
            timestamps = [event.time_start.timestamp(), event.time_end.timestamp()]
            socs = [event.soc_start, event.soc_end]
        else:
            timeseries = event.timeseries
            timedeltas = [datetime.fromisoformat(t) for t in timeseries["time"]]
            timestamps = [td.timestamp() for td in timedeltas]
            socs = timeseries["soc"]

        real_start = max(simulation_start.timestamp(), timestamps[0])
        real_start_soc = np.interp(real_start, timestamps, socs)
        real_end = min(simulation_end.timestamp(), timestamps[-1])
        real_end_soc = np.interp(real_end, timestamps, socs)
        total_soc_charged += real_end_soc - real_start_soc

    return total_soc_charged


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
    # - the count of vehicles of each type
    # - the total mileage of each vehicle type
    # - the total mileage of opportunity charged schedules of each vehicle type
    # - the efficiency of each vehicle type

    # - total energy being charged in the depot
    # - total energy being charged on the road
    # - total energy being charged in each depot

    writer = pd.ExcelWriter(
        f"15_results_by_scenario_and_vehicle_type.xlsx", engine="xlsxwriter"
    )
    for scenario_name_short in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario)
            .filter(Scenario.name_short == scenario_name_short)
            .one()
        )

        results: List[Dict[str, Any]] = []

        # Vehicle Types
        vehicle_types = (
            session.query(VehicleType)
            .filter(VehicleType.scenario_id == scenario.id)
            .all()
        )
        assert len(vehicle_types) > 0, "No vehicle types found for scenario"

        # Vehicle counts

        for vehicle_type in vehicle_types:

            vehicle_query = session.query(Vehicle).filter(
                Vehicle.scenario_id == scenario.id,
                Vehicle.vehicle_type_id == vehicle_type.id,
            )

            # Vehicle count
            vehicle_count_this_type = vehicle_query.count()
            # Mileage

            mileage_this_type = 0
            mileage_opportunity_charged_this_type = 0

            total_trip_time_this_type = 0
            total_passenger_time_this_type = 0

            total_depot_charged_energy_this_type = 0
            total_opportunity_charged_energy_this_type = 0
            vehicles = vehicle_query.all()
            for vehicle in vehicles:
                # Vehicle layer
                rotations_this_vehicle = vehicle.rotations
                for rotation in rotations_this_vehicle:
                    # Rotation layer
                    trips_this_rotation = rotation.trips
                    mileage_this_type += sum(
                        [trip.route.distance for trip in trips_this_rotation]
                    )

                    # Mileage of opportunity charged schedules
                    if scenario.name_short != "DEP":
                        mileage_opportunity_charged_this_type += sum(
                            [
                                trip.route.distance
                                for trip in trips_this_rotation
                                if rotation.allow_opportunity_charging
                            ]
                        )

                    # Efficiency
                    total_trip_time_this_type += (
                        trips_this_rotation[-1].arrival_time
                        - trips_this_rotation[0].departure_time
                    ).total_seconds()
                    total_passenger_time_this_type += sum(
                        [
                            (trip.arrival_time - trip.departure_time).total_seconds()
                            for trip in trips_this_rotation
                            if trip.trip_type == TripType.PASSENGER
                        ]
                    )

                # All the charging events of the vehicle
            opp_charging_events = (
                session.query(Event)
                .filter(
                    Event.vehicle_type_id == vehicle_type.id,
                    Event.event_type == EventType.CHARGING_OPPORTUNITY,
                    Event.scenario_id == scenario.id,
                    or_(
                        and_(
                            Event.time_start >= START_OF_SIMULATION,
                            Event.time_start <= END_OF_SIMULATION,
                        ),
                        and_(
                            Event.time_end >= START_OF_SIMULATION,
                            Event.time_end <= END_OF_SIMULATION,
                        ),
                    ),
                )
                .all()
            )
            total_opportunity_charged_energy_this_type += (
                calculate_total_energy_charged(
                    opp_charging_events, START_OF_SIMULATION, END_OF_SIMULATION
                )
            ) * vehicle_type.battery_capacity

            depot_charging_events = (
                session.query(Event)
                .filter(
                    Event.vehicle_type_id == vehicle_type.id,
                    Event.event_type == EventType.CHARGING_DEPOT,
                    Event.scenario_id == scenario.id,
                    or_(
                        and_(
                            Event.time_start >= START_OF_SIMULATION,
                            Event.time_start <= END_OF_SIMULATION,
                        ),
                        and_(
                            Event.time_end >= START_OF_SIMULATION,
                            Event.time_end <= END_OF_SIMULATION,
                        ),
                    ),
                )
                .all()
            )
            total_depot_charged_energy_this_type += (
                calculate_total_energy_charged(
                    depot_charging_events, START_OF_SIMULATION, END_OF_SIMULATION
                )
                * vehicle_type.battery_capacity
            )

            # Charged energy for each depot
            depots = scenario.depots
            dict_depot_energy_this_type = {}
            for depot in depots:
                depot_charging_events = (
                    session.query(Event)
                    .join(Area)
                    .filter(
                        Event.vehicle_type_id == vehicle_type.id,
                        Event.event_type == EventType.CHARGING_DEPOT,
                        Event.scenario_id == scenario.id,
                        or_(
                            and_(
                                Event.time_start >= START_OF_SIMULATION,
                                Event.time_start <= END_OF_SIMULATION,
                            ),
                            and_(
                                Event.time_end >= START_OF_SIMULATION,
                                Event.time_end <= END_OF_SIMULATION,
                            ),
                        ),
                        Area.depot_id == depot.id,
                    )
                    .all()
                )
                dict_depot_energy_this_type[
                    "Geladene Energie im " + depot.name + " / kWh "
                ] = (
                    calculate_total_energy_charged(
                        depot_charging_events, START_OF_SIMULATION, END_OF_SIMULATION
                    )
                    * vehicle_type.battery_capacity
                )

            result_this_type = {
                "Szenario": scenario.name,
                "Fahrzeugtyp": vehicle_type.name,
                "Fahrzeuganzahl": vehicle_count_this_type,
                "Fahrzeugkilometer": mileage_this_type / 1000,
                "Fahrzeugkilometer auf Endhaltestellenlade-Umläufen": mileage_opportunity_charged_this_type
                / 1000,
                "Zeitliche Fahrplanwirkungsgrad": total_passenger_time_this_type
                / total_trip_time_this_type,
                "Passagierte Zeit (s)": total_passenger_time_this_type,
                "Gesamtzeit (s)": total_trip_time_this_type,
                "Geladene Energie im Depot / kWh": total_depot_charged_energy_this_type,
                "Geladene Energie unterwegs / KWh": total_opportunity_charged_energy_this_type,
            }
            result_this_type.update(dict_depot_energy_this_type)

            results.append(result_this_type)

        df_result_vehicle_type = pd.DataFrame(results)

        # Sort rows by "Fahrzeugtyp
        df_result_vehicle_type = df_result_vehicle_type.sort_values(by="Fahrzeugtyp")

        df_result_vehicle_type.to_excel(writer, sheet_name=scenario.name_short)
    writer.close()
