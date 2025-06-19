#! /usr/bin/env python3
import os
import warnings
from typing import Dict, List, Any

import pandas as pd
from eflips.model import *
from eflips.model import ConsistencyWarning
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

warnings.simplefilter("ignore", ConsistencyWarning)
USE_SIMBA_CONSUMPTION = True
PARALLELISM = True

if __name__ == "__main__":
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    # Load the first depot from the database
    depot = session.query(Depot).first()
    plan = depot.default_plan

    for entry in plan.processes:
        entry: Process
        print(f"Process: {entry.name}")
        print(f"Duration: {entry.duration}")
        print(f"Power: {entry.electric_power}")

    # Create a dataframe of vehicle counts and depot charging area space for each scenario and vehicle type
    SCENARIO_NAMES = ["OU", "DEP", "TERM", "OU_DIESEL"]

    results: List[Dict[str, Any]] = []

    for scenario_name in SCENARIO_NAMES:
        scenario = (
            session.query(Scenario).filter(Scenario.name_short == scenario_name).one()
        )
        for vehicle_type in (
            session.query(VehicleType)
            .filter(VehicleType.scenario_id == scenario.id)
            .all()
        ):
            all_areas = (
                session.query(Area)
                .filter(Area.scenario_id == scenario.id)
                .filter(Area.vehicle_type_id == vehicle_type.id)
                .all()
            )
            charging_areas = []
            total_capacity = 0
            # Charging areas are ones which have a process with electric power and no duration
            for area in all_areas:
                for process in area.processes:
                    if process.electric_power and not process.duration:
                        charging_areas.append(area)

                        # Used capacity is the maximum subloc_no of events at this area
                        used_capacity = (
                            session.query(func.max(Event.subloc_no))
                            .filter(Event.area_id == area.id)
                            .scalar()
                        )

                        total_capacity += used_capacity

                        break

            vehicle_count = (
                session.query(Vehicle)
                .filter(
                    Vehicle.scenario_id == scenario.id,
                    Vehicle.vehicle_type_id == vehicle_type.id,
                )
                .count()
            )
            results.append(
                {
                    "scenario": scenario_name,
                    "vehicle_type": vehicle_type.name_short,
                    "vehicle_count": vehicle_count,
                    "total_capacity": total_capacity,
                }
            )

    df = pd.DataFrame(results)

    # Rename som things to be clearer
    vehicle_name_dict = {
        "EN": "single decker",
        "GN": "articulated bus",
        "DD": "double decker",
    }
    df["vehicle_type"] = df["vehicle_type"].map(vehicle_name_dict)

    scenario_name_dict = {
        "OU": "existing blocks",
        "DEP": "depot charging",
        "TERM": "small batteries",
        "OU_DIESEL": "existing blocks (diesel)",
    }
    df["scenario"] = df["scenario"].map(scenario_name_dict)

    df.to_excel("20_vehicle_counts.xlsx", index=False)
