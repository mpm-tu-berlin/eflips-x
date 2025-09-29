#!/usr/bin/env python3

import os
from datetime import datetime, timedelta
from typing import List, Optional

import pytz
from eflips.eval.output.prepare import power_and_occupancy
from eflips.model import Area, Station, ChargeType, Scenario
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm

# Configuration
tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)
TEMPORAL_RESOLUTION = 60


def has_charging_processes(area: Area) -> bool:
    """
    Check if an area has any charging processes.
    """
    return any([p.electric_power is not None for p in area.processes])


def get_max_occupancy_for_area(area: Area, session: Session) -> Optional[int]:
    """
    Get the maximum occupancy for a given area.
    """
    try:
        df = power_and_occupancy(
            area_id=[area.id],
            session=session,
            station_id=None,
            temporal_resolution=TEMPORAL_RESOLUTION,
            sim_start_time=START_OF_SIMULATION,
            sim_end_time=END_OF_SIMULATION,
        )
        if df.empty or "occupancy_total" not in df.columns:
            print(f"Warning: No occupancy data for area {area.id} ({area.name})")
            return None

        max_occupancy = max(df["occupancy_total"])
        return int(max_occupancy)
    except Exception as e:
        print(f"Error getting occupancy for area {area.id} ({area.name}): {e}")
        return None


def get_max_occupancy_for_station(station: Station, session: Session) -> Optional[int]:
    """
    Get the maximum occupancy for a given terminus station.
    """
    try:
        df = power_and_occupancy(
            area_id=None,
            session=session,
            station_id=station.id,
            temporal_resolution=TEMPORAL_RESOLUTION,
            sim_start_time=START_OF_SIMULATION,
            sim_end_time=END_OF_SIMULATION,
        )
        if df.empty or "occupancy_total" not in df.columns:
            print(f"Warning: No occupancy data for station {station.id} ({station.name})")
            return None

        max_occupancy = max(df["occupancy_total"])
        return int(max_occupancy)
    except Exception as e:
        print(f"Error getting occupancy for station {station.id} ({station.name}): {e}")
        return None


def update_area_and_termini_sizes(session: Session):
    """
    Update area and termini capacities based on actual maximum occupancy.
    """
    updates_made = []

    # Process areas with charging processes
    print("Processing areas with charging processes...")
    areas_with_charging = session.query(Area).filter(
        Area.processes.any()
    ).all()

    charging_areas = [area for area in areas_with_charging if has_charging_processes(area)]

    for area in tqdm(charging_areas, desc="Charging areas"):
        max_occupancy = get_max_occupancy_for_area(area, session)
        if max_occupancy is not None and max_occupancy != area.capacity:
            old_capacity = area.capacity
            area.capacity = max_occupancy
            updates_made.append({
                'type': 'area',
                'id': area.id,
                'name': area.name,
                'old_capacity': old_capacity,
                'new_capacity': max_occupancy
            })
            print(f"Updated area {area.id} ({area.name}): {old_capacity} -> {max_occupancy}")

    # Process termini (stations with opportunity charging)
    print("\nProcessing termini...")
    termini = session.query(Station).filter(
        Station.is_electrified == True,
        Station.charge_type == ChargeType.OPPORTUNITY
    ).all()

    for station in tqdm(termini, desc="Termini"):
        max_occupancy = get_max_occupancy_for_station(station, session)
        if max_occupancy is not None and max_occupancy != station.amount_charging_places:
            old_capacity = station.amount_charging_places
            station.amount_charging_places = max_occupancy
            updates_made.append({
                'type': 'station',
                'id': station.id,
                'name': station.name,
                'old_capacity': old_capacity,
                'new_capacity': max_occupancy
            })
            print(f"Updated station {station.id} ({station.name}): {old_capacity} -> {max_occupancy}")

    return updates_made


def main():
    """
    Main function to update area and termini capacities.
    """
    # Check for DATABASE_URL environment variable
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        database_url = "postgresql://arbeit:moosemoose@localhost/eflips_testing"
        print(f"Using default DATABASE_URL: {database_url}")
    else:
        print(f"Using DATABASE_URL: {database_url}")

    # Create engine and session
    engine = create_engine(database_url)
    session = Session(engine)

    try:
        print("Starting capacity update based on actual utilization...")
        updates_made = update_area_and_termini_sizes(session)

        print(f"\nSummary of updates made:")
        print(f"Total updates: {len(updates_made)}")

        area_updates = [u for u in updates_made if u['type'] == 'area']
        station_updates = [u for u in updates_made if u['type'] == 'station']

        print(f"Area updates: {len(area_updates)}")
        print(f"Station updates: {len(station_updates)}")

        if updates_made:
            print("\nDetailed updates:")
            for update in updates_made:
                print(f"{update['type'].capitalize()} {update['id']} ({update['name']}): "
                      f"{update['old_capacity']} -> {update['new_capacity']}")
        else:
            print("No updates were necessary - all capacities already match actual utilization.")

        print("\nNOTE: All changes are rolled back. Review the results and modify the script to commit if satisfied.")

    finally:
        # Always rollback - user will change this to commit when satisfied
        session.commit()
        session.close()
        print("Database session rolled back.")


if __name__ == "__main__":
    main()