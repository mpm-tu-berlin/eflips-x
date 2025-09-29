#!/usr/bin/env python3
"""
Utility script to extract scenario data from the eflips-model database.

This script extracts:
1. Vehicle type data: battery capacity, short name, vehicle count, revenue kilometers
2. Use phase data: total energy consumption, revenue kilometers, total kilometers
3. Infrastructure data: charging spots and peak power by location (depot/electrified stations)
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from eflips.model import (
    Scenario, VehicleType, Vehicle, Trip, TripType,
    Station, Area, ChargingPointType, Route, Rotation
)


@dataclass
class VehicleManufactureData:
    """Data for vehicle manufacture phase (per vehicle type)"""
    vehicle_type_short_name: str
    battery_capacity: float  # kWh
    vehicle_count: int
    revenue_kilometers: float  # km


@dataclass
class UsePhaseData:
    """Data for use phase (total across all vehicles)"""
    total_energy_consumption: float  # kWh
    total_revenue_kilometers: float  # km
    total_kilometers: float  # km


@dataclass
class InfrastructureData:
    """Data for infrastructure (per location)"""
    location_name: str
    location_type: str  # "depot" or "electrified_terminus"
    charging_spots: int
    total_peak_power: float  # kW


@dataclass
class ScenarioReport:
    """Complete scenario report"""
    vehicle_manufacture: List[VehicleManufactureData]
    use_phase: UsePhaseData
    infrastructure: List[InfrastructureData]


def extract_scenario_data(scenario_id: int, database_url: str) -> tuple[ScenarioReport, str]:
    """
    Extract scenario data from the database.

    Args:
        scenario_id: The ID of the scenario to extract
        database_url: Database connection URL

    Returns:
        ScenarioReport containing all extracted data
    """
    # Create database connection
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Verify scenario exists
        scenario = session.query(Scenario).filter(Scenario.id == scenario_id).first()
        if not scenario:
            raise ValueError(f"Scenario with ID {scenario_id} not found")

        # 1. Extract vehicle manufacture data
        vehicle_manufacture_data = []

        for vehicle_type in session.query(VehicleType).filter(VehicleType.scenario_id == scenario_id):
            # Count vehicles of this type
            vehicle_count = session.query(Vehicle).filter(
                Vehicle.scenario_id == scenario_id,
                Vehicle.vehicle_type_id == vehicle_type.id
            ).count()

            # Calculate revenue kilometers for this vehicle type
            # Join Trip -> Route to get distance, and Trip -> Rotation to filter by vehicle type
            revenue_km_query = session.query(func.sum(Route.distance / 1000.0)).select_from(Trip).join(Route).join(Rotation).filter(
                Trip.scenario_id == scenario_id,
                Trip.trip_type == TripType.PASSENGER,
                Rotation.vehicle_type_id == vehicle_type.id
            )
            revenue_km_weekly = revenue_km_query.scalar() or 0.0
            # Scale from one week to one year (multiply by 52)
            revenue_km = revenue_km_weekly * 52

            vehicle_manufacture_data.append(VehicleManufactureData(
                vehicle_type_short_name=vehicle_type.name_short or vehicle_type.name,
                battery_capacity=vehicle_type.battery_capacity,
                vehicle_count=vehicle_count,
                revenue_kilometers=revenue_km
            ))

        # 2. Extract use phase data
        # Energy consumption constants per vehicle type (kWh/km)
        energy_consumption_rates = {
            "EN": 1.48,
            "GN": 2.16,
            "DD": 2.16
        }

        total_energy_consumption = 0.0

        # Calculate energy consumption by vehicle type using total mileage
        for vehicle_type in session.query(VehicleType).filter(VehicleType.scenario_id == scenario_id):
            vehicle_type_short = vehicle_type.name_short or vehicle_type.name

            # Get total (revenue + empty) kilometers for this vehicle type
            total_km_query = session.query(func.sum(Route.distance / 1000.0)).select_from(Trip).join(Route).join(Rotation).filter(
                Trip.scenario_id == scenario_id,
                Rotation.vehicle_type_id == vehicle_type.id
            )
            total_km_weekly = total_km_query.scalar() or 0.0
            # Scale from one week to one year
            total_km_yearly = total_km_weekly * 52

            # Apply energy consumption rate
            if vehicle_type_short in energy_consumption_rates:
                energy_rate = energy_consumption_rates[vehicle_type_short]
                total_energy_consumption += total_km_yearly * energy_rate

        # Total revenue kilometers across all vehicles
        total_revenue_km_weekly = session.query(func.sum(Route.distance / 1000.0)).select_from(Trip).join(Route).filter(
            Trip.scenario_id == scenario_id,
            Trip.trip_type == TripType.PASSENGER
        ).scalar() or 0.0
        # Scale from one week to one year
        total_revenue_km = total_revenue_km_weekly * 52

        # Total kilometers (including empty trips)
        total_km_weekly = session.query(func.sum(Route.distance / 1000.0)).select_from(Trip).join(Route).filter(
            Trip.scenario_id == scenario_id
        ).scalar() or 0.0
        # Scale from one week to one year
        total_km = total_km_weekly * 52

        use_phase_data = UsePhaseData(
            total_energy_consumption=total_energy_consumption,
            total_revenue_kilometers=total_revenue_km,
            total_kilometers=total_km
        )

        # 3. Extract infrastructure data
        infrastructure_data = []

        # First, get all depot stations to exclude them from electrified stations
        depot_station_ids = set()

        # Depot areas with charging
        depot_areas = session.query(Area).filter(
            Area.scenario_id == scenario_id
        )

        for area in depot_areas:
            if any([p.electric_power is not None for p in area.processes]):
                # For depot areas, charging spots = area capacity
                charging_spots = area.capacity

                # Calculate total peak power
                # We need to get the power per charging point from related data
                # This is a simplified calculation - in practice you might need more complex logic
                power_per_spot = 150.0  # Default assumption, should be derived from charging_point_type or events
                total_peak_power = charging_spots * power_per_spot

                infrastructure_data.append(InfrastructureData(
                    location_name=f"{area.depot.name} - {area.name or f'Area {area.id}'}",
                    location_type="depot",
                    charging_spots=charging_spots,
                    total_peak_power=total_peak_power
                ))

                # Also collect station IDs associated with this depot to exclude from electrified stations
                if hasattr(area.depot, 'station_id') and area.depot.station_id:
                    depot_station_ids.add(area.depot.station_id)

        # Get all depot stations to exclude them from electrified stations
        depot_stations = session.query(Station).join(Station.depot).filter(
            Station.scenario_id == scenario_id
        )

        for depot_station in depot_stations:
            depot_station_ids.add(depot_station.id)

        # Electrified stations (excluding those that have depots)
        electrified_stations = session.query(Station).filter(
            Station.scenario_id == scenario_id,
            Station.is_electrified == True,
            ~Station.id.in_(depot_station_ids)  # Exclude depot stations
        )

        for station in electrified_stations:
            infrastructure_data.append(InfrastructureData(
                location_name=station.name,
                location_type="electrified_terminus",
                charging_spots=station.amount_charging_places or 0,
                total_peak_power=station.amount_charging_places * station.power_per_charger or 0.0
            ))

        return ScenarioReport(
            vehicle_manufacture=vehicle_manufacture_data,
            use_phase=use_phase_data,
            infrastructure=infrastructure_data
        ), scenario.name

    finally:
        session.close()


def export_to_json(report: ScenarioReport, scenario_name: str) -> str:
    """Export report data to JSON file."""
    # Clean scenario name for filename
    clean_name = "".join(c for c in scenario_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_name = clean_name.replace(' ', '_')
    filename = f"{clean_name}_data.json"

    # Convert dataclasses to dict
    data = {
        "vehicle_manufacture": [asdict(vm) for vm in report.vehicle_manufacture],
        "use_phase": asdict(report.use_phase),
        "infrastructure": [asdict(infra) for infra in report.infrastructure]
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filename


def export_to_excel(report: ScenarioReport, scenario_name: str) -> List[str]:
    """Export report data to Excel files."""
    filenames = []

    # Clean scenario name for filename
    clean_name = "".join(c for c in scenario_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    clean_name = clean_name.replace(' ', '_')

    # 1. Vehicle Manufacture Excel - One sheet per vehicle type
    manufacture_filename = f"{clean_name}_vehicle_manufacture.xlsx"
    with pd.ExcelWriter(manufacture_filename, engine='openpyxl') as writer:
        for vm_data in report.vehicle_manufacture:
            sheet_name = vm_data.vehicle_type_short_name

            # Create DataFrame with German descriptions
            df = pd.DataFrame({
                'Parameter': [
                    'Batteriekapazität',
                    'Anzahl Fahrzeuge',
                    'Einnahmekilometer'
                ],
                'Wert': [
                    vm_data.battery_capacity,
                    vm_data.vehicle_count,
                    vm_data.revenue_kilometers
                ],
                'Einheit': [
                    'kWh',
                    'Stück',
                    'km'
                ],
                'Erklärung': [
                    'Nutzbare Batteriekapazität des Fahrzeugtyps',
                    'Gesamtzahl der Fahrzeuge dieses Typs',
                    'Jährliche Kilometer mit Fahrgästen (Einnahmefahrten)'
                ]
            })

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    filenames.append(manufacture_filename)

    # 2. Use Phase Excel - Single sheet
    use_phase_filename = f"{clean_name}_use_phase.xlsx"
    use_phase_df = pd.DataFrame({
        'Parameter': [
            'Gesamtenergieverbrauch',
            'Gesamte Einnahmekilometer',
            'Gesamtkilometer'
        ],
        'Wert': [
            report.use_phase.total_energy_consumption,
            report.use_phase.total_revenue_kilometers,
            report.use_phase.total_kilometers
        ],
        'Einheit': [
            'kWh',
            'km',
            'km'
        ],
        'Erklärung': [
            'Jährlicher Energieverbrauch aller Fahrzeuge basierend auf Fahrstrecke',
            'Jährliche Kilometer aller Fahrzeuge mit Fahrgästen',
            'Jährliche Gesamtkilometer aller Fahrzeuge (inkl. Leerfahrten)'
        ]
    })

    use_phase_df.to_excel(use_phase_filename, index=False, sheet_name='Nutzungsphase')
    filenames.append(use_phase_filename)

    # 3. Infrastructure Excel - Separate sheets for depots and terminals
    infrastructure_filename = f"{clean_name}_infrastructure.xlsx"
    with pd.ExcelWriter(infrastructure_filename, engine='openpyxl') as writer:

        # Depot summary
        depot_infra = [infra for infra in report.infrastructure if infra.location_type == "depot"]
        depot_count = len(depot_infra)
        depot_total_spots = sum(infra.charging_spots for infra in depot_infra) if depot_infra else 0
        depot_total_power = sum(infra.total_peak_power for infra in depot_infra) if depot_infra else 0

        depot_df = pd.DataFrame({
            'Parameter': [
                'Anzahl Depots',
                'Gesamte Ladeplätze',
                'Gesamte Spitzenleistung'
            ],
            'Wert': [
                depot_count,
                depot_total_spots,
                depot_total_power
            ],
            'Einheit': [
                'Stück',
                'Stück',
                'kW'
            ],
            'Erklärung': [
                'Anzahl der Depots mit Ladeinfrastruktur',
                'Summe aller Ladeplätze in allen Depots',
                'Summe der maximalen Ladeleistung aller Depots'
            ]
        })

        depot_df.to_excel(writer, sheet_name='Depots', index=False)

        # Terminal summary
        terminal_infra = [infra for infra in report.infrastructure if infra.location_type == "electrified_terminus"]
        terminal_count = len(terminal_infra)
        terminal_total_spots = sum(infra.charging_spots for infra in terminal_infra) if terminal_infra else 0
        terminal_total_power = sum(infra.total_peak_power for infra in terminal_infra) if terminal_infra else 0

        terminal_df = pd.DataFrame({
            'Parameter': [
                'Anzahl elektrifizierte Haltestellen',
                'Gesamte Ladeplätze',
                'Gesamte Spitzenleistung'
            ],
            'Wert': [
                terminal_count,
                terminal_total_spots,
                terminal_total_power
            ],
            'Einheit': [
                'Stück',
                'Stück',
                'kW'
            ],
            'Erklärung': [
                'Anzahl der elektrifizierten Endhaltestellen',
                'Summe aller Ladeplätze an allen elektrifizierten Haltestellen',
                'Summe der maximalen Ladeleistung aller elektrifizierten Haltestellen'
            ]
        })

        terminal_df.to_excel(writer, sheet_name='Elektrifizierte_Haltestellen', index=False)

    filenames.append(infrastructure_filename)

    return filenames


def main():
    """Main function to handle command line arguments and run the extraction."""
    parser = argparse.ArgumentParser(
        description="Extract scenario data from eflips-model database"
    )
    parser.add_argument(
        "scenario_id",
        type=int,
        help="ID of the scenario to extract data for"
    )
    parser.add_argument(
        "--database-url",
        help="Database URL (if not provided, uses DATABASE_URL environment variable)"
    )

    args = parser.parse_args()

    # Get database URL
    database_url = args.database_url or os.getenv('DATABASE_URL')
    if not database_url:
        print("Error: Database URL not provided. Use --database-url argument or set DATABASE_URL environment variable.")
        sys.exit(1)

    try:
        # Extract data
        report, scenario_name = extract_scenario_data(args.scenario_id, database_url)

        # Print results
        print(f"Scenario {args.scenario_id} Data Report")
        print("=" * 50)

        print("\n1. Vehicle Manufacture Data:")
        print("-" * 30)
        for vm_data in report.vehicle_manufacture:
            print(f"Vehicle Type: {vm_data.vehicle_type_short_name}")
            print(f"  Battery Capacity: {vm_data.battery_capacity:.2f} kWh")
            print(f"  Vehicle Count: {vm_data.vehicle_count}")
            print(f"  Revenue Kilometers: {vm_data.revenue_kilometers:.2f} km")
            print()

        print("2. Use Phase Data:")
        print("-" * 20)
        print(f"Total Energy Consumption: {report.use_phase.total_energy_consumption:.2f} kWh")
        print(f"Total Revenue Kilometers: {report.use_phase.total_revenue_kilometers:.2f} km")
        print(f"Total Kilometers: {report.use_phase.total_kilometers:.2f} km")

        print("\n3. Infrastructure Data:")
        print("-" * 25)
        for infra_data in report.infrastructure:
            print(f"Location: {infra_data.location_name}")
            print(f"  Type: {infra_data.location_type}")
            print(f"  Charging Spots: {infra_data.charging_spots}")
            print(f"  Total Peak Power: {infra_data.total_peak_power:.2f} kW")
            print()

        # Export to files
        print("\nExporting data to files...")
        print("-" * 30)

        # Export JSON
        json_filename = export_to_json(report, scenario_name)
        print(f"✓ JSON exported: {json_filename}")

        # Export Excel files
        excel_filenames = export_to_excel(report, scenario_name)
        for filename in excel_filenames:
            print(f"✓ Excel exported: {filename}")

        print(f"\nAll files exported successfully!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()