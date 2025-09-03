#! /usr/bin/env python3
import os
import warnings
from datetime import timedelta
from tempfile import gettempdir
from typing import List

import pandas as pd
import pytz
import seaborn as sns
from eflips.model import *
from eflips.model import ConsistencyWarning
from eflips.model import VehicleType, Scenario
from eflips.tco.data_queries import init_tco_parameters
from eflips.tco.tco_calculator import TCOCalculator
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session

import plotutils


def tco_parameters(scenario: Scenario, session: Session):
    """
    This function reads the database and initializes the TCO parameters if they are not present.
    :param scenario:
    :param Session:
    :return:
    """

    vehicle_type_parameters = []
    vt_en = (
        session.query(VehicleType)
        .filter(VehicleType.scenario_id == scenario.id, VehicleType.name_short == "EN")
        .one()
    )
    vehicle_type_parameters.append(
        {
            "id": vt_en.id,
            "name": vt_en.name,
            "useful_life": 14,
            "procurement_cost": 580000.0,
            "cost_escalation": 0.02,
        }
    )

    vt_gn = (
        session.query(VehicleType)
        .filter(VehicleType.scenario_id == scenario.id, VehicleType.name_short == "GN")
        .one()
    )
    vehicle_type_parameters.append(
        {
            "id": vt_gn.id,
            "name": vt_gn.name,
            "useful_life": 14,
            "procurement_cost": 780000.0,
            "cost_escalation": 0.02,
        }
    )
    vt_dd = (
        session.query(VehicleType)
        .filter(VehicleType.scenario_id == scenario.id, VehicleType.name_short == "DD")
        .one()
    )
    vehicle_type_parameters.append(
        {
            "id": vt_dd.id,
            "name": vt_dd.name,
            "useful_life": 14,
            "procurement_cost": 780000.0,
            "cost_escalation": 0.02,
        }
    )

    battery_type_parameters = [
        {
            "name": "Ebusco 3.0 12 large battery",
            "procurement_cost": 190,
            "useful_life": 7,
            "cost_escalation": -0.03,
            "vehicle_type_id": session.query(VehicleType).filter(
                VehicleType.scenario_id == scenario.id, VehicleType.name_short == "EN"
            ).one().id,
        },
        {
            "name": "Solaris Urbino 18 large battery",
            "procurement_cost": 190,
            "useful_life": 7,
            "cost_escalation": -0.03,
            "vehicle_type_id": session.query(VehicleType).filter(
                VehicleType.scenario_id == scenario.id, VehicleType.name_short == "GN"
            ).one().id,
        },
        {
            "name": "Alexander Dennis Enviro500EV large battery",
            "procurement_cost": 190,
            "useful_life": 7,
            "cost_escalation": -0.03,
            "vehicle_type_id": session.query(VehicleType).filter(
                VehicleType.scenario_id == scenario.id, VehicleType.name_short == "DD"
            ).one().id,
        },
    ]

    charging_point_type_parameters = [
        {
            "type": "depot",
            "name": "Depot Charging Point",
            "procurement_cost": 119899.50,
            "useful_life": 20,
            "cost_escalation": 0.02,
        },
        {
            "type": "opportunity",
            "name": "Opportunity Charging Point",
            "procurement_cost": 299748.74,
            "useful_life": 20,
            "cost_escalation": 0.02,
        },
    ]

    charging_infrastructure_parameters = [
        {
            "type": "depot",
            "name": "Depot Charging Infrastructure",
            "procurement_cost": 2397989.95,
            "useful_life": 20,
            "cost_escalation": 0.02,
        },
        {
            "type": "station",
            "name": "Opportunity Charging Infrastructure",
            "procurement_cost": 269773.87,
            "useful_life": 20,
            "cost_escalation": 0.02,
        },
    ]

    scenario_tco_parameters = {
        "project_duration": 20,
        "interest_rate": 0.04,
        "inflation_rate": 0.02,
        "staff_cost": 25.0,  # calculated: 35,000 € p.a. per driver/1600 h p.a. per driver
        # Fuel cost in EUR per unit fuel
        "fuel_cost": 0.1794,  # electricity cost
        # Maintenance cost in EUR per km
        "maint_cost": 0.35,
        # Maintenance cost infrastructure per year and charging slot
        "maint_infr_cost": 1000,
        # Taxes and insurance cost in EUR per year and bus
        "taxes": 278,
        "insurance": 9693,  # DCO #9703, # EBU
        # Cost escalation factors (cef / pef)
        "pef_general": 0.02,
        "pef_wages": 0.025,
        "pef_fuel": 0.038,
        "pef_insurance": 0.02,
        "const_energy_consumption": {
            f"{session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id, VehicleType.name_short == 'EN').one().id}": 1.48,
            f"{session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id, VehicleType.name_short == 'GN').one().id}": 2.16,
            f"{session.query(VehicleType).filter(VehicleType.scenario_id == scenario.id, VehicleType.name_short == 'DD').one().id}": 2.16,
        },
    }

    return {
        "vehicle_types": vehicle_type_parameters,
        "battery_types": battery_type_parameters,
        "charging_point_types": charging_point_type_parameters,
        "charging_infrastructure": charging_infrastructure_parameters,
        "scenario_tco_parameters": scenario_tco_parameters,
    }


warnings.simplefilter("ignore", ConsistencyWarning)
USE_SIMBA_CONSUMPTION = True
PARALLELISM = True

TEMPORAL_RESOLUTION = 30

tz = pytz.timezone("Europe/Berlin")
START_OF_SIMULATION = tz.localize(datetime.datetime(2023, 7, 3, 0, 0, 0))
END_OF_SIMULATION = START_OF_SIMULATION + timedelta(days=7)

if __name__ == "__main__":
    assert (
            "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."
    DATABASE_URL = os.environ["DATABASE_URL"]

    engine = create_engine(DATABASE_URL, execution_options={"postgesql_readonly": True})
    session = Session(engine)

    scenarios = session.query(Scenario).all()

    results: List[dict] = []

    cache_file_path = os.path.join(gettempdir(), "tco_cache.pkl")
    if os.path.exists(cache_file_path):
        df = pd.read_pickle(cache_file_path)
    else:
        for scenario in scenarios:
            dict_tco_params = tco_parameters(scenario, session)
            init_tco_parameters(
                scenario=scenario,
                scenario_tco_parameters=dict_tco_params["scenario_tco_parameters"],
                vehicle_types=dict_tco_params["vehicle_types"],
                battery_types=dict_tco_params["battery_types"],
                charging_point_types=dict_tco_params["charging_point_types"],
                charging_infrastructure=dict_tco_params["charging_infrastructure"],
            )

            tco_calculator = TCOCalculator(scenario=scenario, energy_consumption_mode="constant")
            tco_calculator.calculate()
            result = tco_calculator.tco_by_type
            result["INFRASTRUCTURE"] += result.get("CHARGING_POINT", 0.0)
            result.pop("CHARGING_POINT", None)
            result["scenario"] = scenario.name_short
            results.append(result)

        df = pd.DataFrame(results)
        with open(cache_file_path, "wb") as f:
            df.to_pickle(f)

    df.to_excel("30_tco_results.xlsx", index=False)

    # THe dataframe has the columns
    # "['INFRASTRUCTURE', 'OTHER', 'MAINTENANCE', 'VEHICLE', 'ENERGY', 'BATTERY', 'STAFF', 'scenario']"

    # Create TCO plot
    scenario_name_mapping = {
        "OU": "Existing\nBlocks\nUnchanged",
        "DEP": "Depot\nCharging\nOnly",
        "TERM": "Small\nBatteries and\nTermini"
    }

    # Category name mapping from all caps to proper case
    category_name_mapping = {
        'INFRASTRUCTURE': 'Infrastructure',
        'OTHER': 'Other',
        'MAINTENANCE': 'Maintenance',
        'VEHICLE': 'Vehicle',
        'ENERGY': 'Energy',
        'BATTERY': 'Battery',
        'STAFF': 'Staff'
    }

    # Replace scenario names with long names
    df["scenario_long"] = df["scenario"].map(scenario_name_mapping)

    # Melt the dataframe for plotting
    cost_columns = ['INFRASTRUCTURE', 'OTHER', 'MAINTENANCE', 'VEHICLE', 'ENERGY', 'BATTERY', 'STAFF']
    df_melted = df.melt(id_vars=['scenario', 'scenario_long'],
                        value_vars=cost_columns,
                        var_name='Cost Category',
                        value_name='Cost')

    # Map category names to proper case
    df_melted['Cost Category'] = df_melted['Cost Category'].map(category_name_mapping)

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT))

    # Use seaborn color palette
    palette = sns.color_palette("Set2")

    # Pivot for stacked bar chart
    df_pivot = df_melted.pivot(index='scenario_long', columns='Cost Category', values='Cost')

    # Plot stacked bar chart
    df_pivot.plot(kind='bar', stacked=True, ax=ax, color=palette)

    # Add value labels inside each bar segment and sum totals on top
    for i, (scenario, row) in enumerate(df_pivot.iterrows()):
        y_offset = 0
        total_sum = row.sum()  # Calculate sum from unrounded values

        for j, (category, value) in enumerate(row.items()):
            if value > 0:  # Only show labels for non-zero values
                # Position label in the middle of the bar segment
                label_y = y_offset + value / 2
                ax.text(i, label_y, f'{value:.2f}',
                        ha='center', va='center',
                        fontweight='normal', fontsize=8, color='black')
            y_offset += value

        # Add sum total on top of each bar
        ax.text(i, y_offset + 0.02, f'{total_sum:.2f}',
                ha='center', va='bottom',
                fontweight='bold', fontsize=10, color='black')

    ax.set_title("")
    ax.set_ylabel(r"Total Cost of Ownership $\left[ \frac{EUR}{km} \right]$")
    ax.set_xlabel("")

    # Keep x-axis labels horizontal
    plt.xticks(rotation=0, ha='center')

    # Scale y-axis by 10% to accommodate sum totals on top
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.1)

    # Position legend to the right of the plot, arranged vertically
    plt.legend(title="", bbox_to_anchor=(1.05, 1), loc="upper left", ncols=1)

    # Adjust plot to make room for right-side legend and prevent bottom cutoff
    # plt.subplots_adjust(right=0.7, bottom=0.15)
    plt.tight_layout()
    plt.savefig("31_tco_plot.pdf")
    plt.show()
