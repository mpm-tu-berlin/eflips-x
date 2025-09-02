from eflips.tco.tco_calculator import TCOCalculator
from eflips.tco.data_queries import init_tco_parameters

from eflips.model import VehicleType, BatteryType, ChargingPointType, Scenario

from sqlalchemy.orm.session import Session


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
            "procurement_cost": 450000.0,
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
            "procurement_cost": 585000.0,
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
            "procurement_cost": 585000.0,
            "cost_escalation": 0.02,
        }
    )

    battery_type_parameters = [
        {
            "name": "Ebusco 3.0 12 large battery",
            "procurement_cost": 190,
            "useful_life": 7,
            "cost_escalation": -0.03,
            "vehicle_type_id": 12,
        },
        {
            "name": "Solaris Urbino 18 large battery",
            "procurement_cost": 190,
            "useful_life": 7,
            "cost_escalation": -0.03,
            "vehicle_type_id": 13,
        },
        {
            "name": "Alexander Dennis Enviro500EV large battery",
            "procurement_cost": 190,
            "useful_life": 7,
            "cost_escalation": -0.03,
            "vehicle_type_id": 14,
        },
    ]

    charging_point_type_parameters = [
        {
            "type": "depot",
            "name": "Depot Charging Point",
            "procurement_cost": 100000.0,
            "useful_life": 20,
            "cost_escalation": 0.02,
        },
        {
            "type": "opportunity",
            "name": "Opportunity Charging Point",
            "procurement_cost": 250000.0,
            "useful_life": 20,
            "cost_escalation": 0.02,
        },
    ]

    charging_infrastructure_parameters = [
        {
            "type": "depot",
            "name": "Depot Charging Infrastructure",
            "procurement_cost": 2000000.0,
            "useful_life": 20,
            "cost_escalation": 0.02,
        },
        {
            "type": "station",
            "name": "Opportunity Charging Infrastructure",
            "procurement_cost": 225000.0,
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
            "12": 1.48,
            "13": 2.16,
            "14": 2.16,
        },
    }

    return {
        "vehicle_types": vehicle_type_parameters,
        "battery_types": battery_type_parameters,
        "charging_point_types": charging_point_type_parameters,
        "charging_infrastructure": charging_infrastructure_parameters,
        "scenario_tco_parameters": scenario_tco_parameters,
    }
