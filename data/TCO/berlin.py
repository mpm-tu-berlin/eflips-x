"""Berlin-specific TCO parameter defaults."""

from eflips.tco.tco_parameter_config import (
    VehicleTypeTCOParameter,
    BatteryTypeTCOParameter,
    ChargingPointTypeTCOParameter,
    ChargingInfrastructureTCOParameter,
    ScenarioTCOParameter,
)

# Vehicle type defaults. from bvg internal data
VEHICLE_TYPES = [
    VehicleTypeTCOParameter(
        name_short="EN",
        name="Ebusco 3.0 12 large battery",
        useful_life=14,
        procurement_cost=340000.0,
        cost_escalation=-0.02,
        const_energy_consumption=1.48,
    ),
    VehicleTypeTCOParameter(
        name_short="DD",
        name="Solaris Urbino 18 large battery",
        useful_life=14,
        procurement_cost=603000.0,
        cost_escalation=-0.02,
        const_energy_consumption=2.16,
    ),
    VehicleTypeTCOParameter(
        name_short="GN",
        name="Alexander Dennis Enviro500EV large battery",
        useful_life=14,
        procurement_cost=650000.0,
        cost_escalation=-0.02,
        const_energy_consumption=2.16,
    ),
]

# Battery type defaults
BATTERY_TYPES = [
    BatteryTypeTCOParameter(
        name="Ebusco 3.0 12 large battery",
        vehicle_name_short="EN",
        procurement_cost=190,
        useful_life=7,
        cost_escalation=-0.03,
    ),
    BatteryTypeTCOParameter(
        name="Solaris Urbino 18 large battery",
        vehicle_name_short="DD",
        procurement_cost=190,
        useful_life=7,
        cost_escalation=-0.03,
    ),
    BatteryTypeTCOParameter(
        name="Alexander Dennis Enviro500EV large battery",
        vehicle_name_short="GN",
        procurement_cost=190,
        useful_life=7,
        cost_escalation=-0.03,
    ),
]

# Charging point defaults
# Prices from Jefferies and Goehlich (2020)
CHARGING_POINT_TYPES = [
    ChargingPointTypeTCOParameter(
        type="depot",
        name="Depot Charging Point",
        procurement_cost=100000.0,
        useful_life=20,
        cost_escalation=0,
    ),
    ChargingPointTypeTCOParameter(
        type="opportunity",
        name="Opportunity Charging Point",
        procurement_cost=250000.0,
        useful_life=20,
        cost_escalation=0,
    ),
]

# Charging infrastructure defaults
# Prices from Jefferies and Goehlich (2020)
CHARGING_INFRASTRUCTURE = [
    ChargingInfrastructureTCOParameter(
        type="depot",
        name="Depot Charging Infrastructure",
        procurement_cost=2000000.0,
        useful_life=20,
        cost_escalation=0,
    ),
    ChargingInfrastructureTCOParameter(
        type="station",
        name="Opportunity Charging Infrastructure",
        procurement_cost=500000.0,
        useful_life=20,
        cost_escalation=0,
    ),
]

# Scenario TCO parameters. from bvg internal data
SCENARIO_TCO = ScenarioTCOParameter(
    project_duration=14,
    interest_rate=0.04,
    inflation_rate=0.02,
    staff_cost=35.0,
    fuel_cost={"diesel": 1.0, "electricity": 0.15},
    vehicle_maint_cost={"diesel": 0.5, "electricity": 0.20},
    infra_maint_cost=5000.0,
    cost_escalation_rate={
        "general": 0.02,
        "staff": 0.02,
        "diesel": 0.07,
        "electricity": 0.03,
        "insurance": 0.02,
    },
    insurance=3000.0,
    taxes=0.0,
)
