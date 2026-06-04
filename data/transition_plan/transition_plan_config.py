from eflips.transition.parameter_registry import ConstraintsParams

constraints = ConstraintsParams(
    project_duration=20,
    annual_budget=0.8e7,
    maximum_annual_new_station=5,
    maximum_annual_new_depot=10,
    current_year=2026,
    diesel_salvage_decay_rate=0.135,
    diesel_salvage_cutoff_age=12,
)

sets = ["V", "VT", "B", "S", "I", "D", "DE_pairs", "PY", "AgeIdx"]
variables = [
    "ElectrifyVehicle",
    "StationOperational",
    "DepotChargerCount",
    "DepotOperational",
    "ReplacedDieselBus",
    "RetiredDieselBus",
]

constraints_long_term = [
    "YearlyReplacedUpperBound",
    "InitialElectricVehicleConstraint",
    "InitialElectrifiedStationConstraint",
    "NoStationUninstallationConstraint",
    "StationBeforeVehicleConstraint",
    "DepotNoUninstallation",
    "InitialDepotConstraint",
    "DepotBeforeVehicleConstraint",
    "DepotConstructionLimit",
    # "StationConstructionPerYearConstraint",
    # "NoEarlyStationBuildingConstraint",
    "AssignmentBlockYearConstraint",
    # "FullElectrificationConstraint",
    "NoDuplicatedVehicleElectrificationConstraint",
    # "BudgetConstraint",
    "DepotChargerConstructionLimit",
    "DepotChargerRequiresDepot",
    "DepotChargerNoUninstallation",
    "InitialDepotChargerConstraint",
    "DieselReplacedLowerBound",
    "DieselReplacedUpperBound",  # Lower bound and upper bound are for limiting diesel bus number to integers
    "DieselOldestFirstReplacement",
    "DieselMandatoryRetirement",
    "DieselRetirementCap",
    "DieselReplacementCap",
    "DieselEarlyRetirementEquality",
    "DieselNaturalRetirementCap",
    "NoDieselBusProcurement",  # activate to forbid any new diesel bus purchases
]

expressions_long_term = [
    "BlockElectrified",
    "NewlyBuiltStation",
    "ElectricBusDepreciation",
    "DieselBusDepreciation",
    # "DieselSalvageCredit",
    "BatteryDepreciation",
    "StationChargerDepreciation",
    "DepotChargerDepreciation",
    "AnnualEbusProcurement",
    "AnnualBatteryProcurement",
    "AnnualDieselBusProcurement",
    "AnnualVehicleReplacement",
    "AnnualBatteryReplacement",
    "AnnualStationConstructionCost",
    "AnnualStationChargerProcurement",
    "AnnualDepotConstructionCost",
    "AnnualDepotChargerProcurement",
    "ElectricityCost",
    "DieselCost",
    "MaintenanceDieselCost",
    "MaintenanceElectricCost",
    "MaintenanceInfraCost",
    "StaffCostEbus",
    "StaffCostDiesel",
]

objective_components = [
    "ElectricBusDepreciation",
    "DieselBusDepreciation",
    # "DieselSalvageCredit",
    "BatteryDepreciation",
    "AnnualStationConstructionCost",
    "AnnualDepotConstructionCost",
    "StationChargerDepreciation",
    "DepotChargerDepreciation",
    "ElectricityCost",
    "DieselCost",
    "MaintenanceDieselCost",
    "MaintenanceElectricCost",
    "MaintenanceInfraCost",
    "StaffCostEbus",
    "StaffCostDiesel",
    # "AnnualEbusProcurement",
    # "AnnualBatteryProcurement",
    # "AnnualVehicleReplacement",
    # "AnnualBatteryReplacement",
    # "EbusResidualValue",
    # "BatteryResidualValue",
    # "AnnualDepotChargerProcurement",
    # "EbusEnergySaving",
    # "EbusMaintenanceSaving",
    # "EbusExtraStaffCost",
]

name = "transition_plan_berlin_literature"
