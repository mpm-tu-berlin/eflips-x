from eflips.transition.parameter_registry import ConstraintsParams

constraints = ConstraintsParams(
    project_duration=20,
    annual_budget=0.8e7,
    maximum_annual_new_station=10,
    maximum_annual_new_depot=10,
    current_year=2026,
    depot_construction_duration=3,
    min_charger_batch_size=20,
    new_depots=[1102005186, 1102005187, 1102005188],
    scaling_factor_to_year=52,
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
    "DepotChargerProcurement",
]

constraints_long_term = [
    # "YearlyReplacedUpperBound",
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
    "FullElectrificationConstraint",
    "NoDuplicatedVehicleElectrificationConstraint",
    # "BudgetConstraint",
    "DepotChargerConstructionLimit",
    "DepotChargerRequiresDepot",
    "DepotChargerNoUninstallation",
    "DepotChargerProcurementLower",
    "DepotChargerProcurementUpper",
    "InitialDepotChargerConstraint",
    "DieselReplacedLowerBound",
    "DieselReplacedUpperBound",  # Lower bound and upper bound are for limiting diesel bus number to integers
    "DieselOldestFirstReplacement",
    "DieselMandatoryRetirement",
    "DieselRetirementCap",
    "DieselReplacementCap",
    "DieselEarlyRetirementEquality",
    "DieselNaturalRetirementCap",
    "DieselBusCapacityConstraint",
    # "NoDieselBusProcurement",  # activate to forbid any new diesel bus purchases
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
    "InsuranceTaxCostEbus",
    "InsuranceTaxCostDiesel",
    "AnnualTotalCost",
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
    "InsuranceTaxCostEbus",
    "InsuranceTaxCostDiesel",
]

# Procurement / cash-flow view of annual cost: actual capital outlays (NOT
# depreciation) plus OPEX. Counterpart to objective_components (the
# depreciation / TCO view) for the side-by-side cost-breakdown plot.
procurement_components = [
    "AnnualEbusProcurement",
    "AnnualVehicleReplacement",
    "AnnualBatteryProcurement",
    "AnnualBatteryReplacement",
    "AnnualDieselBusProcurement",
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
    "InsuranceTaxCostEbus",
    "InsuranceTaxCostDiesel",
]

# Components paid the year before they become operational; shifted one year by
# compute_cost_breakdown so the breakdown reconciles with AnnualTotalCost.
shifted_procurement_components = [
    "AnnualStationConstructionCost",
    "AnnualStationChargerProcurement",
    "AnnualDepotChargerProcurement",
]

name = "transition_plan_berlin"
