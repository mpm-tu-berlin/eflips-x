import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from eflips.x.transition_plan.multi_stage_simulation import simulate_multi_stage_electrification
from eflips.x.transition_plan.transition_plan import run_transition_planner
from eflips.x.flows import run_steps


if __name__ == "__main__":

    FORCE_RERUN = True
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if FORCE_RERUN:
        work_dir = (
            Path(__file__).parent.parent.parent / "transition_plan" / f"{run_id}_example_workdir"
        )
    else:
        work_dir = Path(__file__).parent.parent.parent / "transition_plan" / "example_workdir"
    os.makedirs(work_dir, exist_ok=True)

    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    input_db = data_dir / "step_007_Simulation.db"

    print(f"Working directory: {work_dir}")
    print(f"Input database: {input_db}")

    sets = ["V", "VT", "B", "S", "I"]
    variables = [
        "X_vehicle_year",
        "Z_station_year",
    ]
    constraints = [
        "InitialElectricVehicleConstraint",
        "InitialElectrifiedStationConstraint",
        "NoStationUninstallationConstraint",
        "StationBeforeVehicleConstraint",
        "VehicleDeployTimeLimitConstraint",
        "StationConstructionPerYearConstraint",
        # "NoEarlyStationBuildingConstraint",
        "AssignmentBlockYearConstraint",
        "FullElectrificationConstraint",
        "NoDuplicatedVehicleElectrificationConstraint",
        # "BlockScheduleOnePathConstraint",
        # "BlockScheduleFlowConservationConstraint",
        # "BlockScheduleCostConstraint",
        # "BudgetConstraint",
    ]
    expressions = [
        "Z_block_year",
        "NewlyBuiltStation",
        # "ElectricBusDepreciation",
        # "DieselBusDepreciation",
        # "BatteryDepreciation",
        # "StationChargerDepreciation",
        # "DepotChargerDepreciation",
        "AnnualEbusProcurement",
        "AnnualBatteryProcurement",
        "AnnualVehicleReplacement",
        "AnnualBatteryReplacement",
        "AnnualStationWithChargerProcurement",
        "AnnualDepotChargerProcurement",
        # "ElectricityCost",
        # "DieselCost",
        # "MaintenanceDieselCost",
        # "MaintenanceElectricCost",
        "MaintenanceInfraCost",
        "StaffCostEbus",
        "StaffCostDiesel",
        "EbusEnergySaving",
        "EbusMaintenanceSaving",
        "EbusExtraStaffCost",
    ]
    objective_components = [
        # "ElectricBusDepreciation",
        # "DieselBusDepreciation",
        # "BatteryDepreciation",
        # "StationChargerDepreciation",
        # "DepotChargerDepreciation",
        # "ElectricityCost",
        # "DieselCost",
        # "MaintenanceDieselCost",
        # "MaintenanceElectricCost",
        "MaintenanceInfraCost",
        # "StaffCostEbus",
        # "StaffCostDiesel",
        "AnnualEbusProcurement",
        "AnnualBatteryProcurement",
        "AnnualVehicleReplacement",
        "AnnualBatteryReplacement",
        "AnnualStationWithChargerProcurement",
        "AnnualDepotChargerProcurement",
        "EbusEnergySaving",
        "EbusMaintenanceSaving",
        "EbusExtraStaffCost",
    ]

    current_db, result = run_transition_planner(
        workdir=work_dir,
        input_db=input_db,
        sets=sets,
        variables=variables,
        constraints=constraints,
        expressions=expressions,
        objective_components=objective_components,
    )

    unelectrified_blocks = result["unelectrified_blocks"]
    yearly_vehicle_assignment = result["yearly_vehicle_assignment"]
    yearly_cost_breakdown = result["yearly_cost_breakdown"]

    # Plot yearly vehicle assignment
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_vehicle_assignment.plot(kind="bar", ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Vehicles")
    ax.set_title("Yearly Vehicle Assignment")
    ax.legend()
    fig.savefig(work_dir / "yearly_vehicle_assignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot yearly cost breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    yearly_cost_breakdown.plot(kind="bar", stacked=True, ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cost")
    ax.set_title("Yearly Cost Breakdown")
    ax.legend()
    fig.savefig(work_dir / "yearly_cost_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {work_dir}")

    # simulate_multi_stage_electrification(
    #     unelectrified_blocks=unelectrified_blocks,
    #     workdir=work_dir,
    #     input_db=current_db,
    #     log_level="INFO",
    #     # force_rerun=True,
    # )
