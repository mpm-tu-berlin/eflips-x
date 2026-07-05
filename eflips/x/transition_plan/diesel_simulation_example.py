"""Smoke test / usage example for :func:`run_diesel_simulation`.

Runs the standalone diesel simulation flow
(``CreateDieselVehicleTypes -> VehicleTypeBlockAssignment -> DepotGenerator ->
Simulation -> DieselFleetAnalyzer``) against a copy of an input database and
prints the resulting :class:`~eflips.transition.parameter_registry.DieselFleetParams`.

The input database is only ever *copied* (by ``CopyCreator``) into ``WORK_DIR``,
so the source file is left untouched.

Usage::

    export SPATIALITE_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mod_spatialite.so
    poetry run python -m eflips.x.transition_plan.diesel_simulation_example
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from eflips.transition.parameter_registry import DieselFleetParams

from eflips.x.framework import PipelineContext
from eflips.x.transition_plan.transition_plan import run_diesel_simulation

# --------------------------------------------------------------------------- #
# Configuration (hard-coded for now)
# --------------------------------------------------------------------------- #

# Input (electric) database to branch from. It is copied, never modified.
# Sibling "eflips-data" checkout next to this "eflips-x" repository.
INPUT_DB = Path(__file__).parent.parent.parent.parent.parent / "eflips-data" / "diesel_mini.db"

# Working directory for the diesel database and its intermediate step files.
# Kept separate from the electric scenario so the two databases never collide.
WORK_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "data"
    / "output"
    / "transition_plan"
    / "diesel"
    / INPUT_DB.stem
)

# Depot charging power in kW passed to the DepotGenerator step.
CHARGING_POWER_KW = 90.0


def main() -> DieselFleetParams:
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Input database:    {INPUT_DB}")
    print(f"Working directory: {WORK_DIR}")

    # The context carries the source database (copied, never modified), the diesel
    # work_dir, and the simulation-step parameters. ignore_* are relaxed so a smoke
    # test does not abort on an unstable/delayed schedule.
    context = PipelineContext(
        work_dir=WORK_DIR,
        current_db=INPUT_DB,
        params={
            "DepotGenerator.charging_power_kw": CHARGING_POWER_KW,
            "Simulation.ignore_unstable_simulation": True,
            "Simulation.ignore_delayed_trips": True,
        },
    )

    counts = run_diesel_simulation(context)

    print("\n=== DieselFleetParams ===")
    print("bus_count_by_type (electric_vt_id -> diesel bus count):")
    for vt_id, n in sorted(counts.bus_count_by_type.items()):
        print(f"  vt {vt_id}: {n}")
    print(f"  total diesel buses: {sum(counts.bus_count_by_type.values())}")
    print("initial_depot_capacities (depot.station_id -> slots):")
    for station_id, slots in sorted(counts.initial_depot_capacities.items()):
        print(f"  station {station_id}: {slots}")
    print(f"  total slots: {sum(counts.initial_depot_capacities.values())}")

    return counts


if __name__ == "__main__":
    main()
