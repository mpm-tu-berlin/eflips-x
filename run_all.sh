#!/bin/sh

# Run all the scripts in the correct order

# Environment variables that are needed:
# - $DATABASE_NAME : The name of the database for `psql` commands
# - $DATABASE_URL : The URL of the database for python scripts

# This script should be run from the root directory of the project
# And in a poetry shell
export PYTHONPATH=.

#Not working atm
exit(-1)

# Verify that the environment variables are set
if [ -z "$DATABASE_NAME" ]; then
    echo "DATABASE_NAME is not set"
    exit 1
fi
if [ -z "$DATABASE_URL" ]; then
    echo "DATABASE_URL is not set"
    exit 1
fi

# Run the scripts
set -e

python 01_import_and_reduce.py
python 01b_more_station_merging.py
python 01c_vehicle_type_and_depot_plot.py

python 02_triplify_scenario.py
python 02b_verify_scenario.py
pg_dump --no-owner --no-acl $DATABASE_NAME | zstd --rsyncable -T0 -19 > 02c_three_scenario_db.sql.zst

python 03a_scheduling_demo.py
python 03b_scheduling_do_it.py
pg_dump --no-owner --no-acl $DATABASE_NAME | zstd --rsyncable -T0 -19 > 03b_scheduling_done.sql.zst

python 03c_scheduling_results.py

python 04_depot_assignment.py
pg_dump --no-owner --no-acl $DATABASE_NAME | zstd --rsyncable -T0 -19 > 04_depot_assignment_done.sql.zst

python 04c_efficiencies.py
python 05a_is_station_electrification_possible.py
python 05b_do_station_electrification.py
pg_dump --no-owner --no-acl $DATABASE_NAME | zstd --rsyncable -T0 -19 > 05_station_electrification_done.sql.zst

python 06_run_simulation.py
pg_dump --no-owner --no-acl $DATABASE_NAME | zstd --rsyncable -T0 -19 > 06_simulation_done.sql.zst

python 07_analyze.py
python 08_interactive_map.py
python 08b_other_html.py

python 09_diesel_vehicle_count.py
python 09_results_by_scenario_and_depot.py
python 10_result_by_scenario_and_vehicle_type.py
python 11_co2_intensity.py
