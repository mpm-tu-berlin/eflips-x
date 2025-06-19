# Electric Bus Fleet Optimization and Charging Infrastructure Planning

## Disclaimer

This codebase represents a research prototype with sequential script dependencies that require manual execution in numerical order. While functional, the current architecture lacks explicit dependency management and automated workflow orchestration. Future development will incorporate a formal build system or directed acyclic graph (DAG) framework such as Apache Airflow to improve reproducibility and maintainability. Users should expect some manual intervention and troubleshooting when running the complete pipeline.

## License

- **Code**: Licensed under GNU Affero General Public License v3.0 (AGPL-3.0-or-later)
- **Data**: Licensed under Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA 4.0)

## Overview

This project provides a comprehensive simulation and optimization framework for electric bus fleet operations, focusing on vehicle scheduling, energy consumption modeling, and charging infrastructure placement. The system analyzes different electrification scenarios for public transit systems using the eFLIPS (Electric Fleet Simulation) framework.

### Key Features

- **Multi-scenario Analysis**: Compare different electrification strategies (Original routes, Depot charging, Terminal charging)
- **Energy Consumption Modeling**: Physics-based and empirical consumption models with temperature effects
- **Optimal Scheduling**: Vehicle rotation optimization using graph-based algorithms  
- **Charging Infrastructure Planning**: Automated placement of charging stations based on operational requirements
- **Interactive Visualization**: Web-based maps and charts for results analysis

## Setup

1. **Install Dependencies**
   ```bash
   poetry install
   ```

2. **Database Configuration**
   ```bash
   export DATABASE_URL="postgresql://user:password@host:port/dbname"
   ```

## Usage

**Important**: Scripts must be run in numerical sequence (01 through 12) for proper operation:

```bash
# Data import and preprocessing
python 01_import_and_reduce.py
python 01b_more_station_merging.py
python 02_triplify_scenario.py
python 02b_verify_scenario.py

# Energy modeling and analysis
python 03a_vehicle_type_and_depot_plot.py
python 03b_consumption_table.py
python 03c_create_consumption_table.py
python 03d_add_temperatures.py
python 03e_total_deadhead_km.py
python 04_trip_consumption_simulation.py

# Optimization
python 05_scheduling.py
python 06_depot_assignment.py
python 06c_vehicle_type_and_depot_plot.py

# Charging infrastructure planning
python 08a_is_station_electrification_possible.py
python 08b_do_station_electrification.py
python 08c_postprocess_term_scenario.py

# Simulation and analysis
python 09_schedule_evaluation.py
python 10_run_simulation.py
python 11_analyze.py
python 12_interactive_map.py
```

## Scenarios

The framework analyzes three main electrification scenarios:

- **OU (Original)**: Baseline scenario with original vehicle routes
- **DEP (Depot)**: Depot-only charging with larger batteries  
- **TERM (Terminal)**: Terminal charging with smaller batteries and opportunity charging

## Outputs

- **Excel Reports**: Detailed analysis results and metrics
- **PDF Plots**: Vehicle distribution and consumption visualizations
- **Interactive Map**: Web-based visualization with charging station utilization (`output_interactive/index.html`)
- **Optimization Logs**: Infrastructure planning progress tracking

## Configuration

Key parameters can be modified in `params.py` for different analysis contexts. Temperature profiles and consumption data are included in the respective Excel files.

## Support

For questions or issues, please contact:
- ludger.heide@tu-berlin.de
- shuyao.guo@tu-berlin.de

## Notes

- Requires PostgreSQL database with PostGIS extension and eFLIPS schema
- Parallel processing is available for computationally intensive operations
- All geographic calculations use WGS84 coordinate system (EPSG:4326)