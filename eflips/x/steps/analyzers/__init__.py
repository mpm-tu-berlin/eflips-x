"""
Analyzer modules for eflips-x pipeline.

This module provides analyzer classes that wrap eflips-eval functionality
for use within the eflips-x pipeline framework.
"""

from eflips.x.steps.analyzers.bvg_tools import (
    RepresentativeVehicleSocAnalyzer,
    RevenueServiceTimelineAnalyzer,
    ScenarioComparisonAnalyzer,
    SchedulingEfficiencyAnalyzer,
    VehicleTypeDepotPlotAnalyzer,
    merge_scenario_comparisons,
)
from eflips.x.steps.analyzers.input_analyzers import (
    GeographicTripPlotAnalyzer,
    RotationInfoAnalyzer,
    SingleRotationInfoAnalyzer,
)
from eflips.x.steps.analyzers.output_analyzers import (
    DepartureArrivalSocAnalyzer,
    DepotActivityAnalyzer,
    DepotEventAnalyzer,
    InteractiveMapAnalyzer,
    PowerAndOccupancyAnalyzer,
    SpecificEnergyConsumptionAnalyzer,
    VehicleSocAnalyzer,
)

__all__ = [
    # BVG-specific analyzers
    "RepresentativeVehicleSocAnalyzer",
    "RevenueServiceTimelineAnalyzer",
    "ScenarioComparisonAnalyzer",
    "SchedulingEfficiencyAnalyzer",
    "VehicleTypeDepotPlotAnalyzer",
    "merge_scenario_comparisons",
    # Input analyzers
    "RotationInfoAnalyzer",
    "GeographicTripPlotAnalyzer",
    "SingleRotationInfoAnalyzer",
    # Output analyzers
    "DepartureArrivalSocAnalyzer",
    "DepotEventAnalyzer",
    "InteractiveMapAnalyzer",
    "PowerAndOccupancyAnalyzer",
    "SpecificEnergyConsumptionAnalyzer",
    "VehicleSocAnalyzer",
    "DepotActivityAnalyzer",
]
