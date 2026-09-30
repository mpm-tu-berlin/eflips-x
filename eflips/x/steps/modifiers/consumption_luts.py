"""
Consumption look-up tables shipped with eflips-x.

The CSVs in ``data/input/consumption_luts/`` are taken verbatim from the
django-simba example dataset (one regression-derived table per typical bus
length). They follow the column convention expected by
``eflips.model.ConsumptionLut.df_to_consumption_obj``.

Measured two-dimensional (speed × temperature) tables such as
``data/input/consumption_lut_gn.xlsx`` use a different, spreadsheet-style
layout and are read with :func:`load_measured_speed_temperature_lut`.

:func:`generate_clamped_consumption_result` wraps eflips-depot's per-trip
consumption calculation so that no trip ends with a net energy gain.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from eflips.depot.api import (  # type: ignore[import-untyped]
    ConsumptionResult,
    generate_consumption_result,
)
from eflips.model import Scenario

logger = logging.getLogger(__name__)

CONSUMPTION_LUT_DIR = Path(__file__).resolve().parents[4] / "data" / "input" / "consumption_luts"


class ConsumptionLut(Enum):
    """Pre-canned consumption look-up tables, indexed by source bus length."""

    SPRINTER_6M = "6m_consumption_sprinter_6m.csv"
    LLE_10M = "10m_consumption_lle_99.csv"
    NOR_BUS_12M = "12m_consumption_nor_bus.csv"
    SOLARIS_18M = "18m_consumption_solaris_18m.csv"

    @property
    def path(self) -> Path:
        return CONSUMPTION_LUT_DIR / self.value


def load_consumption_lut_df(member: ConsumptionLut) -> pd.DataFrame:
    return pd.read_csv(member.path)


def load_measured_speed_temperature_lut(
    path: Union[str, Path],
) -> List[Tuple[float, float, float]]:
    """
    Load a measured two-dimensional consumption table from an Excel file.

    Layout (see ``data/input/consumption_lut_gn.xlsx``): the first column holds
    the mean speeds in km/h, the header row (from the second column on) holds
    the ambient temperatures in °C, and every cell is the measured consumption
    in kWh/km. Empty cells (NaN) are skipped, so the returned point cloud may
    be ragged.

    :param path: Path to the Excel file.
    :return: A list of ``(mean_speed_kmh, t_amb, consumption_kwh_per_km)`` tuples.
    """
    table = pd.read_excel(path)

    temperatures = np.array(table.columns[1:]).astype(np.float64)
    speeds = np.array(table.iloc[:, 0]).astype(np.float64)
    data = np.array(table.iloc[:, 1:]).astype(np.float64)  # shape (n_speeds, n_temps)

    points: List[Tuple[float, float, float]] = []
    for i, temperature in enumerate(temperatures):
        for j, speed in enumerate(speeds):
            value = data[j, i]
            if not np.isnan(value):
                points.append((float(speed), float(temperature), float(value)))
    return points


def generate_clamped_consumption_result(scenario: Scenario) -> Dict[int, ConsumptionResult]:
    """
    Compute per-trip consumption results like eflips-depot's ``generate_consumption_result``,
    but clamp every trip's net SoC change to at most zero.

    With a LUT that has a negative (recuperating) incline term, a short, steep downhill trip
    can come out with a net energy gain, especially at mild temperatures where the flat-ground
    consumption is low. ``simple_consumption_simulation`` rejects such a trip with "The
    delta_soc_total must be <= 0 when using a consumption result." Clamping treats the trip as
    energy-neutral instead: the vehicle arrives with the SoC it departed with, and the
    cumulative SoC timeseries is capped at zero as well.

    :param scenario: The scenario to compute consumption results for.
    :return: A dictionary mapping trip IDs to (clamped) consumption results.
    """
    results: Dict[int, ConsumptionResult] = generate_consumption_result(scenario)

    clamped_trip_ids = []
    for trip_id, result in results.items():
        if result.delta_soc_total > 0:
            clamped_trip_ids.append(trip_id)
            result.delta_soc_total = 0.0
        if result.delta_soc is not None:
            result.delta_soc = [min(d, 0.0) for d in result.delta_soc]

    if clamped_trip_ids:
        logger.info(
            "Clamped %d trip(s) with a net energy gain to zero consumption (trip IDs: %s)",
            len(clamped_trip_ids),
            ", ".join(str(t) for t in sorted(clamped_trip_ids)),
        )
    return results
