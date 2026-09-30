"""
Consumption look-up tables shipped with eflips-x.

The CSVs in ``data/input/consumption_luts/`` are taken verbatim from the
django-simba example dataset (one regression-derived table per typical bus
length). They follow the column convention expected by
``eflips.model.ConsumptionLut.df_to_consumption_obj``.

Measured two-dimensional (speed × temperature) tables such as
``data/input/consumption_lut_gn.xlsx`` use a different, spreadsheet-style
layout and are read with :func:`load_measured_speed_temperature_lut`.
"""

from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

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
