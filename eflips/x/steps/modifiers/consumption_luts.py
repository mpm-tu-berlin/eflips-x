"""
Consumption look-up tables shipped with eflips-x.

The CSVs in ``data/input/consumption_luts/`` are taken verbatim from the
django-simba example dataset (one regression-derived table per typical bus
length). They follow the column convention expected by
``eflips.model.ConsumptionLut.df_to_consumption_obj``.
"""

from enum import Enum
from pathlib import Path

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
