#! /usr/bin/env python3

"""
This script loads the "consumption_lut.xlsx" containing the consumption for the 18 meter buses, turns it into an eFLIPS-
compatible consumption LUT and sets it as the consumption LUT for the vehicle type "GN" in all scenarios.
"""

import os
from datetime import datetime, timedelta
from math import sqrt

import numpy as np
import pandas as pd
from eflips.model import (
    VehicleType,
    VehicleClass,
    ConsumptionLut,
    Scenario,
    Temperatures,
)
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

if __name__ == "__main__":
    if "DATABASE_URL" not in os.environ:
        raise ValueError("Please set the DATABASE_URL environment variable.")
    DATABASE_URL = os.environ["DATABASE_URL"]
    engine = create_engine(DATABASE_URL)
    session = Session(engine)

    tz = ZoneInfo("Europe/Berlin")
    datetimes = [datetime(1971, 1, 1, tzinfo=tz), datetime(2037, 12, 21, tzinfo=tz)]
    temps = [-12, -12]

    try:
        scenarios = session.query(Scenario).all()
        for scenario in scenarios:
            scenario_temperatures = Temperatures(
                scenario_id=scenario.id,
                name="-12 °C",
                use_only_time=False,
                datetimes=datetimes,
                data=temps,
            )

            session.add(scenario_temperatures)
    except:
        session.rollback()
        raise
    finally:
        session.commit()
