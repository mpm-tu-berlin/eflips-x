#! /usr/bin/env python3
"""
Plot the SoC of two representative vehicles over a rotation.
"""

import os
import pickle
import warnings
from datetime import timedelta, datetime
from tempfile import gettempdir
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import eflips.eval.output.prepare
import pandas as pd
import seaborn as sns
from eflips.model import (
    ConsistencyWarning,
    Scenario,
    VehicleType,
    Rotation,
    Trip,
    Event,
)
from eflips.tco.data_queries import init_tco_parameters
from eflips.tco.tco_calculator import TCOCalculator
from matplotlib import pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from matplotlib import pyplot as plt
import plotutils

# Configuration
warnings.simplefilter("ignore", ConsistencyWarning)


CACHE_FILE_NAME = "32_soc_cache.pkl"
OUTPUT_PLOT_FILE_DEP = "32_two_socs_dep.pdf"
OUTPUT_PLOT_FILE_OPP = "32_two_socs_opp.pdf"


def create_soc_graph(
    df: pd.DataFrame,
    events: Dict[str, List[Tuple[str, datetime, datetime]]],
    start: datetime,
    end: datetime,
    outfile: str,
):
    """

    The dataframe contains the following columns:
    - time: the time at which the SoC was recorded
    - soc: the state of charge at the given time

    Additionally, a dictionary for the different kinds of events is returned. For each kind of event, a list of Tuples
    with a description of the event, the start time and the end time is returned.

    The kinds of events are:
    - "rotation": A list of rotation names and the time the rotation started and ended
    - "charging": A list of the location of the charging and the time the charging started and ended
    - "trip": A list of the route name and the time the trip started and ended

    """
    fig, ax = plt.subplots(
        figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT)
    )
    ax.plot(df["time"], df["soc"] * 100, label="SoC", color="black")
    ax.set_ylabel(f"State of Charge (SoC) [\%]")
    ax.set_xlabel("Time")
    ax.set_xlim(start, end)
    ax.set_ylim(0, 100)

    # Set the formatter to be just time
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax.xaxis.set_tick_params(rotation=45)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Plot the events as transparent boxes in the background
    colors = sns.color_palette("pastel")
    color_idx = 0
    legend_labels = {"trip": "Driving", "charging": "Charging"}
    added_labels = set()
    
    for event_type, event_list in events.items():
        if event_type == "rotation":
            continue
        
        # Filter events that are within the time range
        visible_events = []
        for event in event_list:
            description, event_start, event_end = event
            if event_end < start or event_start > end:
                continue
            if event_start < start:
                event_start = start
            if event_end > end:
                event_end = end
            visible_events.append((description, event_start, event_end))
        
        for i, (description, event_start, event_end) in enumerate(visible_events):
            # Only add label for the first occurrence of each event type
            label = legend_labels.get(event_type, event_type.capitalize()) if event_type not in added_labels else None
            if label:
                added_labels.add(event_type)
            
            ax.axvspan(
                event_start,
                event_end,
                alpha=0.5,
                color=colors[color_idx],
                label=label,
                linewidth=0,
            )
            
            # Add vertical text labels for charging events
            if event_type == "charging":
                # Determine if this is the last charging event
                is_last_charging = i == len(visible_events) - 1
                text_label = "Depot Charging" if is_last_charging else "Opportunity Charging"
                
                # Position text in the middle of the event span
                mid_time = event_start + (event_end - event_start) / 2
                ax.text(mid_time, 50, text_label, rotation=90, ha='center', va='center', 
                       fontsize=6, alpha=0.7)

        color_idx += 1

    # Add a legend at the bottom left
    ax.legend(
        loc="lower left",
        fancybox=True,
        shadow=True,
    )
    plt.savefig(outfile)

    plt.close()


def main() -> None:
    """Main execution function."""
    assert (
        "DATABASE_URL" in os.environ
    ), "Please set the DATABASE_URL environment variable."

    database_url = os.environ["DATABASE_URL"]
    engine = create_engine(database_url, execution_options={"postgesql_readonly": True})
    session = Session(engine)

    scenario = session.query(Scenario).filter(Scenario.name_short == "OU").one()

    cache_file_path = os.path.join(gettempdir(), CACHE_FILE_NAME)

    if os.path.exists(cache_file_path):
        with open(cache_file_path, "rb") as fp:
            (
                oppo_df,
                oppo_events,
                oppo_start,
                oppo_end,
                dep_df,
                dep_events,
                dep_start,
                dep_end,
            ) = pickle.load(fp)
    else:
        # Load all opportunity charging rotations
        rotations = (
            session.query(Rotation)
            .filter(Rotation.allow_opportunity_charging == True)
            .filter(Rotation.scenario == scenario)
            .all()
        )

        # Go through the rotations until we find one
        # - that is longer than 12 hours
        # - that goes bleow 80% SoC

        MINIMUM_DURATION = timedelta(hours=12)
        MINIMUM_SOC = 0.8

        valid = False
        for rotation in rotations:
            start = (
                session.query(Trip)
                .filter(Trip.rotation == rotation)
                .order_by(Trip.departure_time)
                .first()
            )
            end = (
                session.query(Trip)
                .filter(Trip.rotation == rotation)
                .order_by(Trip.arrival_time.desc())
                .first()
            )
            duration = end.arrival_time - start.departure_time
            if duration < MINIMUM_DURATION:
                continue

            # Also continue if this is the last rotation of the vehicle
            subsequent_rotation = (
                session.query(Rotation)
                .filter(Rotation.vehicle_id == rotation.vehicle_id)
                .filter(Rotation.id != rotation.id)
                .filter(Rotation.scenario == scenario)
                .join(Trip)
                .filter(Trip.departure_time > end.arrival_time)
                .order_by(Trip.departure_time)
                .first()
            )
            if subsequent_rotation is None:
                continue

            events = (
                session.query(Event).join(Trip).filter(Trip.rotation == rotation).all()
            )
            socs = [event.soc_end for event in events if event.soc_end is not None]
            if min(socs) > MINIMUM_SOC:
                continue
            valid = True
            break
        assert valid, "No suitable rotation found."

        oppo_df, oppo_events = eflips.eval.output.prepare.vehicle_soc(
            rotation.vehicle_id, session, ZoneInfo("Europe/Berlin")
        )
        oppo_start = start.departure_time

        # For the end time, plot up to the departure of the subsequent rotation, if any
        subsequent_rotation = (
            session.query(Rotation)
            .filter(Rotation.vehicle_id == rotation.vehicle_id)
            .filter(Rotation.id != rotation.id)
            .filter(Rotation.scenario == scenario)
            .join(Trip)
            .filter(Trip.departure_time > end.arrival_time)
            .order_by(Trip.departure_time)
            .first()
        )
        oppo_end = end = (
            subsequent_rotation.trips[0].departure_time
            if subsequent_rotation is not None
            else end.arrival_time
        )

        # Load all depot charging rotations
        rotations = (
            session.query(Rotation)
            .filter(Rotation.allow_opportunity_charging == False)
            .filter(Rotation.scenario == scenario)
            .all()
        )
        valid = False
        for rotation in rotations:
            start = (
                session.query(Trip)
                .filter(Trip.rotation == rotation)
                .order_by(Trip.departure_time)
                .first()
            )
            end = (
                session.query(Trip)
                .filter(Trip.rotation == rotation)
                .order_by(Trip.arrival_time.desc())
                .first()
            )
            duration = end.arrival_time - start.departure_time
            if duration < MINIMUM_DURATION:
                continue

            # Also continue if this is the last rotation of the vehicle
            subsequent_rotation = (
                session.query(Rotation)
                .filter(Rotation.vehicle_id == rotation.vehicle_id)
                .filter(Rotation.id != rotation.id)
                .filter(Rotation.scenario == scenario)
                .join(Trip)
                .filter(Trip.departure_time > end.arrival_time)
                .order_by(Trip.departure_time)
                .first()
            )
            if subsequent_rotation is None:
                continue

            events = (
                session.query(Event).join(Trip).filter(Trip.rotation == rotation).all()
            )
            socs = [event.soc_end for event in events if event.soc_end is not None]
            if min(socs) > MINIMUM_SOC:
                continue
            valid = True
            break
        assert valid, "No suitable rotation found."

        dep_df, dep_events = eflips.eval.output.prepare.vehicle_soc(
            rotation.vehicle_id, session, ZoneInfo("Europe/Berlin")
        )
        dep_start = start.departure_time
        subsequent_rotation = (
            session.query(Rotation)
            .filter(Rotation.vehicle_id == rotation.vehicle_id)
            .filter(Rotation.id != rotation.id)
            .filter(Rotation.scenario == scenario)
            .join(Trip)
            .filter(Trip.departure_time > end.arrival_time)
            .order_by(Trip.departure_time)
            .first()
        )
        dep_end = (
            subsequent_rotation.trips[0].departure_time
            if subsequent_rotation is not None
            else end.arrival_time
        )

        with open(cache_file_path, "wb") as fp:
            pickle.dump(
                (
                    oppo_df,
                    oppo_events,
                    oppo_start,
                    oppo_end,
                    dep_df,
                    dep_events,
                    dep_start,
                    dep_end,
                ),
                fp,
            )

    create_soc_graph(oppo_df, oppo_events, oppo_start, oppo_end, OUTPUT_PLOT_FILE_OPP)
    create_soc_graph(dep_df, dep_events, dep_start, dep_end, OUTPUT_PLOT_FILE_DEP)


if __name__ == "__main__":
    main()
