import pandas as pd
from matplotlib import dates as mdates
from matplotlib import pyplot as plt

import plotutils

SMART_CHARGING_DF_PATH = "17_power_and_occupancy_smart.pkl"
NO_SMART_CHARGING_DF_PATH = "19_power_and_occupanc_no_smart.pkl"

if __name__ == "__main__":
    smart_charging_df = pd.read_pickle(SMART_CHARGING_DF_PATH)
    no_smart_charging_df = pd.read_pickle(NO_SMART_CHARGING_DF_PATH)

    # Both DFs have the following columns:
    # - time: the time at which the power and occupancy was recorded
    # - power: the power at the given time
    # - occupancy_charging: the summed occupancy (actively charing vehicles) of the area(s) at the given time
    # - occupancy_total: the summed occupancy of the area(s) at the given time, including all events

    # We want to plot the powers of both dataframes in the same plot

    fig, ax = plt.subplots(
        nrows=1, figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT)
    )
    ax.plot(
        no_smart_charging_df["time"],
        no_smart_charging_df["power"],
        label=f"Uncontrolled charging (peak: {no_smart_charging_df['power'].max() / 1000:.1f} MW)",
    )
    ax.plot(
        smart_charging_df["time"],
        smart_charging_df["power"],
        label=f"Peak shaving (peak: {smart_charging_df['power'].max() / 1000:.1f} MW)",
    )

    # Configure x-axis to show weekdays and midnight
    # Set major ticks at midnight
    ax.xaxis.set_major_locator(mdates.DayLocator())
    # Format the ticks to show weekday name
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))

    # Optional: Add minor ticks for hours or other intervals if needed
    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    ax.set_xlabel("Day of week")
    ax.set_ylabel(r"Power $\left[ kW \right]$")

    # Add two hlines, at each peak, colored the same as the line
    ax.axhline(
        no_smart_charging_df["power"].max(),
        color="C0",
        linestyle="--",
    )
    ax.axhline(
        smart_charging_df["power"].max(),
        color="C1",
        linestyle="--",
    )
    plt.tight_layout()
    ax.legend()

    plt.savefig("19b_smart_charging_power_comparison.pdf")
