import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import plotutils

SMART_CHARGING_DEPOT_POWERS = "17_smart_charging_results.xlsx"
NO_SMART_CHARGING_DEPOT_POWERS = "19_no_smart_charging_results.xlsx"

if __name__ == "__main__":
    smart_charging_depot_powers = pd.read_excel(SMART_CHARGING_DEPOT_POWERS)
    no_smart_charging_depot_powers = pd.read_excel(NO_SMART_CHARGING_DEPOT_POWERS)

    # Clean depot column names by removing "Depot at Betriebshof "
    for df in [smart_charging_depot_powers, no_smart_charging_depot_powers]:
        df.columns = [
            (
                col.replace("Depot at Betriebshof ", "")
                if "Depot at Betriebshof " in col
                else col
            )
            for col in df.columns
        ]

    # Add charging type column to each DataFrame
    smart_charging_depot_powers["charging_type"] = "smart"
    no_smart_charging_depot_powers["charging_type"] = "naive"

    # Identify depot columns (all columns except 'scenario' and 'charging_type')
    depot_columns = [
        col
        for col in smart_charging_depot_powers.columns
        if col != "scenario" and col != "charging_type"
    ]

    # Divide depot power values by 1000 to convert to MW
    for df in [smart_charging_depot_powers, no_smart_charging_depot_powers]:
        for depot in depot_columns:
            df[depot] = df[depot] / 1000

    # Concatenate the DataFrames
    combined_df = pd.concat(
        [smart_charging_depot_powers, no_smart_charging_depot_powers]
    )

    # Sort by scenario name
    combined_df = combined_df.sort_values(["scenario", "charging_type"])

    # Rearrange columns to desired order
    column_order = ["scenario", "charging_type"] + depot_columns
    combined_df = combined_df[column_order]

    # Create DataFrames for smart and naive charging to facilitate comparison
    smart_df = combined_df[combined_df["charging_type"] == "smart"].set_index(
        "scenario"
    )
    naive_df = combined_df[combined_df["charging_type"] == "naive"].set_index(
        "scenario"
    )

    # Calculate differences
    differences = {}
    percent_differences = []
    absolute_differences = []

    for depot in depot_columns:
        # Calculate differences for each scenario and depot
        for scenario in smart_df.index:
            if scenario in naive_df.index:
                smart_value = smart_df.loc[scenario, depot]
                naive_value = naive_df.loc[scenario, depot]

                # Absolute difference in MW
                abs_diff = naive_value - smart_value
                absolute_differences.append(abs_diff)

                # Percent difference
                if naive_value != 0:  # Avoid division by zero
                    percent_diff = (naive_value - smart_value) / naive_value * 100
                    percent_differences.append(percent_diff)

    # Calculate statistics
    avg_percent_diff = np.mean(percent_differences)
    max_abs_diff = np.max(absolute_differences)
    min_abs_diff = np.min(absolute_differences)

    # Output results
    print(f"Combined DataFrame:")
    print(combined_df)
    print("\nStatistics:")
    print(f"Average difference (naive vs smart) in percent: {avg_percent_diff:.2f}%")
    print(f"Maximum absolute difference (MW): {max_abs_diff:.2f} MW")
    print(f"Minimum absolute difference (MW): {min_abs_diff:.2f} MW")

    # Save to Excel
    combined_df.to_excel("19c_combined_charging_results.xlsx", index=False)

    # Create a summary DataFrame and save it
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Average Difference (%)",
                "Maximum Absolute Difference (MW)",
                "Minimum Absolute Difference (MW)",
            ],
            "Value": [
                f"{avg_percent_diff:.2f}%",
                f"{max_abs_diff:.2f}",
                f"{min_abs_diff:.2f}",
            ],
        }
    )

    # Create a dataframe with only the smart charging values
    smart_charging_df = combined_df[combined_df["charging_type"] == "smart"]
    # Remove the charging_type column
    smart_charging_df = smart_charging_df.drop(columns=["charging_type"])
    # Make the scenario column the index
    smart_charging_df = smart_charging_df.set_index("scenario")

    # Shorten the column nsames.
    # replace all "straße" with "str."
    smart_charging_df.columns = [
        col.replace("straße", "str.") for col in smart_charging_df.columns
    ]
    # Replace all " " with line breaks
    smart_charging_df.columns = [
        col.replace(" ", "\n") for col in smart_charging_df.columns
    ]

    # Rename the keys
    # "OU" -> "Existing Blocks"
    # "TERM" -> "Small Batteries"
    # "DEP" -> "Depot Charging"
    smart_charging_df = smart_charging_df.rename(
        index={
            "OU": "Existing Blocks Unchanged",
            "TERM": "Small Batteries \& Termini",
            "DEP": "Depot Charging Only",
        }
    )

    # Assuming 'df' is your original DataFrame
    df_long = smart_charging_df.reset_index().melt(
        id_vars="scenario", var_name="depot", value_name="power"
    )

    fix, ax = plt.subplots(
        figsize=(plotutils.NORMAL_PLOT_WIDTH, plotutils.NORMAL_PLOT_HEIGHT)
    )
    sns.barplot(x="depot", y="power", hue="scenario", data=df_long)

    # Lengend header should be "Scenario"
    plt.legend(title="Scenario")

    # Increase y limit to 10 MW
    plt.ylim(0, 10)

    # No x label
    plt.xlabel("")
    plt.ylabel(r"Power [MW]")

    # Rotate depot labels for better readability
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.savefig("19c_smart_charging_table.pdf")
