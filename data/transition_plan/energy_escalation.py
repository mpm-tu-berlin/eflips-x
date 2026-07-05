"""
Energy price escalation factor fitting.

Main function:
    fit_escalation(path, column, predict_year=None, iqr_k=3.0)

Returns:
    dict with 'escalation_factor', 'base_price', 'base_year',
    and optionally 'predicted_price' if predict_year is given.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# Known geopolitical shock windows per column (inclusive year ranges)
SHOCK_WINDOWS = {
    "electricity_ct_per_kwh": [(2022, 2023)],
    "diesel_eur_per_liter":   [(2008, 2009), (2022, 2023)],
}


def _detect_outliers_iqr(values: np.ndarray, k: float) -> np.ndarray:
    """Boolean mask: True where value is beyond median ± k*IQR."""
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (values < q1 - k * iqr) | (values > q3 + k * iqr)


def _shock_dummy(years: np.ndarray, column: str) -> np.ndarray:
    """Binary dummy = 1 for documented geopolitical shock years."""
    dummy = np.zeros(len(years), dtype=float)
    for start, end in SHOCK_WINDOWS.get(column, []):
        dummy[(years >= start) & (years <= end)] = 1.0
    return dummy


def _ols_log_linear(years: np.ndarray,
                    prices: np.ndarray,
                    dummy: np.ndarray) -> tuple[float, float, float]:
    """
    OLS: log(price) ~ intercept + beta*(year - year[0]) + gamma*dummy
    Returns (intercept, beta, r_squared).
    beta is the log annual growth rate; exp(beta) = escalation factor.
    """
    log_p = np.log(prices)
    t = years - years[0]
    X = np.column_stack([np.ones(len(t)), t, dummy])
    coeffs, *_ = np.linalg.lstsq(X, log_p, rcond=None)
    intercept, beta, gamma = coeffs
    y_hat = X @ coeffs
    ss_res = np.sum((log_p - y_hat) ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(intercept), float(beta), float(r2)


def fit_escalation(path: str,
                   column: str,
                   predict_year: int | None = None,
                   iqr_k: float = 3.0) -> dict:
    """
    Fit a constant annual price escalation factor from historical energy prices.

    Steps
    -----
    1. Load CSV and drop rows where the target column is NaN.
    2. Flag statistical outliers via IQR (threshold: k * IQR beyond Q1/Q3).
    3. Apply a binary shock dummy for known geopolitical events (keeps the
       data point in the fit but prevents the spike inflating the trend).
    4. Fit log-linear OLS: log(price) ~ trend + shock_dummy.
    5. Derive escalation factor as exp(trend_coefficient).

    Parameters
    ----------
    path : str
        Path to CSV. Must contain 'year' and the target column.
    column : str
        Price column name (e.g. 'electricity_ct_per_kwh').
    predict_year : int, optional
        Year to project to. Uses base_price * escalation_factor^n.
    iqr_k : float
        IQR multiplier for outlier flagging. Default 3.0 catches only
        extreme spikes without touching near-normal high years.

    Returns
    -------
    dict
        base_year          : int   — most recent clean year (projection anchor)
        base_price         : float — observed price in base_year
        shock_years        : list  — years flagged by IQR or known shock windows
        n_obs_total        : int   — total observations with valid price data
        n_clean_obs        : int   — observations after excluding shock years
        escalation_factor  : float — annual multiplier (e.g. 1.018 = +1.8 % p.a.)
        escalation_pct     : float — same as percentage
        r_squared          : float — OLS fit quality on log-price series
        predicted_price    : float | None
    """
    df = pd.read_csv(path)

    if "year" not in df.columns:
        raise ValueError("CSV must contain a 'year' column.")
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    df = df[["year", column]].dropna(subset=[column]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)

    years  = df["year"].values.astype(float)
    prices = df[column].values.astype(float)

    # Outlier flags
    iqr_mask   = _detect_outliers_iqr(prices, iqr_k)
    dummy      = _shock_dummy(years.astype(int), column)
    shock_mask = dummy.astype(bool)
    excluded   = iqr_mask | shock_mask

    # OLS with dummy (all data used; dummy absorbs the spike)
    _, beta, r2 = _ols_log_linear(years, prices, dummy)
    escalation_factor = np.exp(beta)

    # Base: most recent non-excluded observation
    clean_idx  = np.where(~excluded)[0]
    base_idx   = clean_idx[-1] if len(clean_idx) else len(years) - 1
    base_year  = int(years[base_idx])
    base_price = float(prices[base_idx])

    # Optional prediction
    predicted_price = None
    if predict_year is not None:
        n = predict_year - base_year
        predicted_price = round(base_price * (escalation_factor ** n), 4)

    return {
        "base_year":         base_year,
        "base_price":        round(base_price, 4),
        "shock_years":       sorted(int(y) for y in years[excluded]),
        "n_obs_total":       len(years),
        "n_clean_obs":       int((~excluded).sum()),
        "escalation_factor": round(float(escalation_factor), 6),
        "escalation_pct":    round((float(escalation_factor) - 1) * 100, 3),
        "r_squared":         round(r2, 4),
        "predicted_price":   predicted_price,
    }


if __name__ == "__main__":
    PATH = "energy_prices.csv"

    print("=== Electricity ===")
    r = fit_escalation(PATH, "electricity_ct_per_kwh", predict_year=2035)
    for k, v in r.items():
        print(f"  {k}: {v}")

    print("\n=== Diesel ===")
    r = fit_escalation(PATH, "diesel_eur_per_liter", predict_year=2035)
    for k, v in r.items():
        print(f"  {k}: {v}")
