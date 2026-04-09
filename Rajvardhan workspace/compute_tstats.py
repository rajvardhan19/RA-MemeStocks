#!/usr/bin/env python3
"""
compute_tstats.py
-----------------
Computes missing t-statistics and p-values for FF4 and multi-frequency
result CSVs, then writes the updated files back to disk.

Run once after the main analysis notebooks have been executed:
    python compute_tstats.py

Updates:
  output/ff4/car_results_ff4.csv          → adds t-stat, p-value, Sig
  output/multi_freq/car_results_5min.csv  → adds t-stat (FF4), p-value, Sig
  output/multi_freq/car_results_10min.csv → adds t-stat (FF4), p-value, Sig
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import timedelta
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = pathlib.Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR     = PROJECT_ROOT / "Minute price and exceed return" / "output_data"
FF_DIR       = PROJECT_ROOT / "FF-Factors"
OUT_FF4      = BASE_DIR / "output" / "ff4"
OUT_MF       = BASE_DIR / "output" / "multi_freq"

# ── Configuration ─────────────────────────────────────────────────────────────
ESTIMATION_WINDOW_DAYS = 120
ESTIMATION_LAG_DAYS    = 2

OUTAGE_EVENTS = [
    {"name": "Outage 1 (Pre-Ban)",  "start": pd.Timestamp("2021-01-27 11:29:00"),
     "end":  pd.Timestamp("2021-01-27 13:40:00"), "duration_minutes": 131, "pre_ban": True},
    {"name": "Outage 2 (Pre-Ban)",  "start": pd.Timestamp("2021-01-27 16:03:00"),
     "end":  pd.Timestamp("2021-01-27 17:01:00"), "duration_minutes": 58,  "pre_ban": True},
    {"name": "Outage 3 (Post-Ban)", "start": pd.Timestamp("2021-01-28 08:44:00"),
     "end":  pd.Timestamp("2021-01-28 10:51:00"), "duration_minutes": 127, "pre_ban": False},
    {"name": "Outage 4 (Post-Ban)", "start": pd.Timestamp("2021-01-28 19:10:00"),
     "end":  pd.Timestamp("2021-01-28 21:00:00"), "duration_minutes": 110, "pre_ban": False},
]

STOCKS = {
    "GME": {"excess_return_file": "GME-minute_price-excess-return.csv",
            "return_col": "gme_return", "price_col": "gme_price"},
    "AMC": {"excess_return_file": "AMC-minute_price-excess-return.csv",
            "return_col": "amc_return", "price_col": "amc_price"},
}

MULTI_MODEL_FILES = {
    "GME": {
        "5min":  "GME-5min-multi-model-excess-return.csv",
        "10min": "GME-10min-multi-model-excess-return.csv",
    },
    "AMC": {
        "5min":  "AMC-5min-multi-model-excess-return.csv",
        "10min": "AMC-10min-multi-model-excess-return.csv",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def sig_stars(p):
    if pd.isna(p):
        return ""
    p = float(p)
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def t_and_p(car, sigma, n):
    """Patell t-statistic: t = CAR / (sigma * sqrt(N))."""
    if pd.isna(car) or pd.isna(sigma) or sigma <= 0 or n <= 0:
        return np.nan, np.nan
    t = float(car) / (float(sigma) * np.sqrt(int(n)))
    p = 2 * (1 - scipy_stats.norm.cdf(abs(t)))
    return t, p


# ── 1. FF4 1-minute analysis ──────────────────────────────────────────────────

def load_stock_data_1min(symbol):
    info = STOCKS[symbol]
    df = pd.read_csv(DATA_DIR / info["excess_return_file"], parse_dates=["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return df


def load_ff_data():
    df = pd.read_csv(FF_DIR / "ff_factors_20201101_20210430_minute.csv",
                     parse_dates=["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return df


def estimate_ff4(stock_df, symbol, ff_df, estimation_end):
    """Fit FF4 model over estimation window; return dict with residual_std."""
    est_start = estimation_end - timedelta(days=ESTIMATION_WINDOW_DAYS)
    window = stock_df.loc[est_start:estimation_end].copy()
    window = window.join(ff_df[["MKT_RF", "SMB", "HML", "MOM"]], how="inner")
    col = STOCKS[symbol]["return_col"]
    window = window.dropna(subset=[col, "MKT_RF", "SMB", "HML", "MOM"])
    y = window[col]
    X = sm.add_constant(window[["MKT_RF", "SMB", "HML", "MOM"]])
    model = sm.OLS(y, X, missing="drop").fit()
    return {
        "sigma": np.sqrt(model.mse_resid),
        "alpha": model.params.get("const", 0.0),
        "beta_mkt": model.params.get("MKT_RF", 0.0),
        "beta_smb": model.params.get("SMB", 0.0),
        "beta_hml": model.params.get("HML", 0.0),
        "beta_mom": model.params.get("MOM", 0.0),
    }


def add_tstats_ff4():
    """Add t-stat, p-value, Sig to car_results_ff4.csv."""
    csv_path = OUT_FF4 / "car_results_ff4.csv"
    df = pd.read_csv(csv_path)

    if "t-stat" in df.columns and "p-value" in df.columns:
        print("FF4 CSV already has t-stat/p-value — recomputing to ensure accuracy.")

    ff_df = load_ff_data()
    first_event = min(e["start"] for e in OUTAGE_EVENTS)
    estimation_end = first_event - timedelta(days=ESTIMATION_LAG_DAYS)

    sigmas = {}
    for symbol in ["GME", "AMC"]:
        stock_df = load_stock_data_1min(symbol)
        params = estimate_ff4(stock_df, symbol, ff_df, estimation_end)
        sigmas[symbol] = params["sigma"]
        print(f"  FF4 sigma  {symbol}: {params['sigma']:.8f}")

    # Build event lookup: N = number of 1-min bars in event window
    event_n = {}
    for symbol in ["GME", "AMC"]:
        stock_df = load_stock_data_1min(symbol)
        for evt in OUTAGE_EVENTS:
            win = stock_df.loc[evt["start"]:evt["end"]]
            event_n[(symbol, evt["name"])] = len(win)

    t_stats, p_vals, sigs = [], [], []
    for _, row in df.iterrows():
        sym = row["Stock"]
        evt = row["Event"]
        car = row["CAR @End"]
        sigma = sigmas.get(sym, np.nan)
        n = event_n.get((sym, evt), 0)
        t, p = t_and_p(car, sigma, n)
        t_stats.append(t)
        p_vals.append(p)
        sigs.append(sig_stars(p))

    df["t-stat"]  = t_stats
    df["p-value"] = p_vals
    df["Sig"]     = sigs
    df.to_csv(csv_path, index=False)
    print(f"  Updated: {csv_path.name}")


# ── 2. Multi-frequency analysis ───────────────────────────────────────────────

def load_multifreq_data(symbol, freq):
    fname = MULTI_MODEL_FILES[symbol][freq]
    df = pd.read_csv(DATA_DIR / fname, parse_dates=["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return df


def compute_sigma_multifreq(df, estimation_end):
    """Compute per-bar sigma from FF4 (or CAPM fallback) residuals."""
    est_start = estimation_end - timedelta(days=ESTIMATION_WINDOW_DAYS)
    window = df.loc[est_start:estimation_end]
    ar_col = "AR_ff4" if "AR_ff4" in window.columns else "AR_capm"
    residuals = window[ar_col].dropna()
    return float(residuals.std()) if len(residuals) > 1 else np.nan


def add_tstats_multifreq(freq, tnum):
    """Add t-stat (FF4), p-value, Sig to car_results_{freq}.csv."""
    csv_path = OUT_MF / f"car_results_{freq}.csv"
    df = pd.read_csv(csv_path)

    first_event = min(e["start"] for e in OUTAGE_EVENTS)
    estimation_end = first_event - timedelta(days=ESTIMATION_LAG_DAYS)

    sigmas = {}
    for symbol in ["GME", "AMC"]:
        mf_df = load_multifreq_data(symbol, freq)
        sigmas[symbol] = compute_sigma_multifreq(mf_df, estimation_end)
        print(f"  {freq} sigma {symbol}: {sigmas[symbol]:.8f}")

    # N = number of bars in event window per stock
    event_n = {}
    for symbol in ["GME", "AMC"]:
        mf_df = load_multifreq_data(symbol, freq)
        for evt in OUTAGE_EVENTS:
            win = mf_df.loc[evt["start"]:evt["end"]]
            event_n[(symbol, evt["name"])] = len(win)

    t_stats_ff4, p_vals_ff4, sigs = [], [], []
    t_stats_capm, p_vals_capm     = [], []

    for _, row in df.iterrows():
        sym = row["Stock"]
        evt = row["Event"]
        sigma = sigmas.get(sym, np.nan)
        n = event_n.get((sym, evt), 0)

        car_ff4  = row.get("CAR @End (FF4)",  np.nan)
        car_capm = row.get("CAR @End (CAPM)", np.nan)

        t_ff4,  p_ff4  = t_and_p(car_ff4,  sigma, n)
        t_capm, p_capm = t_and_p(car_capm, sigma, n)

        t_stats_ff4.append(t_ff4)
        p_vals_ff4.append(p_ff4)
        sigs.append(sig_stars(p_ff4))
        t_stats_capm.append(t_capm)
        p_vals_capm.append(p_capm)

    df["t-stat (FF4)"]   = t_stats_ff4
    df["p-value (FF4)"]  = p_vals_ff4
    df["t-stat (CAPM)"]  = t_stats_capm
    df["p-value (CAPM)"] = p_vals_capm
    df["Sig"]            = sigs
    df.to_csv(csv_path, index=False)
    print(f"  Updated: {csv_path.name}")


# ── 3. CAPM Recovery — verify p-values already present ───────────────────────

def verify_capm_recovery():
    """Verify car_results_capm_recovery.csv has p-values; add if missing."""
    csv_path = BASE_DIR / "output" / "spy-only" / "car_results_capm_recovery.csv"
    if not csv_path.exists():
        print(f"  Not found: {csv_path.name} (skip)")
        return
    df = pd.read_csv(csv_path)
    needed = ["p_value_30", "p_value_1h", "p_value_2h"]
    if all(c in df.columns for c in needed):
        # Add Sig column if missing
        if "Sig" not in df.columns:
            df["Sig"] = df["p_value_2h"].apply(sig_stars)
            df.to_csv(csv_path, index=False)
            print(f"  Added Sig column to {csv_path.name}")
        else:
            print(f"  {csv_path.name}: already complete")
    else:
        print(f"  WARNING: {csv_path.name} missing p-value columns — re-run CAPM analysis")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Computing missing t-statistics ===\n")

    print("1. FF4 1-min results:")
    add_tstats_ff4()

    print("\n2. Multi-frequency 5-min results:")
    add_tstats_multifreq("5min", 3)

    print("\n3. Multi-frequency 10-min results:")
    add_tstats_multifreq("10min", 4)

    print("\n4. CAPM Recovery verification:")
    verify_capm_recovery()

    print("\nDone. Re-run generate_tables.py to rebuild all output tables.")
