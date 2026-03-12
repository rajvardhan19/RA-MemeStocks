#!/usr/bin/env python3
"""
CAPM-based event study: Cumulative Abnormal Returns (CAR) around Reddit
outage windows for GME and AMC meme stocks.

Usage:
    python event_study.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import timedelta
from scipy.stats import mannwhitneyu
import warnings

import config

warnings.filterwarnings("ignore")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_stock_data(symbol):
    """Load excess-return CSV for a stock, returning a datetime-indexed DataFrame."""
    info = config.STOCKS[symbol]
    path = config.DATA_DIR / info["excess_return_file"]
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df = df.set_index("datetime")
    return df


# ── CAPM estimation ───────────────────────────────────────────────────────────

def estimate_capm(stock_data, symbol, estimation_end, window_days=None):
    """
    Estimate CAPM: stock_return = alpha + beta * spy_return
    over an estimation window ending at `estimation_end`.

    Returns dict with alpha, beta, residual_std, r_squared, n_obs.
    """
    if window_days is None:
        window_days = config.ESTIMATION_WINDOW_DAYS

    estimation_start = estimation_end - timedelta(days=window_days)
    window = stock_data.loc[estimation_start:estimation_end].dropna(
        subset=["spy_return", config.STOCKS[symbol]["return_col"]]
    )

    y = window[config.STOCKS[symbol]["return_col"]]
    X = sm.add_constant(window["spy_return"])
    model = sm.OLS(y, X, missing="drop").fit()

    return {
        "alpha": model.params.get("const", 0.0),
        "beta": model.params.get("spy_return", 0.0),
        "residual_std": np.sqrt(model.mse_resid),
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
    }


# ── CAR calculation ───────────────────────────────────────────────────────────

def calculate_car(stock_data, event, model_params, symbol, pre_period_hours=1):
    """
    Calculate Cumulative Abnormal Return for a single outage event.

    Includes a pre-period of `pre_period_hours` before the event start so that
    figures can show whether pre-trends exist. CAR is normalized to 0 at
    event_start so that pre-period drift is visible but post-event metrics
    are unaffected.

    Returns a dict with summary metrics and the full event_data DataFrame,
    or None if no data is available for the window.
    """
    event_start = event["start"]
    event_end_extended = event["end"] + timedelta(hours=2)
    pre_period_start = event_start - timedelta(hours=pre_period_hours)

    event_data = stock_data.loc[pre_period_start:event_end_extended].copy()
    if event_data.empty:
        return None

    return_col = config.STOCKS[symbol]["return_col"]
    price_col = config.STOCKS[symbol]["price_col"]
    alpha = model_params["alpha"]
    beta = model_params["beta"]

    # Expected and abnormal returns
    event_data["expected_return"] = alpha + beta * event_data["spy_return"]
    event_data["AR"] = event_data[return_col] - event_data["expected_return"]

    # Compute cumulative AR from pre-period start, then normalize so CAR = 0
    # at event_start — preserves all post-event metrics while revealing pre-trends.
    event_data["CAR_raw"] = event_data["AR"].cumsum()
    pre_mask = event_data.index <= event_start
    car_offset = event_data.loc[pre_mask, "CAR_raw"].iloc[-1] if pre_mask.any() else 0.0
    event_data["CAR"] = event_data["CAR_raw"] - car_offset

    event_data["time_from_start"] = (
        (event_data.index - event_start).total_seconds() / 60
    )

    # Helper to get CAR closest to a target time
    def car_at(target_time):
        mask = event_data.index <= target_time
        if mask.any():
            return event_data.loc[mask, "CAR"].iloc[-1]
        return np.nan

    car_30min = car_at(event_start + timedelta(minutes=30))
    car_1h = car_at(event_start + timedelta(hours=1))
    car_outage_end = car_at(event["end"])
    car_end_plus_2h = event_data["CAR"].iloc[-1] if len(event_data) else np.nan

    # Price change during outage
    outage_data = stock_data.loc[event_start : event["end"]]
    if len(outage_data) >= 2:
        p_start = outage_data[price_col].iloc[0]
        p_end = outage_data[price_col].iloc[-1]
        price_change_pct = (p_end - p_start) / p_start * 100
    else:
        price_change_pct = np.nan

    # Proper t-statistic: CAR / (sigma * sqrt(N))
    n_minutes = len(event_data.loc[event_start : event["end"]])
    sigma = model_params["residual_std"]
    if sigma > 0 and n_minutes > 0:
        t_stat = car_outage_end / (sigma * np.sqrt(n_minutes))
        p_value = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(t_stat)))
    else:
        t_stat = np.nan
        p_value = np.nan

    return {
        "event_name": event["name"],
        "start_time": event["start"],
        "end_time": event["end"],
        "pre_period_start": pre_period_start,
        "duration_minutes": event["duration_minutes"],
        "pre_ban": event["pre_ban"],
        "CAR_30min": car_30min,
        "CAR_1h": car_1h,
        "CAR_at_outage_end": car_outage_end,
        "CAR_end_plus_2h": car_end_plus_2h,
        "price_change_pct": price_change_pct,
        "t_stat": t_stat,
        "p_value": p_value,
        "event_data": event_data,
    }


def significance_stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(symbols=None):
    """
    Run CAPM event study for the given stock symbols.
    Returns a dict mapping symbol -> list of event result dicts.
    Also saves a summary CSV.
    """
    if symbols is None:
        symbols = ["GME", "AMC"]

    config.ensure_dirs()
    all_results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  {symbol} CAPM Event Study")
        print(f"{'='*60}")

        data = load_stock_data(symbol)
        print(f"Loaded {len(data):,} minute observations")

        # Estimation window ends 1 day before first event
        first_event_start = min(e["start"] for e in config.OUTAGE_EVENTS)
        estimation_end = first_event_start - timedelta(days=1)

        params = estimate_capm(data, symbol, estimation_end)
        print(f"CAPM: alpha={params['alpha']:.6f}, beta={params['beta']:.4f}, "
              f"R²={params['r_squared']:.4f}, n={params['n_obs']}")
        print(f"Residual std: {params['residual_std']:.6f}")

        # Calculate CAR for each outage event
        event_results = []
        for event in config.OUTAGE_EVENTS:
            result = calculate_car(data, event, params, symbol)
            if result:
                event_results.append(result)

        all_results[symbol] = event_results

        # Print results table
        print(f"\n{'Event':<25} {'CAR@30m':>10} {'CAR@1h':>10} "
              f"{'CAR@End':>10} {'Price%':>8} {'t-stat':>8} {'Sig':>5}")
        print("-" * 80)
        for r in event_results:
            stars = significance_stars(r["p_value"])
            print(f"{r['event_name']:<25} {r['CAR_30min']:>10.4f} "
                  f"{r['CAR_1h']:>10.4f} {r['CAR_at_outage_end']:>10.4f} "
                  f"{r['price_change_pct']:>7.2f}% {r['t_stat']:>8.2f} {stars:>5}")

        # Pre-ban vs post-ban comparison
        pre_cars = [r["CAR_at_outage_end"] for r in event_results if r["pre_ban"]]
        post_cars = [r["CAR_at_outage_end"] for r in event_results if not r["pre_ban"]]
        if len(pre_cars) >= 1 and len(post_cars) >= 1:
            print(f"\nPre-ban avg CAR:  {np.mean(pre_cars):.4f}")
            print(f"Post-ban avg CAR: {np.mean(post_cars):.4f}")
            if len(pre_cars) >= 2 and len(post_cars) >= 2:
                stat, p = mannwhitneyu(pre_cars, post_cars, alternative="two-sided")
                print(f"Mann-Whitney U: stat={stat:.4f}, p={p:.4f} "
                      f"{significance_stars(p)}")

    # Save summary CSV
    rows = []
    for symbol, results in all_results.items():
        for r in results:
            rows.append({
                "stock": symbol,
                "event_name": r["event_name"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "duration_minutes": r["duration_minutes"],
                "pre_ban": r["pre_ban"],
                "CAR_30min": r["CAR_30min"],
                "CAR_1h": r["CAR_1h"],
                "CAR_at_outage_end": r["CAR_at_outage_end"],
                "CAR_end_plus_2h": r["CAR_end_plus_2h"],
                "price_change_pct": r["price_change_pct"],
                "t_stat": r["t_stat"],
                "p_value": r["p_value"],
            })

    summary_df = pd.DataFrame(rows)
    out_path = config.OUTPUT_DIR / "car_results_capm.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    run_analysis()
