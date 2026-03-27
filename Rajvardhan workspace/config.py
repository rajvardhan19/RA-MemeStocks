"""
Centralized configuration for meme stock excess return & outage event study.
"""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Minute price and exceed return" / "output_data"
FF_DIR = PROJECT_ROOT / "FF-Factors"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [OUTPUT_DIR, FIGURES_DIR, FIGURES_DIR / "gme", FIGURES_DIR / "amc"]:
        d.mkdir(parents=True, exist_ok=True)


# ── Stock column mappings ─────────────────────────────────────────────────────
STOCKS = {
    "GME": {
        "excess_return_file": "GME-minute_price-excess-return.csv",
        "minute_price_file": "GME-202011-202104-minute_price.csv",
        "price_col": "gme_price",
        "return_col": "gme_return",
    },
    "AMC": {
        "excess_return_file": "AMC-minute_price-excess-return.csv",
        "minute_price_file": "AMC-202011-202104-minute_price.csv",
        "price_col": "amc_price",
        "return_col": "amc_return",
    },
}

# ── Event definitions ─────────────────────────────────────────────────────────
ROBINHOOD_BAN_DATE = pd.Timestamp("2021-01-28 09:30:00")
ESTIMATION_WINDOW_DAYS = 120
ESTIMATION_LAG_DAYS = 2    # estimation_end = first_event_start - 2 days ([-120, -2] window)

OUTAGE_EVENTS = [
    {
        "name": "Outage 1 (Pre-Ban)",
        "start": pd.Timestamp("2021-01-27 11:29:00"),
        "end": pd.Timestamp("2021-01-27 13:40:00"),
        "duration_minutes": 131,
    },
    {
        "name": "Outage 2 (Pre-Ban)",
        "start": pd.Timestamp("2021-01-27 16:03:00"),
        "end": pd.Timestamp("2021-01-27 17:01:00"),
        "duration_minutes": 58,
    },
    {
        "name": "Outage 3 (Post-Ban)",
        "start": pd.Timestamp("2021-01-28 08:44:00"),
        "end": pd.Timestamp("2021-01-28 10:51:00"),
        "duration_minutes": 127,
    },
    {
        "name": "Outage 4 (Post-Ban)",
        "start": pd.Timestamp("2021-01-28 19:10:00"),
        "end": pd.Timestamp("2021-01-28 21:00:00"),
        "duration_minutes": 110,
    },
]

# Add pre_ban flag based on midpoint of outage relative to ban date
for _evt in OUTAGE_EVENTS:
    midpoint = _evt["start"] + (_evt["end"] - _evt["start"]) / 2
    _evt["pre_ban"] = midpoint < ROBINHOOD_BAN_DATE

# ── Matplotlib defaults ───────────────────────────────────────────────────────
def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })
