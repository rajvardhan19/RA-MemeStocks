#!/usr/bin/env python3
"""
Visualization suite for meme stock outage event study.

Generates:
  1. CAR comparison bar charts (per stock)
  2. Detailed per-event price + CAR plots
  3. Full excess-return time series
  4. Pre-ban vs post-ban summary across both stocks

Usage:
    python visualizations.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

import config
import event_study

config.set_plot_style()

# ── Colors ────────────────────────────────────────────────────────────────────
PRE_BAN_COLOR = "#D32F2F"   # red
POST_BAN_COLOR = "#1976D2"  # blue


def _sig_stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ── 1. CAR comparison bar chart ──────────────────────────────────────────────

def plot_car_comparison(event_results, symbol):
    """2x2 bar chart: CAR at 30min, 1h, outage end, and price change."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"{symbol} — Cumulative Abnormal Returns Around Reddit Outages",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("CAR_30min", "CAR at 30 Minutes"),
        ("CAR_1h", "CAR at 1 Hour"),
        ("CAR_at_outage_end", "CAR at Outage End"),
        ("price_change_pct", "Price Change During Outage (%)"),
    ]

    for ax, (key, title) in zip(axes.flat, metrics):
        names = [r["event_name"] for r in event_results]
        values = [r[key] for r in event_results]
        colors = [PRE_BAN_COLOR if r["pre_ban"] else POST_BAN_COLOR
                  for r in event_results]

        bars = ax.bar(range(len(names)), values, color=colors, alpha=0.85,
                      edgecolor="black", linewidth=0.5)

        # Add significance stars for CAR metrics
        if key != "price_change_pct":
            for i, (bar, r) in enumerate(zip(bars, event_results)):
                stars = _sig_stars(r["p_value"])
                if stars:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height(), stars,
                            ha="center", va="bottom", fontsize=11,
                            fontweight="bold")

        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" (Pre-Ban)", "\n(Pre)")
                             .replace(" (Post-Ban)", "\n(Post)")
                            for n in names], fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=PRE_BAN_COLOR, label="Pre-Ban"),
                       Patch(facecolor=POST_BAN_COLOR, label="Post-Ban")]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=11, frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = config.FIGURES_DIR / symbol.lower() / f"car_comparison_capm.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── 2. Detailed per-event plots ──────────────────────────────────────────────

def plot_event_detail(result, symbol, event_num):
    """Two-panel figure: stock price + CAR path for a single outage event."""
    ed = result["event_data"]
    if ed.empty:
        return

    price_col = config.STOCKS[symbol]["price_col"]
    event_start = result["start_time"]
    event_end = result["end_time"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"{symbol} — {result['event_name']}", fontsize=13,
                 fontweight="bold")

    # Top: price
    ax1.plot(ed.index, ed[price_col], color="black", linewidth=1.0)
    ax1.axvline(event_start, color=PRE_BAN_COLOR, linestyle="--",
                alpha=0.7, label="Outage Start")
    ax1.axvline(event_end, color=POST_BAN_COLOR, linestyle="--",
                alpha=0.7, label="Outage End")
    ax1.axvspan(event_start, event_end, alpha=0.08, color="red")
    ax1.set_ylabel(f"{symbol} Price ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    # Bottom: CAR
    ax2.plot(ed.index, ed["CAR"], color="#2E7D32", linewidth=1.2)
    ax2.axvline(event_start, color=PRE_BAN_COLOR, linestyle="--", alpha=0.7)
    ax2.axvline(event_end, color=POST_BAN_COLOR, linestyle="--", alpha=0.7)
    ax2.axvspan(event_start, event_end, alpha=0.08, color="red")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("CAR")
    ax2.set_xlabel("Time")
    ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.tight_layout()

    tag = result["event_name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
    out = config.FIGURES_DIR / symbol.lower() / f"event_{event_num}_{tag}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── 3. Excess return time series ──────────────────────────────────────────────

def plot_excess_return_timeseries(symbol):
    """Full Nov 2020 – Apr 2021 excess return with Jan 27-28 highlighted."""
    data = event_study.load_stock_data(symbol)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"{symbol} — Minute-Level Excess Returns (vs SPY)",
                 fontsize=13, fontweight="bold")

    # Top: raw excess return
    ax1.plot(data.index, data["excess_return"], color="steelblue",
             alpha=0.5, linewidth=0.3)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Excess Return (per minute)")

    # Highlight event cluster
    cluster_start = pd.Timestamp("2021-01-27 00:00")
    cluster_end = pd.Timestamp("2021-01-29 00:00")
    ax1.axvspan(cluster_start, cluster_end, alpha=0.12, color="red",
                label="Jan 27-28 (Outage Period)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Bottom: rolling 1-hour cumulative excess return
    col_1h = "excess_return_cum_1h"
    if col_1h in data.columns:
        ax2.plot(data.index, data[col_1h], color="darkgreen",
                 alpha=0.6, linewidth=0.4)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.axvspan(cluster_start, cluster_end, alpha=0.12, color="red")
    ax2.set_ylabel("Rolling 1h Cumulative Excess Return")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    out = config.FIGURES_DIR / symbol.lower() / f"excess_return_timeseries.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── 4. Pre-ban vs post-ban summary ───────────────────────────────────────────

def plot_pre_vs_post_ban_summary(all_results):
    """Grouped bar chart comparing GME and AMC CARs, pre-ban vs post-ban."""
    fig, ax = plt.subplots(figsize=(10, 6))

    symbols = list(all_results.keys())
    x = np.arange(len(symbols))
    width = 0.3

    for i, label, color in [(0, "Pre-Ban", PRE_BAN_COLOR),
                             (1, "Post-Ban", POST_BAN_COLOR)]:
        means = []
        for sym in symbols:
            cars = [r["CAR_at_outage_end"] for r in all_results[sym]
                    if r["pre_ban"] == (label == "Pre-Ban")]
            means.append(np.mean(cars) if cars else 0)
        ax.bar(x + i * width - width / 2, means, width, label=label,
               color=color, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(symbols, fontsize=12)
    ax.set_ylabel("Average CAR at Outage End")
    ax.set_title("Pre-Ban vs Post-Ban Average CAR — GME & AMC",
                 fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = config.FIGURES_DIR / "pre_vs_post_ban_summary.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ── Generate all ──────────────────────────────────────────────────────────────

def generate_all():
    """Run event study and generate all visualizations."""
    config.ensure_dirs()

    # Run analysis to get event-level data
    all_results = event_study.run_analysis()

    print("\n" + "=" * 60)
    print("  Generating Visualizations")
    print("=" * 60)

    for symbol, results in all_results.items():
        print(f"\n{symbol}:")
        plot_car_comparison(results, symbol)
        for i, r in enumerate(results, 1):
            plot_event_detail(r, symbol, i)
        plot_excess_return_timeseries(symbol)

    plot_pre_vs_post_ban_summary(all_results)
    print(f"\nAll figures saved to: {config.FIGURES_DIR}")


if __name__ == "__main__":
    generate_all()
