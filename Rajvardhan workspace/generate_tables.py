"""
Generate formatted LaTeX and CSV table files from analysis result CSVs.
Output goes into: output-tables/

Tables generated:
  1  CAPM CAR results (1-min bars)
  2  FF4 CAR results (1-min bars)          ← now includes t-stat / p-value
  3  FF4 vs CAPM CAR results (5-min bars)  ← now includes t-stat / p-value
  4  FF4 vs CAPM CAR results (10-min bars) ← now includes t-stat / p-value
  5  Post-outage recovery CARs (5-min)
  6  Post-outage recovery CARs (10-min)
  7  Post-outage recovery CARs (FF4, 1-min)
  8  CAPM pre-ban vs post-ban summary
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "output-tables")
os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def pct(x, decimals=2):
    """Format a decimal as a percentage string."""
    if pd.isna(x) or x == "":
        return ""
    return f"{float(x)*100:.{decimals}f}\\%"

def fmt(x, decimals=3):
    if pd.isna(x) or x == "":
        return ""
    return f"{float(x):.{decimals}f}"

def sig_stars(p):
    """Return significance stars from a p-value."""
    if pd.isna(p):
        return ""
    p = float(p)
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

def write_latex(tex: str, fname: str):
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Written: {fname}")

def write_csv(df: pd.DataFrame, fname: str):
    path = os.path.join(OUT_DIR, fname)
    df.to_csv(path, index=False)
    print(f"  Written: {fname}")

def latex_header(caption: str, label: str, cols: str) -> str:
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{cols}}}\n"
        "    \\hline\\hline\n"
    )

def latex_footer(extra_note: str = "") -> str:
    base = (
        "    \\hline\\hline\n"
        "  \\end{tabular}\n"
        "  \\\\[4pt]\n"
        "  {\\small *** $p<0.01$, ** $p<0.05$, * $p<0.10$. "
        "CAR = cumulative abnormal return relative to the estimation window "
        "(120 trading days ending 2 days before each event). "
        "t-statistics computed via Patell methodology: $t = \\text{CAR} / (\\hat{\\sigma} \\sqrt{N})$, "
        "where $\\hat{\\sigma}$ is the residual standard deviation from the estimation window "
        "and $N$ is the number of return observations in the event window. "
        "p-values are two-tailed under the standard normal distribution."
    )
    if extra_note:
        base += " " + extra_note
    base += "}\n\\end{table}\n"
    return base


# ── Table 1: CAPM CAR Results ────────────────────────────────────────────────

def table_capm():
    """
    Table 1 — CAPM Cumulative Abnormal Returns During Reddit Outage Events (1-min bars).

    Reports event-window CARs at four horizons (30 min, 1 h, outage end, end+2 h)
    alongside the raw price change during the outage.  The Patell t-statistic and
    two-tailed p-value test whether the CAR at outage end is statistically different
    from zero under the CAPM single-factor expected return.  Significance stars
    (Sig column) summarise the p-value at conventional thresholds.
    """
    src = os.path.join(BASE_DIR, "output", "spy-only", "car_results_capm.csv")
    df  = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
        sig = str(r.get("Sig", "")).strip() if not pd.isna(r.get("Sig", np.nan)) else ""
        rows.append({
            "Stock":       r["Stock"],
            "Event":       r["Event"],
            "Dur. (min)":  int(r["Duration (min)"]),
            "Pre-Ban":     "Yes" if r["Pre-Ban"] else "No",
            "CAR @30m":    f"{float(r['CAR @30min'])*100:.2f}\\%",
            "CAR @1h":     f"{float(r['CAR @1h'])*100:.2f}\\%",
            "CAR @End":    f"{float(r['CAR @End'])*100:.2f}\\%",
            "CAR @End+2h": f"{float(r['CAR @End+2h'])*100:.2f}\\%",
            "Price Chg %": f"{float(r['Price Change %']):.2f}\\%",
            "t-stat":      f"{float(r['t-stat']):.3f}",
            "p-value":     f"{float(r['p-value']):.4f}",
            "Sig":         sig,
        })
    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")), "table1_capm_results.csv")

    # LaTeX
    cols = "llcccccccrrc"
    tex  = latex_header(
        "CAPM Cumulative Abnormal Returns During Reddit Outage Events (1-min bars)",
        "tab:capm_car",
        cols
    )
    header = ("    Stock & Event & Dur. & Pre-Ban & CAR @30m & CAR @1h & CAR @End "
              "& CAR @End+2h & Price Chg & t-stat & p-val & Sig \\\\\n    \\hline\n")
    tex += header
    prev_stock = None
    for row in rows:
        if prev_stock and row["Stock"] != prev_stock:
            tex += "    \\hline\n"
        prev_stock = row["Stock"]
        tex += (
            f"    {row['Stock']} & {row['Event']} & {row['Dur. (min)']} & {row['Pre-Ban']} "
            f"& {row['CAR @30m']} & {row['CAR @1h']} & {row['CAR @End']} "
            f"& {row['CAR @End+2h']} & {row['Price Chg %']} "
            f"& {row['t-stat']} & {row['p-value']} & {row['Sig']} \\\\\n"
        )
    tex += latex_footer()
    write_latex(tex, "table1_capm_results.tex")


# ── Table 2: FF4 CAR Results (1-min bars) ────────────────────────────────────

def table_ff4():
    """
    Table 2 — Fama-French 4-Factor Cumulative Abnormal Returns During Reddit Outage
    Events (1-min bars).

    Uses the four-factor model (market, size, value, momentum) estimated over the
    same 120-day pre-event window.  After-hours minutes where FF factor data are
    unavailable fall back to a market-model using SPY as the sole factor.
    'FF Mins/Total' shows how many outage minutes had genuine four-factor coverage.
    t-statistics and p-values are based on the FF4 residual standard deviation.
    """
    src = os.path.join(BASE_DIR, "output", "ff4", "car_results_ff4.csv")
    df  = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
        sig = str(r.get("Sig", "")).strip() if not pd.isna(r.get("Sig", np.nan)) else ""
        t_val = r.get("t-stat", np.nan)
        p_val = r.get("p-value", np.nan)
        rows.append({
            "Stock":         r["Stock"],
            "Event":         r["Event"],
            "Pre-Ban":       "Yes" if r["Pre-Ban"] else "No",
            "CAR @30m":      f"{float(r['CAR @30min'])*100:.2f}\\%",
            "CAR @1h":       f"{float(r['CAR @1h'])*100:.2f}\\%",
            "CAR @End":      f"{float(r['CAR @End'])*100:.2f}\\%",
            "CAR @End+2h":   f"{float(r['CAR @End+2h'])*100:.2f}\\%",
            "Price Chg %":   f"{float(r['Price Change %']):.2f}\\%",
            "t-stat":        fmt(t_val, 3) if not pd.isna(t_val) else "—",
            "p-value":       f"{float(p_val):.4f}" if not pd.isna(p_val) else "—",
            "Sig":           sig,
            "FF Mins/Total": r["FF Mins / Total"],
        })
    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")), "table2_ff4_results.csv")

    cols = "llcccccccrrc"
    tex  = latex_header(
        "Fama-French 4-Factor Cumulative Abnormal Returns During Reddit Outage Events (1-min bars)",
        "tab:ff4_car",
        cols
    )
    header = ("    Stock & Event & Pre-Ban & CAR @30m & CAR @1h & CAR @End "
              "& CAR @End+2h & Price Chg & t-stat & p-val & Sig & FF Mins/Total \\\\\n    \\hline\n")
    tex += header
    prev_stock = None
    for row in rows:
        if prev_stock and row["Stock"] != prev_stock:
            tex += "    \\hline\n"
        prev_stock = row["Stock"]
        tex += (
            f"    {row['Stock']} & {row['Event']} & {row['Pre-Ban']} "
            f"& {row['CAR @30m']} & {row['CAR @1h']} & {row['CAR @End']} "
            f"& {row['CAR @End+2h']} & {row['Price Chg %']} "
            f"& {row['t-stat']} & {row['p-value']} & {row['Sig']} & {row['FF Mins/Total']} \\\\\n"
        )
    tex += latex_footer(
        "FF Mins/Total = minutes with genuine four-factor coverage / total outage minutes."
    )
    write_latex(tex, "table2_ff4_results.tex")


# ── Table 3 & 4: Multi-Frequency FF4 vs CAPM ────────────────────────────────

def table_multifreq(freq_label: str, fname_csv: str, tnum: int):
    """
    Tables 3 & 4 — FF4 vs CAPM Cumulative Abnormal Returns at 5-min and 10-min bar
    frequencies.

    Each row shows both model CARs at key horizons together with their respective
    Patell t-statistics and p-values.  Coarser bar frequencies reduce the number of
    observations per event window, which affects the precision of the t-statistic.
    'FF4 Bars/Total' indicates the fraction of bars that had full four-factor data.
    """
    src = os.path.join(BASE_DIR, "output", "multi_freq", fname_csv)
    df  = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
        def safe_pct(col):
            v = r.get(col, np.nan)
            if pd.isna(v) or str(v).strip() == "":
                return "—"
            return f"{float(v)*100:.2f}\\%"

        def safe_t(col):
            v = r.get(col, np.nan)
            if pd.isna(v) or str(v).strip() == "":
                return "—"
            return f"{float(v):.3f}"

        def safe_p(col):
            v = r.get(col, np.nan)
            if pd.isna(v) or str(v).strip() == "":
                return "—"
            return f"{float(v):.4f}"

        sig = str(r.get("Sig", "")).strip() if not pd.isna(r.get("Sig", np.nan)) else ""

        rows.append({
            "Stock":           r["Stock"],
            "Event":           r["Event"],
            "Pre-Ban":         "Yes" if r["Pre-Ban"] else "No",
            "FF4 @30m":        safe_pct("CAR @30min (FF4)"),
            "FF4 @1h":         safe_pct("CAR @1h (FF4)"),
            "FF4 @End":        safe_pct("CAR @End (FF4)"),
            "t-stat (FF4)":    safe_t("t-stat (FF4)"),
            "p-val (FF4)":     safe_p("p-value (FF4)"),
            "CAPM @30m":       safe_pct("CAR @30min (CAPM)"),
            "CAPM @End":       safe_pct("CAR @End (CAPM)"),
            "t-stat (CAPM)":   safe_t("t-stat (CAPM)"),
            "p-val (CAPM)":    safe_p("p-value (CAPM)"),
            "Sig":             sig,
            "FF4 Bars/Total":  r["FF4 bars/Total"],
        })

    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")),
              f"table{tnum}_multifreq_{freq_label}_results.csv")

    cols = "llccccccccccc"
    tex  = latex_header(
        f"FF4 vs CAPM Cumulative Abnormal Returns -- {freq_label} bars",
        f"tab:multifreq_{freq_label}",
        cols
    )
    header = (
        "    Stock & Event & Pre-Ban & FF4 @30m & FF4 @1h & FF4 @End & t(FF4) & p(FF4) "
        "& CAPM @30m & CAPM @End & t(CAPM) & p(CAPM) & Sig \\\\\n    \\hline\n"
    )
    tex += header
    prev_stock = None
    for row in rows:
        if prev_stock and row["Stock"] != prev_stock:
            tex += "    \\hline\n"
        prev_stock = row["Stock"]
        tex += (
            f"    {row['Stock']} & {row['Event']} & {row['Pre-Ban']} "
            f"& {row['FF4 @30m']} & {row['FF4 @1h']} & {row['FF4 @End']} "
            f"& {row['t-stat (FF4)']} & {row['p-val (FF4)']} "
            f"& {row['CAPM @30m']} & {row['CAPM @End']} "
            f"& {row['t-stat (CAPM)']} & {row['p-val (CAPM)']} & {row['Sig']} \\\\\n"
        )
    tex += latex_footer(
        "t-statistics use the per-bar residual sigma from the estimation window. "
        "Sig stars based on FF4 p-value."
    )
    write_latex(tex, f"table{tnum}_multifreq_{freq_label}_results.tex")


# ── Table 5 & 6: Multi-Freq Recovery ────────────────────────────────────────

def table_recovery_multifreq(freq_label: str, fname_csv: str, tnum: int):
    """
    Tables 5 & 6 — Post-Outage Recovery Cumulative Abnormal Returns at 5-min and
    10-min bar frequencies.

    CARs are anchored to zero at the moment the outage ends and accumulated forward
    over three recovery horizons (+30 min, +1 h, +2 h).  Patell t-statistics are
    computed separately at each horizon, allowing assessment of whether the recovery
    (or continued decline) is statistically distinguishable from zero at each point.
    'FF Bars/Total' indicates data availability during the recovery window.
    """
    src = os.path.join(BASE_DIR, "output", "multi_freq", fname_csv)
    df  = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
        def safe_pct(col):
            v = r.get(col, np.nan)
            if pd.isna(v) or str(v).strip() == "":
                return "—"
            return f"{float(v)*100:.2f}\\%"

        def safe_t(col):
            v = r.get(col, np.nan)
            if pd.isna(v) or str(v).strip() == "":
                return "—"
            return f"{float(v):.3f}"

        sig = str(r.get("Sig", "")).strip() if not pd.isna(r.get("Sig", np.nan)) else ""
        rows.append({
            "Stock":          r["Stock"],
            "Event":          r["Event"],
            "Recovery Start": r["Recovery Start"],
            "CAR @+30m":      safe_pct("CAR @+30min"),
            "t (@30m)":       safe_t("t (@30m)"),
            "CAR @+1h":       safe_pct("CAR @+1h"),
            "t (@1h)":        safe_t("t (@1h)"),
            "CAR @+2h":       safe_pct("CAR @+2h"),
            "t (@2h)":        safe_t("t (@2h)"),
            "Sig":            sig,
            "FF Bars/Total":  r.get("FF Bars / Total", ""),
        })

    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")),
              f"table{tnum}_recovery_{freq_label}.csv")

    cols = "llccccccccc"
    tex  = latex_header(
        f"Post-Outage Recovery Cumulative Abnormal Returns -- {freq_label} bars",
        f"tab:recovery_{freq_label}",
        cols
    )
    header = ("    Stock & Event & Rec. Start & CAR @+30m & t & CAR @+1h & t "
              "& CAR @+2h & t & Sig & FF Bars \\\\\n    \\hline\n")
    tex += header
    prev_stock = None
    for row in rows:
        if prev_stock and row["Stock"] != prev_stock:
            tex += "    \\hline\n"
        prev_stock = row["Stock"]
        tex += (
            f"    {row['Stock']} & {row['Event']} & {row['Recovery Start']} "
            f"& {row['CAR @+30m']} & {row['t (@30m)']} "
            f"& {row['CAR @+1h']} & {row['t (@1h)']} "
            f"& {row['CAR @+2h']} & {row['t (@2h)']} "
            f"& {row['Sig']} & {row['FF Bars/Total']} \\\\\n"
        )
    tex += latex_footer()
    write_latex(tex, f"table{tnum}_recovery_{freq_label}.tex")


# ── Table 7: FF4 Recovery (1-min) ───────────────────────────────────────────

def table_ff4_recovery():
    """
    Table 7 — Post-Outage Recovery Cumulative Abnormal Returns using the FF4 Model
    (1-min bars).

    Mirrors Tables 5–6 but uses the more granular 1-minute bar frequency and the
    full four-factor model.  CARs anchored at outage end with Patell t-statistics
    at each of three recovery horizons.  Also reports the raw price change over the
    2-hour recovery window for comparison with the model-adjusted CARs.
    Outage 4 has no recovery data because it falls entirely outside regular trading
    hours when FF factor data are unavailable.
    """
    src = os.path.join(BASE_DIR, "output", "ff4", "car_results_ff4_recovery.csv")
    df  = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
        def safe_pct(col):
            v = r.get(col, np.nan)
            if pd.isna(v) or str(v).strip() == "":
                return "—"
            return f"{float(v)*100:.2f}\\%"

        def safe_t(col):
            v = r.get(col, np.nan)
            if pd.isna(v) or str(v).strip() == "":
                return "—"
            return f"{float(v):.3f}"

        sig = str(r.get("Sig", "")).strip() if not pd.isna(r.get("Sig", np.nan)) else ""
        rows.append({
            "Stock":           r["Stock"],
            "Event":           r["Event"],
            "Recovery Start":  r["Recovery Start"],
            "CAR @+30m":       safe_pct("CAR @+30min"),
            "t (@30m)":        safe_t("t (@30m)"),
            "CAR @+1h":        safe_pct("CAR @+1h"),
            "t (@1h)":         safe_t("t (@1h)"),
            "CAR @+2h":        safe_pct("CAR @+2h"),
            "t (@2h)":         safe_t("t (@2h)"),
            "Sig":             sig,
            "Price Chg Rec %": f"{float(r['Price Change Recovery %']):.2f}\\%" if not pd.isna(r.get("Price Change Recovery %", np.nan)) else "—",
            "FF Mins/Total":   r.get("FF Mins / Total", ""),
        })

    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")),
              "table7_ff4_recovery.csv")

    cols = "llcccccccccc"
    tex  = latex_header(
        "Post-Outage Recovery Cumulative Abnormal Returns -- FF4 Model (1-min bars)",
        "tab:ff4_recovery",
        cols
    )
    header = ("    Stock & Event & Rec. Start & CAR @+30m & t & CAR @+1h & t "
              "& CAR @+2h & t & Sig & Price Chg Rec & FF Mins \\\\\n    \\hline\n")
    tex += header
    prev_stock = None
    for row in rows:
        if prev_stock and row["Stock"] != prev_stock:
            tex += "    \\hline\n"
        prev_stock = row["Stock"]
        tex += (
            f"    {row['Stock']} & {row['Event']} & {row['Recovery Start']} "
            f"& {row['CAR @+30m']} & {row['t (@30m)']} "
            f"& {row['CAR @+1h']} & {row['t (@1h)']} "
            f"& {row['CAR @+2h']} & {row['t (@2h)']} "
            f"& {row['Sig']} & {row['Price Chg Rec %']} & {row['FF Mins/Total']} \\\\\n"
        )
    tex += latex_footer()
    write_latex(tex, "table7_ff4_recovery.tex")


# ── Table 8: CAPM summary: pre-ban vs post-ban ────────────────────────────────

def table_capm_summary():
    """
    Table 8 — CAPM Summary: Average CARs Pre-Ban vs Post-Ban.

    Aggregates the four individual event results into a 2×2 matrix (two stocks ×
    two periods).  The Robinhood trading ban (Jan 28, 09:30 ET) divides events into
    pre-ban (Outages 1–2, Jan 27) and post-ban (Outages 3–4, Jan 28).  Average
    t-statistics are reported alongside average CARs as a summary measure of
    statistical significance within each group.
    """
    src = os.path.join(BASE_DIR, "output", "spy-only", "car_results_capm.csv")
    df  = pd.read_csv(src)

    rows = []
    for stock in ["GME", "AMC"]:
        sub = df[df["Stock"] == stock]
        for period in ["Pre-Ban", "Post-Ban"]:
            is_pre = (period == "Pre-Ban")
            mask   = sub["Pre-Ban"] == is_pre
            grp    = sub[mask]
            if grp.empty:
                continue
            avg_car_end  = grp["CAR @End"].mean()
            avg_car_1h   = grp["CAR @1h"].mean()
            avg_price    = grp["Price Change %"].mean()
            # Average t-stat over events in the group (summary measure)
            avg_tstat    = grp["t-stat"].mean() if "t-stat" in grp.columns else np.nan
            rows.append({
                "Stock":        stock,
                "Period":       period,
                "Events":       len(grp),
                "Avg CAR @1h":  f"{avg_car_1h*100:.2f}\\%",
                "Avg CAR @End": f"{avg_car_end*100:.2f}\\%",
                "Avg Price Chg": f"{avg_price:.2f}\\%",
                "Avg t-stat":   f"{avg_tstat:.3f}" if not pd.isna(avg_tstat) else "—",
            })

    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")),
              "table8_capm_summary_prepost.csv")

    cols = "llcccccc"
    tex  = latex_header(
        "CAPM Summary: Average CARs Pre-Ban vs Post-Ban",
        "tab:capm_summary",
        cols
    )
    header = ("    Stock & Period & Events & Avg CAR @1h & Avg CAR @End "
              "& Avg Price Chg & Avg t-stat \\\\\n    \\hline\n")
    tex += header
    prev_stock = None
    for row in rows:
        if prev_stock and row["Stock"] != prev_stock:
            tex += "    \\hline\n"
        prev_stock = row["Stock"]
        tex += (
            f"    {row['Stock']} & {row['Period']} & {row['Events']} "
            f"& {row['Avg CAR @1h']} & {row['Avg CAR @End']} "
            f"& {row['Avg Price Chg']} & {row['Avg t-stat']} \\\\\n"
        )
    tex += latex_footer(
        "The Robinhood trading-ban cut-off is January 28, 2021 at 09:30 ET. "
        "Pre-Ban = Outages 1 \\& 2 (January 27); Post-Ban = Outages 3 \\& 4 (January 28). "
        "Avg t-stat is the simple mean of individual event Patell t-statistics within each group."
    )
    write_latex(tex, "table8_capm_summary_prepost.tex")


# ── Table descriptions (stand-alone CSV) ─────────────────────────────────────

TABLE_DESCRIPTIONS = [
    {
        "Table": "Table 1",
        "File":  "table1_capm_results.csv / .tex",
        "Model": "CAPM (single-factor, 1-min bars)",
        "Description": (
            "Reports Cumulative Abnormal Returns (CARs) for GME and AMC during each of the "
            "four Reddit server outage windows on January 27–28, 2021, estimated using the "
            "Capital Asset Pricing Model (CAPM) with SPY as the market proxy. "
            "The estimation window spans 120 trading days ending 2 days before the first outage. "
            "CARs are measured at four horizons within the event window: 30 minutes, 1 hour, "
            "the outage end, and 2 hours after the outage ends. "
            "The Patell t-statistic (t = CAR / sigma*sqrt(N)) tests whether the CAR at outage "
            "end differs from zero; two-tailed p-values and significance stars are reported. "
            "Pre-Ban events occurred before the Robinhood trading restriction (Jan 28, 09:30 ET); "
            "Post-Ban events occurred after."
        ),
    },
    {
        "Table": "Table 2",
        "File":  "table2_ff4_results.csv / .tex",
        "Model": "Fama-French 4-Factor (1-min bars)",
        "Description": (
            "Same event study as Table 1, but expected returns are modelled using the "
            "Fama-French four-factor model (market, SMB, HML, MOM). "
            "FF factor data are only available during regular trading hours (09:30–16:00 ET); "
            "after-hours minutes fall back to a market-model using SPY as the sole factor. "
            "The column 'FF Mins/Total' shows how many outage minutes had genuine four-factor "
            "coverage versus total minutes in the event window. "
            "t-statistics and p-values are computed using the FF4 residual standard deviation "
            "from the estimation window."
        ),
    },
    {
        "Table": "Table 3",
        "File":  "table3_multifreq_5min_results.csv / .tex",
        "Model": "FF4 vs CAPM (5-min bars)",
        "Description": (
            "Compares FF4 and CAPM event-window CARs aggregated to 5-minute bar frequency. "
            "Coarser bars reduce microstructure noise but also reduce the number of observations "
            "per event window (N), making t-statistics less precise for short events. "
            "Both models' CARs at 30 min, 1 h, and outage end are reported alongside individual "
            "t-statistics and p-values. 'FF4 Bars/Total' indicates data availability. "
            "Significance stars are based on the FF4 p-value."
        ),
    },
    {
        "Table": "Table 4",
        "File":  "table4_multifreq_10min_results.csv / .tex",
        "Model": "FF4 vs CAPM (10-min bars)",
        "Description": (
            "Same as Table 3 but at 10-minute bar frequency. "
            "The coarser resolution further reduces N per event window; "
            "Outage 2 (58 min) produces only ~6 bars, limiting statistical power. "
            "Results should be interpreted alongside the 1-min and 5-min findings "
            "to assess robustness across bar frequencies."
        ),
    },
    {
        "Table": "Table 5",
        "File":  "table5_recovery_5min.csv / .tex",
        "Model": "FF4 / CAPM Recovery (5-min bars)",
        "Description": (
            "Post-outage recovery CARs computed from the moment each outage ends, "
            "accumulated forward over three horizons (+30 min, +1 h, +2 h) at 5-minute "
            "bar frequency. CARs are anchored to zero at outage end so that positive values "
            "indicate a rebound and negative values indicate a continued decline. "
            "Patell t-statistics are computed at each horizon separately. "
            "A hybrid model is used: FF4 where factor data are available, CAPM otherwise."
        ),
    },
    {
        "Table": "Table 6",
        "File":  "table6_recovery_10min.csv / .tex",
        "Model": "FF4 / CAPM Recovery (10-min bars)",
        "Description": (
            "Same as Table 5 but at 10-minute bar frequency. "
            "Outage 4 recovery data are unavailable at coarse frequencies because the outage "
            "ends after market hours with no subsequent trading bars."
        ),
    },
    {
        "Table": "Table 7",
        "File":  "table7_ff4_recovery.csv / .tex",
        "Model": "FF4 Recovery (1-min bars)",
        "Description": (
            "Post-outage recovery CARs using the full four-factor model at 1-minute bar "
            "frequency. Provides the most granular view of price dynamics immediately after "
            "each Reddit outage ends. Also reports the raw price change over the 2-hour "
            "recovery window for comparison with the model-adjusted abnormal returns. "
            "Outage 4 has no entry because the recovery window falls entirely outside "
            "regular trading hours."
        ),
    },
    {
        "Table": "Table 8",
        "File":  "table8_capm_summary_prepost.csv / .tex",
        "Model": "CAPM Summary",
        "Description": (
            "Aggregates the CAPM event-study results into a pre-ban vs post-ban comparison. "
            "The Robinhood trading restriction (January 28, 2021 at 09:30 ET) divides the "
            "four outage events into two pre-ban events (Outages 1–2, January 27) and two "
            "post-ban events (Outages 3–4, January 28). "
            "Reports the simple average of CARs at the 1-hour horizon and at outage end, "
            "together with the average raw price change and the average Patell t-statistic "
            "across events in each group."
        ),
    },
]


def table_descriptions():
    """Write a stand-alone CSV summarising all tables."""
    df = pd.DataFrame(TABLE_DESCRIPTIONS)
    write_csv(df, "table_descriptions.csv")

    # Also write a LaTeX table of descriptions
    tex = (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        "  \\caption{Summary of All Output Tables}\n"
        "  \\label{tab:descriptions}\n"
        "  \\begin{tabular}{llp{9cm}}\n"
        "    \\hline\\hline\n"
        "    Table & Model & Description \\\\\n"
        "    \\hline\n"
    )
    for d in TABLE_DESCRIPTIONS:
        desc_escaped = d["Description"].replace("%", "\\%").replace("&", "\\&")
        tex += f"    {d['Table']} & {d['Model']} & {desc_escaped} \\\\\n"
        tex += "    \\hline\n"
    tex += (
        "    \\hline\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    write_latex(tex, "table_descriptions.tex")


# ── run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating tables into:", OUT_DIR)
    table_capm()
    table_ff4()
    table_multifreq("5min",  "car_results_5min.csv",  tnum=3)
    table_multifreq("10min", "car_results_10min.csv", tnum=4)
    table_recovery_multifreq("5min",  "car_results_5min_recovery.csv",  tnum=5)
    table_recovery_multifreq("10min", "car_results_10min_recovery.csv", tnum=6)
    table_ff4_recovery()
    table_capm_summary()
    table_descriptions()
    print("Done.")
