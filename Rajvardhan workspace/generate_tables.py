"""
Generate formatted LaTeX and CSV table files from analysis result CSVs.
Output goes into: output-tables/
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

def latex_footer() -> str:
    return (
        "    \\hline\\hline\n"
        "  \\end{tabular}\n"
        "  \\\\[4pt]\n"
        "  {\\small *** $p<0.01$, ** $p<0.05$, * $p<0.10$. CAR = cumulative abnormal return "
        "relative to the estimation window (120 trading days ending 2 days before each event). "
        "t-statistics in parentheses.}\n"
        "\\end{table}\n"
    )

# ── Table 1: CAPM CAR Results ────────────────────────────────────────────────

def table_capm():
    src = os.path.join(BASE_DIR, "output", "spy-only", "car_results_capm.csv")
    df  = pd.read_csv(src)

    # Build a formatted display dataframe
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
        "CAPM Cumulative Abnormal Returns During Reddit Outage Events",
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
    src = os.path.join(BASE_DIR, "output", "ff4", "car_results_ff4.csv")
    df  = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Stock":        r["Stock"],
            "Event":        r["Event"],
            "Pre-Ban":      "Yes" if r["Pre-Ban"] else "No",
            "CAR @30m":     f"{float(r['CAR @30min'])*100:.2f}\\%",
            "CAR @1h":      f"{float(r['CAR @1h'])*100:.2f}\\%",
            "CAR @End":     f"{float(r['CAR @End'])*100:.2f}\\%",
            "CAR @End+2h":  f"{float(r['CAR @End+2h'])*100:.2f}\\%",
            "Price Chg %":  f"{float(r['Price Change %']):.2f}\\%",
            "FF Mins/Total": r["FF Mins / Total"],
        })
    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")), "table2_ff4_results.csv")

    cols = "llccccccc"
    tex  = latex_header(
        "Fama-French 4-Factor Cumulative Abnormal Returns During Reddit Outage Events (1-min bars)",
        "tab:ff4_car",
        cols
    )
    header = ("    Stock & Event & Pre-Ban & CAR @30m & CAR @1h & CAR @End "
              "& CAR @End+2h & Price Chg & FF Mins/Total \\\\\n    \\hline\n")
    tex += header
    prev_stock = None
    for row in rows:
        if prev_stock and row["Stock"] != prev_stock:
            tex += "    \\hline\n"
        prev_stock = row["Stock"]
        tex += (
            f"    {row['Stock']} & {row['Event']} & {row['Pre-Ban']} "
            f"& {row['CAR @30m']} & {row['CAR @1h']} & {row['CAR @End']} "
            f"& {row['CAR @End+2h']} & {row['Price Chg %']} & {row['FF Mins/Total']} \\\\\n"
        )
    tex += latex_footer()
    write_latex(tex, "table2_ff4_results.tex")


# ── Table 3 & 4: Multi-Frequency FF4 vs CAPM ────────────────────────────────

def table_multifreq(freq_label: str, fname_csv: str, tnum: int):
    src = os.path.join(BASE_DIR, "output", "multi_freq", fname_csv)
    df  = pd.read_csv(src)

    rows = []
    for _, r in df.iterrows():
        def safe_pct(col):
            v = r.get(col, np.nan)
            if pd.isna(v) or str(v).strip() == "":
                return "—"
            return f"{float(v)*100:.2f}\\%"

        rows.append({
            "Stock":            r["Stock"],
            "Event":            r["Event"],
            "Pre-Ban":          "Yes" if r["Pre-Ban"] else "No",
            "FF4 @30m":         safe_pct("CAR @30min (FF4)"),
            "FF4 @1h":          safe_pct("CAR @1h (FF4)"),
            "FF4 @End":         safe_pct("CAR @End (FF4)"),
            "CAPM @30m":        safe_pct("CAR @30min (CAPM)"),
            "CAPM @End":        safe_pct("CAR @End (CAPM)"),
            "Price Chg %":      f"{float(r['Price Chg %']):.2f}\\%",
            "FF4 Bars/Total":   r["FF4 bars/Total"],
        })

    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")),
              f"table{tnum}_multifreq_{freq_label}_results.csv")

    cols = "llccccccc"
    tex  = latex_header(
        f"FF4 vs CAPM Cumulative Abnormal Returns – {freq_label} bars",
        f"tab:multifreq_{freq_label}",
        cols
    )
    header = ("    Stock & Event & Pre-Ban & FF4 @30m & FF4 @1h & FF4 @End "
              "& CAPM @30m & CAPM @End & FF4 Bars/Total \\\\\n    \\hline\n")
    tex += header
    prev_stock = None
    for row in rows:
        if prev_stock and row["Stock"] != prev_stock:
            tex += "    \\hline\n"
        prev_stock = row["Stock"]
        tex += (
            f"    {row['Stock']} & {row['Event']} & {row['Pre-Ban']} "
            f"& {row['FF4 @30m']} & {row['FF4 @1h']} & {row['FF4 @End']} "
            f"& {row['CAPM @30m']} & {row['CAPM @End']} & {row['FF4 Bars/Total']} \\\\\n"
        )
    tex += latex_footer()
    write_latex(tex, f"table{tnum}_multifreq_{freq_label}_results.tex")


# ── Table 5 & 6: Multi-Freq Recovery ────────────────────────────────────────

def table_recovery_multifreq(freq_label: str, fname_csv: str, tnum: int):
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
        f"Post-Outage Recovery Cumulative Abnormal Returns – {freq_label} bars",
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
            "Price Chg Rec %": f"{float(r['Price Change Recovery %']):.2f}\\%" if not pd.isna(r.get("Price Change Recovery %", np.nan)) else "—",
            "FF Mins/Total":  r.get("FF Mins / Total", ""),
        })

    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")),
              "table7_ff4_recovery.csv")

    cols = "llcccccccccc"
    tex  = latex_header(
        "Post-Outage Recovery Cumulative Abnormal Returns – FF4 Model (1-min bars)",
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


# ── CAPM summary: pre-ban vs post-ban ────────────────────────────────────────

def table_capm_summary():
    """Aggregate CAPM results into a pre-ban vs post-ban summary table."""
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
            rows.append({
                "Stock":       stock,
                "Period":      period,
                "Events":      len(grp),
                "Avg CAR @1h": f"{avg_car_1h*100:.2f}\\%",
                "Avg CAR @End": f"{avg_car_end*100:.2f}\\%",
                "Avg Price Chg": f"{avg_price:.2f}\\%",
            })

    out = pd.DataFrame(rows)
    write_csv(out.rename(columns=lambda c: c.replace("\\%", "%")),
              "table8_capm_summary_prepost.csv")

    cols = "llccccc"
    tex  = latex_header(
        "CAPM Summary: Average CARs Pre-Ban vs Post-Ban",
        "tab:capm_summary",
        cols
    )
    header = "    Stock & Period & Events & Avg CAR @1h & Avg CAR @End & Avg Price Chg \\\\\n    \\hline\n"
    tex += header
    prev_stock = None
    for row in rows:
        if prev_stock and row["Stock"] != prev_stock:
            tex += "    \\hline\n"
        prev_stock = row["Stock"]
        tex += (
            f"    {row['Stock']} & {row['Period']} & {row['Events']} "
            f"& {row['Avg CAR @1h']} & {row['Avg CAR @End']} & {row['Avg Price Chg']} \\\\\n"
        )
    tex += latex_footer()
    write_latex(tex, "table8_capm_summary_prepost.tex")


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
    print("Done.")
