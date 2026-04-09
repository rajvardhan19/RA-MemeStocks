# Reddit Outage Impact on Meme Stock Returns

An empirical event study measuring the effect of Reddit backend server outages on GameStop (GME) and AMC Entertainment (AMC) abnormal returns during the January 2021 retail-trading frenzy.

---

## Research Question

Do Reddit server outages causally disrupt the retail investor coordination that drives meme-stock prices? Specifically, do GME and AMC exhibit statistically significant Cumulative Abnormal Returns (CARs) during the four documented Reddit outage windows on January 27–28, 2021, and do these effects differ before versus after the Robinhood trading restriction imposed on January 28 at 09:30 ET?

---

## Key Findings

| Stock | Event | CAR @ Outage End | t-stat | p-value |
|-------|-------|-----------------|--------|---------|
| GME | Outage 3 (Post-Ban) | −33.3% | −3.61 | 0.0003 |
| AMC | Outage 3 (Post-Ban) | −62.9% | −11.84 | <0.001 |
| AMC | Outage 2 (Pre-Ban) | −14.9% | −3.66 | 0.0003 |
| GME | Outage 2 (Pre-Ban) | −15.3% | −2.21 | 0.027 |

- Post-ban outage events produced substantially larger negative abnormal returns for both stocks.
- Recovery CARs after most outages remained negative, suggesting the price impact was not immediately reversed.
- Results are robust across CAPM and Fama-French 4-factor models and across 1-min, 5-min, and 10-min bar frequencies.

---

## Repository Structure

```
RA-MemeStocks/
│
├── README.md                                    ← this file
│
├── Rajvardhan workspace/                        ← main analysis directory
│   ├── config.py                                ← paths, event definitions, plot settings
│   ├── event_study.py                           ← CAPM event study + recovery analysis
│   ├── compute_tstats.py                        ← adds t-stats/p-values to FF4 & multi-freq CSVs
│   ├── generate_tables.py                       ← builds all LaTeX/CSV output tables
│   ├── visualizations.py                        ← publication-quality figures
│   │
│   ├── meme_stock_outage_analysis.ipynb         ← main CAPM analysis notebook
│   ├── meme_stock_outage_ff4.ipynb              ← Fama-French 4-factor analysis
│   ├── meme_stock_outage_multi_freq.ipynb       ← 5-min & 10-min frequency analysis
│   ├── excess_return_multi_model.ipynb          ← multi-model comparison
│   │
│   ├── output/
│   │   ├── spy-only/                            ← CAPM results (CSV + figures)
│   │   ├── ff4/                                 ← FF4 results (CSV + figures)
│   │   └── multi_freq/                          ← 5-min & 10-min results (CSV + figures)
│   │
│   └── output-tables/                           ← final formatted tables (CSV + LaTeX)
│       ├── table1_capm_results.*
│       ├── table2_ff4_results.*
│       ├── table3_multifreq_5min_results.*
│       ├── table4_multifreq_10min_results.*
│       ├── table5_recovery_5min.*
│       ├── table6_recovery_10min.*
│       ├── table7_ff4_recovery.*
│       ├── table8_capm_summary_prepost.*
│       └── table_descriptions.*
│
├── Minute price and exceed return/
│   └── output_data/                             ← cleaned minute-level return CSVs
│       ├── GME-minute_price-excess-return.csv
│       ├── AMC-minute_price-excess-return.csv
│       ├── SPY_excess_returns_202011-202104.csv
│       ├── GME-{1,5,10}min-multi-model-excess-return.csv
│       └── AMC-{1,5,10}min-multi-model-excess-return.csv
│
├── FF-Factors/                                  ← Fama-French factor data
│   ├── ff_factors_20201101_20210430_minute.csv  ← 1-min FF factors (trading hours only)
│   ├── ff_factors_daily_202011_202104.csv
│   └── ff_factors_monthly_202011_202104.csv
│
├── ff_factor_minute_construction/               ← notebooks for building minute FF factors
│   └── minute_prices_etf/                       ← ETF price data (SPY, VUG, VTV, IWM, etc.)
│
├── outage_window_CAR/                           ← legacy alternative CAR implementation
├── SPY fund/                                    ← SPY constituent/price data
└── relevant_literature/                         ← reference papers
```

---

## Event Study Design

### Outage Events

| # | Name | Date | Start | End | Duration | Period |
|---|------|------|-------|-----|----------|--------|
| 1 | Outage 1 (Pre-Ban) | Jan 27, 2021 | 11:29 ET | 13:40 ET | 131 min | Pre-Ban |
| 2 | Outage 2 (Pre-Ban) | Jan 27, 2021 | 16:03 ET | 17:01 ET | 58 min | Pre-Ban |
| 3 | Outage 3 (Post-Ban) | Jan 28, 2021 | 08:44 ET | 10:51 ET | 127 min | Post-Ban |
| 4 | Outage 4 (Post-Ban) | Jan 28, 2021 | 19:10 ET | 21:00 ET | 110 min | Post-Ban |

The Robinhood trading ban cut-off is **January 28, 2021 at 09:30 ET**.

### Estimation Window

- **Length**: 120 trading days
- **Lag**: 2 days before the first event (clean separation from event contamination)
- **Effective window**: approximately September 2020 – January 25, 2021

### CAR Measurement Horizons

CARs are accumulated from event start and reported at:
- **+30 min** — early outage effect
- **+1 h** — mid-outage effect
- **Outage end** — full outage effect (used for primary t-tests)
- **End +2 h** — post-outage persistence

Recovery CARs are separately anchored at outage end (+30 min, +1 h, +2 h).

### Statistical Testing

**Patell t-statistic** (two-tailed):

```
t = CAR_at_end / (σ̂ × √N)
```

where `σ̂` is the residual standard deviation from the OLS estimation window and `N` is the number of return observations in the event window. p-values are computed under the standard normal distribution. Significance thresholds: *** p<0.01, ** p<0.05, * p<0.10.

---

## Models

### CAPM (Single-Factor)

```
R_i,t = α + β × R_SPY,t + ε_i,t
```

Estimated over the 120-day window. SPY is used as the market proxy. All minutes (including after-hours) are included.

### Fama-French 4-Factor (FF4)

```
R_i,t = α + β_MKT × MKT_RF_t + β_SMB × SMB_t + β_HML × HML_t + β_MOM × MOM_t + ε_i,t
```

FF factor data are available only during regular trading hours (09:30–16:00 ET). After-hours minutes fall back to a market-model using SPY as `MKT_RF` with SMB = HML = MOM = 0.

### Multi-Frequency (5-min and 10-min bars)

Both models are also estimated on 5-minute and 10-minute aggregated bars to assess robustness to bar-frequency choice and reduce microstructure noise.

---

## Output Tables

Each table is saved as both a `.csv` (for analysis) and a `.tex` (for inclusion in a LaTeX paper). All tables are in `Rajvardhan workspace/output-tables/`.

### Table 1 — CAPM CAR Results (`table1_capm_results`)

**Model**: CAPM, 1-min bars

Reports CARs for GME and AMC during each of the four Reddit outage windows. Expected returns use the single-factor CAPM (SPY). Columns: Event duration, Pre-Ban flag, CAR at 30 min / 1 h / outage end / end+2 h, raw price change during outage, Patell t-statistic at outage end, two-tailed p-value, significance stars.

### Table 2 — FF4 CAR Results (`table2_ff4_results`)

**Model**: Fama-French 4-factor, 1-min bars

Same structure as Table 1 but with expected returns from the FF4 model. An additional column `FF Mins/Total` shows how many outage minutes had genuine four-factor coverage (trading hours) versus total event minutes. t-statistics use the FF4 residual standard deviation.

### Table 3 — FF4 vs CAPM, 5-min Bars (`table3_multifreq_5min_results`)

**Model**: FF4 and CAPM side-by-side, 5-min bars

Compares FF4 and CAPM CARs at 30 min, 1 h, and outage end using 5-minute aggregated bars. Each model's Patell t-statistic and p-value are reported separately. Significance stars are based on the FF4 p-value. Outage 4 has no CAPM @End or FF4 @End because the event falls outside regular trading hours with no subsequent bars.

### Table 4 — FF4 vs CAPM, 10-min Bars (`table4_multifreq_10min_results`)

**Model**: FF4 and CAPM side-by-side, 10-min bars

Same as Table 3 at 10-minute bar frequency. Outage 2 (58 min) produces only ~6 bars at this frequency, limiting statistical power. Intended as a robustness check against Table 3.

### Table 5 — Post-Outage Recovery CARs, 5-min Bars (`table5_recovery_5min`)

**Model**: Hybrid FF4/CAPM, 5-min bars

CARs anchored at outage end (+30 min, +1 h, +2 h) to measure post-outage price dynamics. A positive CAR indicates a price rebound; a negative CAR indicates a continued decline. Separate Patell t-statistics at each recovery horizon. Outage 4 has no recovery data (after-hours, no subsequent bars).

### Table 6 — Post-Outage Recovery CARs, 10-min Bars (`table6_recovery_10min`)

**Model**: Hybrid FF4/CAPM, 10-min bars

Same as Table 5 at 10-minute bar frequency. Serves as a robustness check for the recovery dynamics.

### Table 7 — FF4 Recovery CARs, 1-min Bars (`table7_ff4_recovery`)

**Model**: FF4, 1-min bars

Most granular recovery analysis using 1-minute bars and the full four-factor model. Also reports the raw price change over the 2-hour recovery window for comparison with model-adjusted CARs. Outage 4 is excluded because its recovery window falls entirely outside trading hours.

### Table 8 — CAPM Summary: Pre-Ban vs Post-Ban (`table8_capm_summary_prepost`)

**Model**: CAPM summary

Aggregates CAPM event results into a 2×2 matrix (GME/AMC × Pre-Ban/Post-Ban). Reports average CARs at 1 h and outage end, average raw price change, and the average Patell t-statistic across events within each group. Highlights whether the Robinhood trading ban amplified the impact of Reddit outages.

### Table Descriptions (`table_descriptions`)

A catalogue of all eight tables with their model, file name, and plain-language description. Useful as a reference appendix in a paper.

---

## Data Sources

| Dataset | Description | Frequency | Coverage |
|---------|-------------|-----------|----------|
| GME / AMC prices & returns | Cleaned minute-level prices and log returns | 1-min | Nov 2020 – Apr 2021 |
| SPY prices & returns | S&P 500 ETF (market proxy) | 1-min | Nov 2020 – Apr 2021 |
| Fama-French 4 factors | MKT_RF, SMB, HML, MOM constructed from ETF returns | 1-min | Nov 2020 – Apr 2021 |
| Multi-model excess returns | Pre-computed CAPM, FF3, FF4 model residuals | 1/5/10-min | Nov 2020 – Apr 2021 |

Minute-level FF factors are constructed from ETF price data (SPY, VUG, VTV, IWM, VV, MTUM) as documented in `ff_factor_minute_construction/`.

---

## How to Reproduce

### Prerequisites

```bash
pip install pandas numpy scipy statsmodels matplotlib
```

### Step 1 — Run the main analysis notebooks

Execute the four Jupyter notebooks in order (or independently):

```bash
cd "Rajvardhan workspace"
jupyter notebook meme_stock_outage_analysis.ipynb       # CAPM (1-min)
jupyter notebook meme_stock_outage_ff4.ipynb            # FF4 (1-min)
jupyter notebook meme_stock_outage_multi_freq.ipynb     # FF4 + CAPM (5-min, 10-min)
jupyter notebook excess_return_multi_model.ipynb        # multi-model comparison
```

Each notebook writes result CSVs and figures to `output/spy-only/`, `output/ff4/`, or `output/multi_freq/`.

### Step 2 — Add t-statistics to FF4 and multi-frequency CSVs

```bash
python compute_tstats.py
```

This script reads the data files, re-estimates the FF4 model and multi-frequency sigma values, and appends `t-stat`, `p-value`, and `Sig` columns to the relevant result CSVs.

### Step 3 — Generate formatted tables

```bash
python generate_tables.py
```

Writes all eight tables (plus the description catalogue) to `output-tables/` as both CSV and LaTeX files.

### Step 4 — Regenerate figures (optional)

```bash
python visualizations.py
```

---

## Core Modules

| File | Purpose |
|------|---------|
| `config.py` | Central configuration: file paths, event timestamps, estimation window parameters, plot style |
| `event_study.py` | CAPM event study: data loading, model estimation, CAR calculation, Patell t-statistics, recovery analysis, Mann-Whitney test |
| `compute_tstats.py` | Adds missing t-statistics and p-values to FF4 and multi-frequency result CSVs |
| `generate_tables.py` | Reads result CSVs and writes publication-ready tables in CSV and LaTeX format |
| `visualizations.py` | CAR comparison charts, per-event price/CAR plots, volatility and excess return time series |

---

## Methodology Notes

- **CAR normalisation**: CARs are set to zero at the event start so that figures can reveal pre-trends while keeping all post-event metrics unaffected by pre-period drift.
- **After-hours fallback**: For FF4 and hybrid models, minutes outside 09:30–16:00 ET use SPY as the sole factor (MKT_RF = SPY return; SMB = HML = MOM = 0).
- **Recovery anchoring**: Recovery CARs are independently anchored at outage end; they are not a continuation of the event-window CAR series.
- **Mann-Whitney U test**: Non-parametric comparison of pre-ban vs post-ban CAR distributions (reported in event_study.py console output).

---

## Author

Rajvardhan Patil
