# =============================================================================
# PHASE 2 — FORECAST, SCORE & MAP
# No-Fault Claims Intelligence System
# =============================================================================
# What this does:
#   1. Builds weekly time series from Date of Loss
#   2. Runs Moving Average (4-week) and ARIMA models side by side
#   3. Forecasts total dollar reserve (30/60/90 days)
#   4. Scores every claim for fraud risk
#   5. Flags volume spikes and city billing anomalies
#   6. Saves 6 charts + scored CSV
#
# Charts produced:
#   chart_1_weekly_volume.png     — weekly claims + 4-week moving average
#   chart_2_ma_vs_arima.png       — MA vs ARIMA 8-week forecast
#   chart_3_seasonality.png       — claims by month
#   chart_4_fraud_vs_volume.png   — fraud-flagged claims over time
#   chart_5_top_cities.png        — top 10 cities by total amount billed
#   chart_6_risk_distribution.png — fraud risk level breakdown
# =============================================================================

import pandas as pd
import numpy as np
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import warnings
warnings.filterwarnings("ignore")

DB_FILE = "outputs/claims.db"
OUT_DIR = "outputs"

# ── Style ─────────────────────────────────────────────────────────────────────
BLUE   = "#2E75B6"
ORANGE = "#ED7D31"
RED    = "#C00000"
GREEN  = "#70AD47"
GRAY   = "#AAAAAA"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

RISK_COLORS = {"LOW": GREEN, "MEDIUM": "#FFD966", "HIGH": ORANGE, "CRITICAL": RED}

# ── Helpers ───────────────────────────────────────────────────────────────────
def load():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT * FROM claims", conn)
    conn.close()
    df["Date of Loss"] = pd.to_datetime(df["Date of Loss"], errors="coerce")
    return df

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")

def simple_arima(series, steps=8):
    """
    Forecast using pmdarima if available, falls back to
    exponential smoothing, then plain linear extrapolation.
    """
    try:
        from pmdarima import auto_arima
        model = auto_arima(series, seasonal=False, suppress_warnings=True, error_action="ignore")
        fc, ci = model.predict(n_periods=steps, return_conf_int=True)
        return fc, ci[:, 0], ci[:, 1]
    except ImportError:
        pass
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(series, trend="add").fit()
        fc = model.forecast(steps)
        std = series.std()
        return fc.values, fc.values - 1.5 * std, fc.values + 1.5 * std
    except ImportError:
        pass
    x = np.arange(len(series))
    m, b = np.polyfit(x, series.values, 1)
    fc = np.array([m * (len(series) + i) + b for i in range(steps)])
    std = series.std()
    return fc, fc - 1.5 * std, fc + 1.5 * std

# ── Fraud Risk Scoring ────────────────────────────────────────────────────────
def score_claim(row):
    score = 0
    flags = []

    try:
        if float(row.get("Vehicles in Accident", 1)) >= 4:
            score += 30
            flags.append(f"High claimants ({int(row['Vehicles in Accident'])})")
    except (ValueError, TypeError):
        pass

    if str(row.get("Fraud Flag", "N")).strip().upper() == "Y":
        score += 25
        flags.append("Fraud flagged")

    try:
        billed = float(row.get("Total Amount Billed ($)", 0))
        if billed > 50000:
            score += 25
            flags.append(f"High bill (${billed:,.0f})")
        elif billed > 25000:
            score += 15
            flags.append(f"Elevated bill (${billed:,.0f})")
    except (ValueError, TypeError):
        pass

    if str(row.get("Police Report on File", "YES")).strip().upper() == "NO":
        score += 10
        flags.append("No police report")

    try:
        if int(row.get("Witness Count", 1)) == 0:
            score += 10
            flags.append("No witnesses")
    except (ValueError, TypeError):
        pass

    if str(row.get("Incident Severity", "")).strip().lower() in ["total loss", "major damage"]:
        score += 10
        flags.append("Severe damage")

    return score, "; ".join(flags) if flags else "None"

def risk_label(score):
    if score >= 75: return "CRITICAL"
    if score >= 46: return "HIGH"
    if score >= 21: return "MEDIUM"
    return "LOW"

# ── CHART 1: Weekly claim volume + moving average ─────────────────────────────
def chart_weekly_volume(df):
    weekly = df.groupby("Week").size().reset_index(name="Claims")
    weekly = weekly.sort_values("Week").reset_index(drop=True)
    weekly["MA4"] = weekly["Claims"].rolling(4, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(weekly)), weekly["Claims"], color=BLUE, alpha=0.45, label="Weekly claims")
    ax.plot(range(len(weekly)), weekly["MA4"], color=ORANGE, lw=2.2, label="4-week moving average")

    step = max(1, len(weekly) // 8)
    ax.set_xticks(range(0, len(weekly), step))
    ax.set_xticklabels(weekly["Week"].iloc[::step], rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Number of claims")
    ax.set_title("Weekly claim volume with 4-week moving average", fontweight="bold", pad=10)
    ax.legend()
    fig.tight_layout()
    save(fig, "chart_1_weekly_volume.png")
    return weekly

# ── CHART 2: MA vs ARIMA forecast ─────────────────────────────────────────────
def chart_arima_forecast(weekly):
    series = weekly["Claims"].copy()
    fc, lo, hi = simple_arima(series, steps=8)
    future_idx = list(range(len(series), len(series) + 8))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(series)), series, color=BLUE, alpha=0.35, label="Actual claims")
    ax.plot(range(len(series)), weekly["MA4"], color=ORANGE, lw=1.8, label="Moving average")
    ax.plot(future_idx, fc, color=RED, lw=2.2, linestyle="--", label="ARIMA forecast")
    ax.fill_between(future_idx, lo, hi, color=RED, alpha=0.12, label="Confidence band")
    ax.axvline(len(series) - 0.5, color=GRAY, lw=1, linestyle=":")

    step = max(1, len(weekly) // 8)
    all_len = len(series) + 8
    ax.set_xticks(range(0, all_len, max(step, 1)))
    tick_labels = list(weekly["Week"].iloc[::step]) + [""] * 8
    ax.set_xticklabels(tick_labels[:len(range(0, all_len, max(step, 1)))],
                       rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Number of claims")
    ax.set_title("Moving average vs ARIMA forecast — next 8 weeks", fontweight="bold", pad=10)
    ax.legend()
    fig.tight_layout()
    save(fig, "chart_2_ma_vs_arima.png").

    print(f"\n  ARIMA 8-week forecast:")
    for i, (f, l, h) in enumerate(zip(fc, lo, hi), 1):
        print(f"    Week +{i}: {f:.0f} claims  (range: {max(0,l):.0f} – {h:.0f})")

# ── CHART 3: Seasonality — claims by month ────────────────────────────────────
def chart_seasonality(df):
    df2 = df.copy()
    df2["Month Name"] = df2["Date of Loss"].dt.strftime("%b")
    df2["Month Num"]  = df2["Date of Loss"].dt.month
    monthly = (df2.groupby(["Month Num", "Month Name"])
                  .size().reset_index(name="Claims")
                  .sort_values("Month Num"))

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(monthly["Month Name"], monthly["Claims"],
                  color=BLUE, alpha=0.75, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, str(int(bar.get_height())),
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total claims")
    ax.set_title("Seasonal claim distribution by month", fontweight="bold", pad=10)
    fig.tight_layout()
    save(fig, "chart_3_seasonality.png")

# ── CHART 4: Fraud flag over time ─────────────────────────────────────────────
def chart_fraud_volume(df):
    weekly = df.groupby("Week").agg(
        Total=("Fraud Flag", "count"),
        Fraud=("Fraud Flag", lambda x: (x == "Y").sum())
    ).reset_index().sort_values("Week")
    weekly["Fraud Rate %"] = (weekly["Fraud"] / weekly["Total"] * 100).round(1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax1.bar(range(len(weekly)), weekly["Total"], color=BLUE, alpha=0.5, label="All claims")
    ax1.bar(range(len(weekly)), weekly["Fraud"], color=RED, alpha=0.7, label="Fraud flagged")
    ax1.set_ylabel("Claims")
    ax1.set_title("Claim volume vs fraud-flagged claims over time", fontweight="bold", pad=10)
    ax1.legend()

    ax2.plot(range(len(weekly)), weekly["Fraud Rate %"], color=RED, lw=2)
    ax2.fill_between(range(len(weekly)), weekly["Fraud Rate %"], alpha=0.15, color=RED)
    ax2.set_ylabel("Fraud rate (%)")
    ax2.set_xlabel("Week")

    step = max(1, len(weekly) // 8)
    ax2.set_xticks(range(0, len(weekly), step))
    ax2.set_xticklabels(weekly["Week"].iloc[::step], rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    save(fig, "chart_4_fraud_vs_volume.png")

# ── CHART 5: Top 10 cities by total amount billed ─────────────────────────────
def chart_top_cities(df):
    if "City" not in df.columns:
        print("  Skipping chart 5 — City column not found.")
        return

    city_totals = (df.groupby("City")["Total Amount Billed ($)"]
                     .sum()
                     .nlargest(10)
                     .sort_values()
                     .reset_index())
    city_totals.columns = ["City", "Total Billed ($)"]

    avg_billed = city_totals["Total Billed ($)"].mean()
    colors = [RED if v > avg_billed else BLUE
              for v in city_totals["Total Billed ($)"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(city_totals["City"], city_totals["Total Billed ($)"] / 1000,
                   color=colors, alpha=0.82, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"${bar.get_width():.0f}K",
                va="center", fontsize=9)

    ax.set_xlabel("Total amount billed ($000s)")
    ax.set_title("Top 10 cities by total amount billed", fontweight="bold", pad=10)
    ax.legend(handles=[
        Patch(color=RED,  alpha=0.82, label="Above average — review"),
        Patch(color=BLUE, alpha=0.82, label="Below average"),
    ])
    fig.tight_layout()
    save(fig, "chart_5_top_cities.png")

# ── CHART 6: Risk level distribution ──────────────────────────────────────────
def chart_risk_distribution(df):
    counts = df["Risk Level"].value_counts()
    order  = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = counts.reindex([x for x in order if x in counts.index])
    colors = [RISK_COLORS[r] for r in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, str(val),
                ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Risk level")
    ax.set_ylabel("Number of claims")
    ax.set_title("Fraud risk level distribution across all claims", fontweight="bold", pad=10)
    fig.tight_layout()
    save(fig, "chart_6_risk_distribution.png")

# ── Loss reserve forecast ─────────────────────────────────────────────────────
def reserve_forecast(df):
    monthly = df.groupby("Month")["Total Amount Billed ($)"].sum().reset_index()
    monthly = monthly.sort_values("Month").reset_index(drop=True)
    if len(monthly) < 3:
        print("  Not enough monthly data for reserve forecast.")
        return
    fc, lo, hi = simple_arima(monthly["Total Amount Billed ($)"], steps=3)
    print(f"\n  Loss reserve forecast (next 3 months):")
    for label, f, l, h in zip(["30-day", "60-day", "90-day"], fc, lo, hi):
        print(f"    {label}: ${max(0,f):>10,.0f}  (range: ${max(0,l):,.0f} – ${max(0,h):,.0f})")

# ── City billing alerts ───────────────────────────────────────────────────────
def city_alerts(df):
    if "City" not in df.columns:
        return
    city_weekly = (df.groupby(["City", "Week"])["Total Amount Billed ($)"]
                     .sum().reset_index().sort_values("Week"))
    alerts = []
    for city, grp in city_weekly.groupby("City"):
        if len(grp) < 3:
            continue
        grp = grp.sort_values("Week").reset_index(drop=True)
        rolling_avg = grp["Total Amount Billed ($)"].rolling(4, min_periods=2).mean()
        last_actual = grp["Total Amount Billed ($)"].iloc[-1]
        last_avg    = rolling_avg.iloc[-1]
        if last_avg > 0 and last_actual > last_avg * 1.3:
            pct = (last_actual - last_avg) / last_avg * 100
            alerts.append((city, last_actual, last_avg, pct))
    if alerts:
        print(f"\n  City billing alerts ({len(alerts)} flagged — > 30% above rolling avg):")
        for city, actual, avg, pct in sorted(alerts, key=lambda x: -x[3]):
            print(f"    {city:<20} actual: ${actual:>8,.0f}  avg: ${avg:>8,.0f}  (+{pct:.0f}%)")
    else:
        print("\n  No city billing alerts — all within normal range.")

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    print("\n" + "="*60)
    print("  PHASE 2 — FORECAST, SCORE & MAP")
    print("="*60)

    if not os.path.exists(DB_FILE):
        print("  ERROR: claims.db not found. Run phase1_ingest.py first.")
        return False

    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    print(f"\n  Loaded {len(df):,} claims from database")

    print("\n  Scoring fraud risk...")
    results = df.apply(lambda r: score_claim(r.to_dict()), axis=1)
    df["Risk Score"] = [r[0] for r in results]
    df["Risk Flags"] = [r[1] for r in results]
    df["Risk Level"] = df["Risk Score"].apply(risk_label)

    conn = sqlite3.connect(DB_FILE)
    df.to_sql("claims_scored", conn, if_exists="replace", index=False)
    conn.close()

    csv_cols = ["Date of Loss", "State", "City", "Accident Type",
                "Total Amount Billed ($)", "Medical Bills ($)",
                "Vehicles in Accident", "Fraud Flag", "Police Report on File",
                "Witness Count", "Risk Score", "Risk Level", "Risk Flags"]
    csv_cols = [c for c in csv_cols if c in df.columns]
    df[csv_cols].to_csv(f"{OUT_DIR}/nofault_scored.csv", index=False)
    print(f"  Scored CSV saved: {OUT_DIR}/nofault_scored.csv")

    rc = df["Risk Level"].value_counts()
    print(f"\n  Risk summary:")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        if level in rc:
            print(f"    {level:<10}: {rc[level]:>4} claims")

    print("\n  Generating charts...")
    weekly = chart_weekly_volume(df)
    chart_arima_forecast(weekly)
    chart_seasonality(df)
    chart_fraud_volume(df)
    chart_top_cities(df)
    chart_risk_distribution(df)

    reserve_forecast(df)
    city_alerts(df)

    print("\n  Phase 2 complete.\n")
    return True

if __name__ == "__main__":
    run()
