# No-Fault Claims Intelligence System

> A time series forecasting, fraud detection, and geographic analytics pipeline — built by a no-fault claims examiner, for no-fault claims examiners.

---

## What This Project Is

Every week, no-fault claims examiners face the same challenge: a growing stack of claims, tight 30-day deadlines, and limited tools to separate routine filings from fraud. Most analysis still happens manually — in spreadsheets, by instinct, one claim at a time.

This project changes that. It takes publicly available auto insurance claims data and runs it through a three-phase intelligence pipeline that forecasts claim volume, scores every claim for fraud risk, maps geographic patterns, and delivers a ready-to-use weekly briefing — automatically.

The goal is not to replace the examiner. It is to put better information in front of them faster.

---

## Who This Is For

This project was built as a graduate-level capstone demonstrating how data science techniques apply directly to real insurance operations. It is designed to be readable by:

- **Claims examiners and supervisors** who want to understand what the system does and why
- **Insurance operations teams** evaluating analytics tools
- **Recruiters and hiring managers** reviewing applied data science work
- **Data science students** learning how ML pipelines work in a real business context

---

## The Dataset

**Source:** [Auto Insurance Claims — Kaggle (bunty shah)](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data)  
**Size:** ~1,000 claims · 40 columns · free download

This dataset covers auto insurance claims from **7 mid-Atlantic US states**, two of which — New York and Pennsylvania — are no-fault states, making the data directly applicable to no-fault claims work.

| State | Claims | No-Fault? |
|---|---|---|
| New York (NY) | 262 | Yes |
| South Carolina (SC) | 248 | No |
| West Virginia (WV) | 217 | No |
| Virginia (VA) | 110 | No |
| North Carolina (NC) | 110 | No |
| Pennsylvania (PA) | 30 | Yes |
| Ohio (OH) | 23 | No |

---

## Column Remapping

Every Kaggle column is renamed to real no-fault language on load. Every chart, CSV, and PDF uses the terminology a claims examiner actually uses on the job.

| Original (Kaggle) | Renamed (No-Fault) |
|---|---|
| `incident_date` | Date of Loss |
| `incident_type` | Accident Type |
| `total_claim_amount` | Total Amount Billed ($) |
| `injury_claim` | Medical Bills ($) |
| `fraud_reported` | Fraud Flag |
| `number_of_vehicles_involved` | Vehicles in Accident |
| `witnesses` | Witness Count |
| `police_report_available` | Police Report on File |
| `incident_state` | State |
| `incident_city` | City |
| `auto_make` | Vehicle Make |
| `months_as_customer` | Policy Tenure (Months) |

---

## The Three Phases

### Phase 1 — Ingest, Clean & Store
Loads the CSV, renames all columns, parses dates, and writes everything into a persistent database. Runs once and takes about five seconds. Every other phase reads from the database — not the original file.

### Phase 2 — Forecast, Score & Map
All intelligence lives here. Four things run simultaneously:

- **Time series forecasting** — Moving Average (4-week) and ARIMA models run side by side on weekly claim counts. A third ARIMA series forecasts total dollar payouts for 30, 60, and 90 days ahead for reserve planning.
- **Fraud risk scoring** — every claim receives a score based on weighted indicators and is categorized as LOW, MEDIUM, HIGH, or CRITICAL.
- **Anomaly flagging** — volume spikes and provider billing breaches are detected automatically.
- **Geographic mapping** — claim density and fraud rate are mapped by US state using choropleth charts.

### Phase 3 — Report & Deliver
Charts, KPIs, scores, and alerts are packaged into a PDF weekly briefing and an updated Excel claims tracker. Generated automatically — no manual steps.

---

## Sample Output Charts

The following charts are actual outputs from the system, generated using simulated data that mirrors the Kaggle dataset structure.

---

### Chart 1 — Weekly Claim Volume with Moving Average

![Weekly claim volume](outputs/chart1.png)

The bars show how many new claims were filed each week. The orange line is the **4-week moving average** — it smooths random week-to-week variation so the underlying trend becomes visible. A sharp spike where the bars suddenly tower above the orange line is an anomaly worth investigating — either a real accident surge or an organized filing ring.

---

### Chart 2 — Moving Average vs ARIMA Forecast

![MA vs ARIMA forecast](outputs/chart2.png)

Historical actuals on the left of the dotted line. To the right is the **ARIMA 8-week forecast** (dashed red) with its confidence band (shaded area). If actual claims land above the top edge of that band, the week is automatically flagged as a spike. This chart goes directly into supervisor presentations or reserve planning meetings.

---

### Chart 3 — Seasonal Claim Distribution

![Seasonality](outputs/chart3.png)

Average claims by calendar month across the full dataset. Patterns visible here — summer peaks, January spikes from winter road conditions — allow examiners to plan staffing and reserve budgets in advance rather than reacting after the fact. In a real no-fault book, certain months consistently generate 30–40% more claims than others.

---

### Chart 4 — Claim Volume vs Fraud-Flagged Claims

![Fraud vs volume](outputs/chart4.png)

The top panel overlays total weekly claims (blue) against fraud-flagged claims (red). The bottom panel tracks the fraud rate percentage week by week. The key question this answers: when fraud spikes, is it because total volume went up, or because fraud increased independently? Independent fraud spikes — red jumps while blue stays flat — point to organized activity rather than random variation.

---

### Chart 5 — Fraud Risk Level Distribution

![Risk distribution](outputs/chart5.png)

Every claim receives a risk score from the scoring engine. This chart shows how scores distribute across the four tiers. In a healthy book, most claims land LOW or MEDIUM. A large CRITICAL or HIGH bar signals significant fraud exposure requiring immediate SIU escalation. Examiners use this as a daily triage tool: start with CRITICAL, work down.

---

## Project Structure

```
claimsiq/
│
├── README.md
├── run_all.py                     ← One command runs everything
├── requirements.txt
├── .gitignore
│
├── data/
│   └── insurance_claims.csv       ← Download from Kaggle (not in repo)
│
├── scripts/
│   ├── phase1_ingest.py           ← Load, remap, store
│   ├── phase2_forecast.py         ← Forecast, score, map, charts
│   └── phase3_report.py           ← PDF + Excel update
│
├── outputs/                       ← All outputs land here (auto-created)
│   ├── claims.db
│   ├── nofault_scored.csv
│   ├── chart_1_weekly_volume.png
│   ├── chart_2_ma_vs_arima.png
│   ├── chart_3_seasonality.png
│   ├── chart_4_fraud_vs_volume.png
│   ├── chart_5_billing_trends.png
│   ├── chart_6_geo_claim_density.png
│   ├── chart_7_geo_fraud_rate.png
│   ├── chart_8_risk_distribution.png
│   └── weekly_report.pdf
│
└── excel/
    └── nofault_claims_toolkit.xlsx
```

---

## How to Run It

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/claimsiq.git
cd claimsiq

# 2. Install libraries
pip install -r requirements.txt

# 3. Download the dataset
# Visit: kaggle.com/datasets/buntyshah/auto-insurance-claims-data
# Download insurance_claims.csv and place it in the data/ folder

# 4. Run the full pipeline
python run_all.py
```

The pipeline takes 30–60 seconds. When finished, open `outputs/weekly_report.pdf` for your generated briefing and `outputs/nofault_scored.csv` for the full scored claim list.

---

## Plain Language Glossary

**Moving Average** — Smooths week-to-week variation by averaging the last few weeks of data. Like a weather forecast that reads the trend rather than reacting to every single day.

**ARIMA** — A statistical model that learns patterns from historical data — including trends and seasonal cycles — and uses them to predict the future, with a confidence range so you know how certain the prediction is.

**Time Series** — Any data recorded over time at regular intervals. Weekly claim counts are a time series. Monthly payouts are a time series. The models in this system are built specifically for this kind of data.

**Choropleth Map** — A map where geographic regions are shaded by intensity. Darker means more.

**Risk Scoring** — A points system where fraud indicators accumulate into a total score. No single flag triggers an alert — it is the combination that matters.

---

## What This Demonstrates

From a data science perspective: time series modeling, supervised scoring, geospatial visualization, database persistence, and automated reporting — applied to a regulated business domain with real compliance constraints.

From a claims operations perspective: that the same intelligence frameworks used in large carriers can be built and understood by a working examiner — not just a dedicated data science team.

---

## Roadmap

- [ ] Fraud probability score (0–100%) using scikit-learn classifier
- [ ] Claim aging and days-to-close trend analysis
- [ ] Cohort analysis by filing month
- [ ] PowerPoint briefing auto-generator

---

## Data Privacy

This project uses only publicly available, anonymized sample data. No real claimant personally identifiable information is included. If adapting for live employer data, ensure compliance with your organization's data governance and privacy policies before storing or processing any real claims.

---

## License

MIT — free to use, adapt, and share with attribution.
