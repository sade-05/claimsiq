# =============================================================================
# PHASE 1 — INGEST, CLEAN & STORE
# No-Fault Claims Intelligence System
# =============================================================================
# What this does:
#   1. Loads the Kaggle insurance_claims.csv
#   2. Renames all columns to real no-fault equivalents
#   3. Cleans dates and fills missing values
#   4. Writes everything into a SQLite database (claims.db)
#
# Run this first, once. All other phases read from claims.db.
# =============================================================================

import pandas as pd
import sqlite3
import os

# ── Config ────────────────────────────────────────────────────────────────────
DATA_FILE = "data/insurance_claims.csv"
DB_FILE   = "outputs/claims.db"

# ── Column remapping: Kaggle names → No-Fault equivalents ────────────────────
COLUMN_MAP = {
    "policy_number":              "Policy Number",
    "policy_bind_date":           "Policy Bind Date",
    "policy_state":               "Policy State",
    "policy_csl":                 "Policy CSL",
    "policy_deductable":          "Deductible ($)",
    "policy_annual_premium":      "Annual Premium ($)",
    "umbrella_limit":             "Umbrella Limit",
    "insured_zip":                "Claimant ZIP",
    "insured_sex":                "Claimant Gender",
    "insured_education_level":    "Education Level",
    "insured_occupation":         "Occupation",
    "insured_hobbies":            "Claimant Background",
    "insured_relationship":       "Relationship to Insured",
    "capital-gains":              "Capital Gains",
    "capital-loss":               "Capital Loss",
    "incident_date":              "Date of Loss",
    "incident_type":              "Accident Type",
    "collision_type":             "Collision Type",
    "incident_severity":          "Incident Severity",
    "authorities_contacted":      "Authorities Notified",
    "incident_state":             "State",
    "incident_city":              "City",
    "incident_location":          "Incident Location",
    "incident_hour_of_the_day":   "Hour of Incident",
    "number_of_vehicles_involved":"Vehicles in Accident",
    "property_damage":            "Property Damage",
    "bodily_injuries":            "Reported Injuries",
    "witnesses":                  "Witness Count",
    "police_report_available":    "Police Report on File",
    "total_claim_amount":         "Total Amount Billed ($)",
    "injury_claim":               "Medical Bills ($)",
    "property_claim":             "Property Damage ($)",
    "vehicle_claim":              "Vehicle Damage ($)",
    "auto_make":                  "Vehicle Make",
    "auto_model":                 "Vehicle Model",
    "auto_year":                  "Vehicle Year",
    "fraud_reported":             "Fraud Flag",
    "months_as_customer":         "Policy Tenure (Months)",
    "age":                        "Claimant Age",
}

def run():
    print("\n" + "="*60)
    print("  PHASE 1 — INGEST, CLEAN & STORE")
    print("="*60)

    # Check file exists
    if not os.path.exists(DATA_FILE):
        print(f"\n  ERROR: '{DATA_FILE}' not found.")
        print("  Please download insurance_claims.csv from Kaggle and")
        print("  place it in the data/ folder, then run again.\n")
        print("  Download: kaggle.com/datasets/buntyshah/auto-insurance-claims-data\n")
        return False

    # Load
    df = pd.read_csv(DATA_FILE)
    print(f"\n  Loaded {len(df):,} claims from '{DATA_FILE}'")

    # Replace '?' placeholders with NaN
    df.replace("?", pd.NA, inplace=True)

    # Rename columns
    df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns}, inplace=True)
    print(f"  Remapped {len(COLUMN_MAP)} columns to no-fault labels")

    # Parse dates
    if "Date of Loss" in df.columns:
        df["Date of Loss"] = pd.to_datetime(df["Date of Loss"], errors="coerce")
        df["Week"]  = df["Date of Loss"].dt.to_period("W").astype(str)
        df["Month"] = df["Date of Loss"].dt.to_period("M").astype(str)
        df["Year"]  = df["Date of Loss"].dt.year

    # Normalize fraud flag to Y/N.
    if "Fraud Flag" in df.columns:
        df["Fraud Flag"] = df["Fraud Flag"].str.strip().str.upper()

    # Fill key numerics
    for col in ["Total Amount Billed ($)", "Medical Bills ($)",
                "Property Damage ($)", "Vehicle Damage ($)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Write to SQLite
    os.makedirs("outputs", exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    df.to_sql("claims", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Written to database: {DB_FILE}")
    print(f"  Table 'claims' — {len(df):,} rows, {len(df.columns)} columns")
    print("\n  Phase 1 complete.\n")
    return True

if __name__ == "__main__":
    run()
