"""
monitor.py
Drift detection using Evidently AI.
Monitors incoming prediction distributions
against training baseline.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = "mediscan"

# Drift threshold — alert if PSI > 0.2
PSI_THRESHOLD = 0.2


def get_predictions_df() -> pd.DataFrame:
    """
    Load last 500 predictions from MongoDB
    into a DataFrame for drift analysis.
    """
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    docs = list(
        db["predictions"]
        .find({}, {"_id": 0, "probability": 1,
                   "predicted_class": 1, "risk_level": 1,
                   "timestamp": 1})
        .sort("timestamp", -1)
        .limit(500)
    )

    if not docs:
        return pd.DataFrame()

    return pd.DataFrame(docs)


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1  → No drift
    PSI < 0.2  → Moderate drift
    PSI >= 0.2 → Significant drift — retrain!
    """
    baseline_pct, edges = np.histogram(
        baseline, bins=bins, range=(0, 1)
    )
    current_pct, _      = np.histogram(
        current, bins=edges
    )

    # Avoid division by zero
    baseline_pct = np.where(
        baseline_pct == 0, 0.0001, baseline_pct
    ) / len(baseline)

    current_pct = np.where(
        current_pct == 0, 0.0001, current_pct
    ) / len(current)

    psi = np.sum(
        (current_pct - baseline_pct)
        * np.log(current_pct / baseline_pct)
    )

    return float(round(psi, 4))


def run_drift_check(baseline_probs: list) -> dict:
    """
    Compare incoming predictions against baseline.
    Returns drift report with alert flag.
    """
    df = get_predictions_df()

    if df.empty or len(df) < 50:
        return {
            "status":    "insufficient_data",
            "message":   "Need at least 50 predictions to run drift check.",
            "alert":     False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    current_probs = df["probability"].values
    psi           = compute_psi(
        np.array(baseline_probs),
        current_probs,
    )

    alert = psi >= PSI_THRESHOLD

    report = {
        "psi":                round(psi, 4),
        "alert":              alert,
        "threshold":          PSI_THRESHOLD,
        "total_predictions":  len(df),
        "pneumonia_rate":     round(
            (df["predicted_class"] == "PNEUMONIA").mean(), 4
        ),
        "avg_probability":    round(df["probability"].mean(), 4),
        "high_risk_rate":     round(
            (df["risk_level"] == "HIGH").mean(), 4
        ),
        "status": "DRIFT DETECTED — Retrain recommended!" if alert
                  else "Stable — No action needed.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Save report to file
    report_path = Path("drift_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[monitor] PSI: {psi:.4f} | Alert: {alert}")
    print(f"[monitor] Status: {report['status']}")

    return report


if __name__ == "__main__":
    # Baseline — training set probability distribution
    # These are approximate values from our training run
    baseline = list(np.random.beta(2, 5, 1000))
    report   = run_drift_check(baseline)
    print(json.dumps(report, indent=2))