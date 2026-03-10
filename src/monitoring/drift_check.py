"""
drift_check.py — Data drift detection using Evidently AI.

Compares incoming data against the reference dataset (training baseline).
Generates an HTML report and returns a drift decision.

Usage:
    python src/monitoring/drift_check.py --incoming data/incoming/new_patients.csv

Reads:  data/processed/reference.csv (baseline)
        incoming CSV (new data to check)
Writes: reports/drift_report.html
Returns: drift decision (PASS / FAIL)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from src.config import (
    REFERENCE_PATH, ALL_FEATURES, TARGET,
    DRIFT_THRESHOLD, PROJECT_ROOT
)


REPORTS_DIR = PROJECT_ROOT / "reports"


def load_reference():
    """Load the reference dataset (training baseline)."""
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference data not found at {REFERENCE_PATH}")
    df = pd.read_csv(REFERENCE_PATH)
    print(f"Reference data: {len(df)} records")
    return df


def load_incoming(path):
    """Load incoming data to check for drift."""
    incoming_path = Path(path)
    if not incoming_path.exists():
        raise FileNotFoundError(f"Incoming data not found at {incoming_path}")
    df = pd.read_csv(incoming_path)
    print(f"Incoming data:  {len(df)} records")
    return df


def check_drift(reference_df, incoming_df):
    """
    Run Evidently drift detection.

    Returns:
        dict with drift results:
            - drift_detected: bool
            - drift_share: float (fraction of drifted features)
            - drifted_features: list of feature names that drifted
            - n_drifted: int
            - n_total: int
            - report_path: str
    """
    # Use only feature columns (exclude target if present)
    feature_cols = [c for c in ALL_FEATURES if c in reference_df.columns and c in incoming_df.columns]

    ref = reference_df[feature_cols].copy()
    inc = incoming_df[feature_cols].copy()

    print(f"\nRunning drift detection on {len(feature_cols)} features...")

    # Build Evidently report
    report = Report(metrics=[
        DataDriftPreset(),
    ])

    report.run(reference_data=ref, current_data=inc)

    # Extract results
    report_dict = report.as_dict()

    # Get drift results from the report
    drift_results = report_dict["metrics"][0]["result"]
    drift_share = drift_results["share_of_drifted_columns"]
    n_drifted = drift_results["number_of_drifted_columns"]
    n_total = drift_results["number_of_columns"]
    dataset_drift = drift_results["dataset_drift"]

    # Get per-feature drift details
    drifted_features = []
    drift_by_columns = drift_results.get("drift_by_columns", {})
    for col, details in drift_by_columns.items():
        if details.get("drift_detected", False):
            drifted_features.append(col)

    # Save HTML report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
    report.save_html(str(report_path))

    # Drift decision based on threshold
    drift_exceeded = drift_share > DRIFT_THRESHOLD

    result = {
        "drift_detected": drift_exceeded,
        "dataset_drift": dataset_drift,
        "drift_share": round(drift_share, 4),
        "n_drifted": n_drifted,
        "n_total": n_total,
        "drifted_features": drifted_features,
        "report_path": str(report_path),
        "threshold": DRIFT_THRESHOLD,
    }

    return result


def print_result(result):
    """Pretty print drift detection results."""
    print(f"\n{'=' * 50}")
    print("DRIFT DETECTION RESULT")
    print(f"{'=' * 50}")
    print(f"  Features checked:   {result['n_total']}")
    print(f"  Features drifted:   {result['n_drifted']}")
    print(f"  Drift share:        {result['drift_share']:.1%}")
    print(f"  Threshold:          {result['threshold']:.1%}")

    if result["drifted_features"]:
        print(f"  Drifted features:   {', '.join(result['drifted_features'])}")

    if result["drift_detected"]:
        print(f"\n  Decision: ❌ DRIFT DETECTED — Inference blocked")
    else:
        print(f"\n  Decision: ✅ NO SIGNIFICANT DRIFT — Inference can proceed")

    print(f"\n  Report saved: {result['report_path']}")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Check data drift against reference")
    parser.add_argument("--incoming", required=True, help="Path to incoming CSV")
    args = parser.parse_args()

    # Load data
    reference_df = load_reference()
    incoming_df = load_incoming(args.incoming)

    # Check drift
    result = check_drift(reference_df, incoming_df)

    # Print result
    print_result(result)

    # Return exit code (0 = pass, 1 = drift detected)
    sys.exit(1 if result["drift_detected"] else 0)


if __name__ == "__main__":
    main()