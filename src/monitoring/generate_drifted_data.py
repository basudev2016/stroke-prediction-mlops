"""
generate_drifted_data.py — Create a deliberately drifted dataset for demo.

Modifies feature distributions to trigger drift detection.

Usage:
    python src/monitoring/generate_drifted_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np

from src.config import REFERENCE_PATH, PROJECT_ROOT

INCOMING_DIR = PROJECT_ROOT / "data" / "incoming"


def main():
    ref = pd.read_csv(REFERENCE_PATH)
    drifted = ref.copy()

    np.random.seed(42)

    # Shift age distribution — make everyone much older
    drifted["age"] = drifted["age"] + np.random.uniform(15, 25, len(drifted))
    drifted["age"] = drifted["age"].clip(18, 100)

    # Shift glucose — much higher
    drifted["avg_glucose_level"] = drifted["avg_glucose_level"] * 1.8

    # Shift BMI — much higher
    drifted["bmi"] = drifted["bmi"] + np.random.uniform(8, 15, len(drifted))

    # Change smoking distribution — everyone smokes now
    drifted["smoking_status"] = "smokes"

    # Change gender distribution — all male
    drifted["gender"] = "Male"

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INCOMING_DIR / "drifted_patients.csv"
    drifted.to_csv(output_path, index=False)

    print(f"Generated drifted data: {len(drifted)} records")
    print(f"Saved to: {output_path}")
    print(f"\nModifications:")
    print(f"  age:                shifted +15 to +25 years")
    print(f"  avg_glucose_level:  multiplied by 1.8x")
    print(f"  bmi:                shifted +8 to +15")
    print(f"  smoking_status:     all set to 'smokes'")
    print(f"  gender:             all set to 'Male'")
    print(f"\nTest with: python src/monitoring/drift_check.py --incoming {output_path}")


if __name__ == "__main__":
    main()