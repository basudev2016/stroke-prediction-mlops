"""
preprocess.py — Data cleaning, feature pipeline, and train/test split.

Usage:
    python src/training/preprocess.py

Reads:  data/raw/stroke-data.csv
Writes: data/processed/train.csv
        data/processed/test.csv
        data/processed/reference.csv  (drift detection baseline)
"""

import sys
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.config import (
    RAW_DATA_PATH, PROCESSED_DIR, TRAIN_PATH, TEST_PATH, REFERENCE_PATH,
    NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, TARGET,
    BMI_IMPUTE_STRATEGY, MIN_AGE, TEST_SIZE, RANDOM_STATE, REFERENCE_SAMPLE_SIZE
)


def load_raw():
    """Load the raw CSV file."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw data: {df.shape[0]} records, {df.shape[1]} columns")
    return df


def clean(df):
    """
    Clean the raw dataset.

    Steps:
        1. Drop 'id' column (not a feature)
        2. Drop gender='Other' (1 record — too rare to be useful)
        3. Filter age >= MIN_AGE (children not clinically relevant)
        4. Fix BMI: 'N/A' string → NaN → median imputation
    """
    df = df.copy()

    # Drop ID
    df = df.drop(columns=["id"])

    # Drop 'Other' gender
    n_other = (df["gender"] == "Other").sum()
    df = df[df["gender"] != "Other"]
    print(f"Dropped {n_other} record(s) with gender='Other'")

    # Filter adults
    n_children = (df["age"] < MIN_AGE).sum()
    df = df[df["age"] >= MIN_AGE]
    print(f"Filtered out {n_children} records with age < {MIN_AGE}")

    # Fix BMI
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    n_bmi_null = df["bmi"].isna().sum()
    bmi_median = df["bmi"].median()
    df["bmi"] = df["bmi"].fillna(bmi_median)
    print(f"Imputed {n_bmi_null} missing BMI values with median ({bmi_median:.1f})")

    df = df.reset_index(drop=True)
    print(f"Clean dataset: {df.shape[0]} records")
    return df


def build_preprocessor():
    """
    Build a sklearn ColumnTransformer for the feature pipeline.

    This object is reused in:
        - train.py (inside imblearn Pipeline)
        - app.py (FastAPI loads the full pipeline, preprocessor included)
        - pipeline.py (Cloud Function batch inference)

    Returns:
        ColumnTransformer
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


def split(df):
    """
    Stratified train/test split.
    Also creates a reference dataset for Evidently drift detection.

    Returns:
        train_df, test_df, reference_df
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET]
    )

    # Reference data: subset of train for drift detection baseline
    reference_df = train_df.head(REFERENCE_SAMPLE_SIZE).copy()

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    reference_df = reference_df.reset_index(drop=True)

    return train_df, test_df, reference_df


def save(train_df, test_df, reference_df):
    """Save processed splits to CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    reference_df.to_csv(REFERENCE_PATH, index=False)

    print(f"\nSaved to {PROCESSED_DIR}/")
    print(f"  train.csv:     {len(train_df)} records")
    print(f"  test.csv:      {len(test_df)} records")
    print(f"  reference.csv: {len(reference_df)} records")


def main():
    print("=" * 50)
    print("STROKE DATA PREPROCESSING")
    print("=" * 50)

    # Load
    raw_df = load_raw()

    # Clean
    clean_df = clean(raw_df)

    # Show class distribution
    stroke_pct = clean_df[TARGET].mean() * 100
    print(f"\nClass distribution:")
    print(f"  No stroke (0): {(clean_df[TARGET] == 0).sum()} ({100 - stroke_pct:.1f}%)")
    print(f"  Stroke    (1): {(clean_df[TARGET] == 1).sum()} ({stroke_pct:.1f}%)")

    # Show feature summary
    print(f"\nFeatures:")
    print(f"  Numerical:   {NUMERICAL_FEATURES}")
    print(f"  Binary:      {BINARY_FEATURES}")
    print(f"  Categorical: {CATEGORICAL_FEATURES}")
    print(f"  Target:      {TARGET}")

    # Split
    train_df, test_df, reference_df = split(clean_df)

    # Save
    save(train_df, test_df, reference_df)

    print("\nDone. Next step: python src/training/train.py")


if __name__ == "__main__":
    main()