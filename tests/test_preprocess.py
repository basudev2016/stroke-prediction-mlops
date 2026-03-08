"""
test_preprocess.py — Unit tests for data preprocessing.

Run: pytest tests/test_preprocess.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import pytest

from src.training.preprocess import clean, build_preprocessor, split
from src.config import (
    NUMERICAL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES,
    TARGET, ALL_FEATURES, MIN_AGE
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def sample_raw_df():
    """Minimal raw dataset mimicking the real CSV structure."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "gender": ["Male", "Female", "Other", "Male", "Female"],
        "age": [67, 25, 45, 10, 80],
        "hypertension": [0, 0, 1, 0, 1],
        "heart_disease": [1, 0, 0, 0, 1],
        "ever_married": ["Yes", "No", "Yes", "No", "Yes"],
        "work_type": ["Private", "Private", "Govt_job", "children", "Self-employed"],
        "Residence_type": ["Urban", "Rural", "Urban", "Rural", "Urban"],
        "avg_glucose_level": [228.69, 75.0, 180.0, 90.0, 200.0],
        "bmi": [36.6, 24.5, "N/A", 18.0, 29.0],
        "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown", "never smoked"],
        "stroke": [1, 0, 0, 0, 1]
    })


@pytest.fixture
def clean_df(sample_raw_df):
    """Cleaned dataset."""
    return clean(sample_raw_df)


# ──────────────────────────────────────────────
# Unit Tests: Cleaning
# ──────────────────────────────────────────────
class TestClean:

    def test_id_column_dropped(self, clean_df):
        """ID column should be removed."""
        assert "id" not in clean_df.columns

    def test_other_gender_removed(self, clean_df):
        """Records with gender='Other' should be removed."""
        assert "Other" not in clean_df["gender"].values

    def test_children_filtered(self, clean_df):
        """Records with age < MIN_AGE should be removed."""
        assert (clean_df["age"] >= MIN_AGE).all()

    def test_bmi_no_nulls(self, clean_df):
        """BMI should have no missing values after imputation."""
        assert clean_df["bmi"].isna().sum() == 0

    def test_bmi_is_numeric(self, clean_df):
        """BMI should be float after conversion."""
        assert clean_df["bmi"].dtype in [np.float64, np.float32]

    def test_expected_columns(self, clean_df):
        """All expected feature columns + target should be present."""
        for col in ALL_FEATURES + [TARGET]:
            assert col in clean_df.columns

    def test_record_count(self, clean_df):
        """Should have fewer records after filtering (removed Other + children)."""
        # Original 5: remove 1 Other + 1 child = 3 remaining
        assert len(clean_df) == 3


# ──────────────────────────────────────────────
# Unit Tests: Preprocessor
# ──────────────────────────────────────────────
class TestPreprocessor:

    def test_preprocessor_builds(self):
        """ColumnTransformer should build without error."""
        preprocessor = build_preprocessor()
        assert preprocessor is not None

    def test_preprocessor_fits(self, clean_df):
        """Preprocessor should fit on clean data."""
        preprocessor = build_preprocessor()
        X = clean_df[ALL_FEATURES]
        preprocessor.fit(X)
        transformed = preprocessor.transform(X)
        assert transformed.shape[0] == len(X)
        assert transformed.shape[1] > 0

    def test_preprocessor_handles_unknown_categories(self, clean_df):
        """OneHotEncoder with handle_unknown='ignore' should not crash on unseen values."""
        preprocessor = build_preprocessor()
        X = clean_df[ALL_FEATURES]
        preprocessor.fit(X)

        # Create a record with an unseen category
        new_record = X.iloc[[0]].copy()
        new_record["work_type"] = "Astronaut"
        transformed = preprocessor.transform(new_record)
        assert transformed.shape[0] == 1


# ──────────────────────────────────────────────
# Unit Tests: Split
# ──────────────────────────────────────────────
class TestSplit:

    def test_split_sizes(self):
        """Train/test split should have correct proportions."""
        df = pd.DataFrame({
            **{f: range(100) for f in ALL_FEATURES},
            TARGET: [0] * 90 + [1] * 10
        })
        # Make categorical columns string type
        for col in CATEGORICAL_FEATURES:
            df[col] = "value"
        df["gender"] = "Male"

        train, test, ref = split(df)
        assert len(train) == 80
        assert len(test) == 20
        assert len(ref) <= 500

    def test_split_stratified(self):
        """Both splits should preserve target distribution."""
        df = pd.DataFrame({
            **{f: range(200) for f in ALL_FEATURES},
            TARGET: [0] * 180 + [1] * 20
        })
        for col in CATEGORICAL_FEATURES:
            df[col] = "value"

        train, test, _ = split(df)
        train_pct = train[TARGET].mean()
        test_pct = test[TARGET].mean()
        # Should be roughly equal (10% positive)
        assert abs(train_pct - test_pct) < 0.05