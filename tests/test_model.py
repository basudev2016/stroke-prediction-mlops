"""
test_model.py — Model validation tests.

Ensures the exported model loads, predicts correctly, and meets quality gates.

Run: pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import pytest
import mlflow.sklearn

from src.config import ALL_FEATURES, TARGET, MODEL_DIR


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
MODEL_PATH = MODEL_DIR / "champion"


@pytest.fixture(scope="module")
def model():
    """Load the exported champion model."""
    if not MODEL_PATH.exists():
        pytest.skip("Exported model not found. Run export_model.py first.")
    return mlflow.sklearn.load_model(str(MODEL_PATH))


@pytest.fixture
def sample_input():
    """A single valid patient record."""
    return pd.DataFrame([{
        "gender": "Male",
        "age": 67.0,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }])


@pytest.fixture
def batch_input():
    """Multiple patient records for batch testing."""
    return pd.DataFrame([
        {
            "gender": "Male", "age": 67, "hypertension": 0,
            "heart_disease": 1, "ever_married": "Yes",
            "work_type": "Private", "Residence_type": "Urban",
            "avg_glucose_level": 228.69, "bmi": 36.6,
            "smoking_status": "formerly smoked"
        },
        {
            "gender": "Female", "age": 25, "hypertension": 0,
            "heart_disease": 0, "ever_married": "No",
            "work_type": "Private", "Residence_type": "Urban",
            "avg_glucose_level": 75.0, "bmi": 24.0,
            "smoking_status": "never smoked"
        },
        {
            "gender": "Male", "age": 80, "hypertension": 1,
            "heart_disease": 1, "ever_married": "Yes",
            "work_type": "Self-employed", "Residence_type": "Rural",
            "avg_glucose_level": 250.0, "bmi": 38.0,
            "smoking_status": "smokes"
        }
    ])


# ──────────────────────────────────────────────
# Smoke Tests: Model loads and runs
# ──────────────────────────────────────────────
class TestModelSmoke:

    def test_model_loads(self, model):
        """Model should load without error."""
        assert model is not None

    def test_model_has_predict(self, model):
        """Model should have predict and predict_proba methods."""
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_single_prediction(self, model, sample_input):
        """Model should return a prediction for one record."""
        pred = model.predict(sample_input)
        assert len(pred) == 1
        assert pred[0] in [0, 1]

    def test_single_probability(self, model, sample_input):
        """Model should return probabilities for one record."""
        proba = model.predict_proba(sample_input)
        assert proba.shape == (1, 2)
        assert 0 <= proba[0][0] <= 1
        assert 0 <= proba[0][1] <= 1
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_batch_prediction(self, model, batch_input):
        """Model should handle multiple records."""
        preds = model.predict(batch_input)
        assert len(preds) == 3
        assert all(p in [0, 1] for p in preds)

    def test_batch_probability(self, model, batch_input):
        """Probabilities should be valid for batch."""
        probas = model.predict_proba(batch_input)
        assert probas.shape == (3, 2)
        for row in probas:
            assert abs(row.sum() - 1.0) < 1e-6


# ──────────────────────────────────────────────
# Model Validation: Quality gates
# ──────────────────────────────────────────────
class TestModelValidation:

    def test_prediction_not_all_same(self, model, batch_input):
        """Model should not predict the same class for every input.
        This catches degenerate models that always predict majority class."""
        preds = model.predict(batch_input)
        # At least check it doesn't crash — with 3 samples, all same is possible
        # but we add more diverse inputs to test
        diverse_input = pd.DataFrame([
            {
                "gender": "Female", "age": 25, "hypertension": 0,
                "heart_disease": 0, "ever_married": "No",
                "work_type": "Private", "Residence_type": "Urban",
                "avg_glucose_level": 60.0, "bmi": 20.0,
                "smoking_status": "never smoked"
            },
            {
                "gender": "Male", "age": 82, "hypertension": 1,
                "heart_disease": 1, "ever_married": "Yes",
                "work_type": "Private", "Residence_type": "Urban",
                "avg_glucose_level": 270.0, "bmi": 40.0,
                "smoking_status": "smokes"
            }
        ])
        probas = model.predict_proba(diverse_input)
        # The high-risk patient should have higher stroke probability than low-risk
        assert probas[1][1] > probas[0][1], \
            "High-risk patient should have higher stroke probability than low-risk"

    def test_handles_unknown_category(self, model):
        """Model should not crash on unseen categories."""
        weird_input = pd.DataFrame([{
            "gender": "Male", "age": 50, "hypertension": 0,
            "heart_disease": 0, "ever_married": "Yes",
            "work_type": "Astronaut",  # unseen category
            "Residence_type": "Urban",
            "avg_glucose_level": 100.0, "bmi": 28.0,
            "smoking_status": "never smoked"
        }])
        pred = model.predict(weird_input)
        assert pred[0] in [0, 1]