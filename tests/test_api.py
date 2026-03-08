"""
test_api.py — API integration and smoke tests.

Tests FastAPI endpoints with a test client (no server needed).

Run: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_patient():
    """A valid patient payload."""
    return {
        "gender": "Male",
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked"
    }


# ──────────────────────────────────────────────
# Smoke Tests: API is alive
# ──────────────────────────────────────────────
class TestSmoke:

    def test_health_endpoint(self, client):
        """Health check should return 200 with model status."""
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["model_name"] is not None

    def test_root_returns_html(self, client):
        """Root should serve the UI page."""
        res = client.get("/")
        assert res.status_code == 200

    def test_model_info(self, client):
        """Model info endpoint should return metadata."""
        res = client.get("/model-info")
        assert res.status_code == 200
        data = res.json()
        assert "model_name" in data
        assert "features" in data


# ──────────────────────────────────────────────
# Integration Tests: Prediction
# ──────────────────────────────────────────────
class TestPredict:

    def test_valid_prediction(self, client, valid_patient):
        """Valid input should return a prediction."""
        res = client.post("/predict", json=valid_patient)
        assert res.status_code == 200
        data = res.json()
        assert data["prediction"] in [0, 1]
        assert data["prediction_label"] in ["Stroke", "No Stroke"]
        assert 0 <= data["confidence"] <= 1
        assert data["model_version"] is not None

    def test_prediction_response_shape(self, client, valid_patient):
        """Response should have all expected fields."""
        res = client.post("/predict", json=valid_patient)
        data = res.json()
        expected_keys = {"prediction", "prediction_label", "confidence", "model_version"}
        assert expected_keys == set(data.keys())

    def test_low_risk_patient(self, client):
        """Young healthy patient should get No Stroke prediction."""
        payload = {
            "gender": "Female",
            "age": 25,
            "hypertension": 0,
            "heart_disease": 0,
            "ever_married": "No",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 75.0,
            "bmi": 24.0,
            "smoking_status": "never smoked"
        }
        res = client.post("/predict", json=payload)
        data = res.json()
        assert data["prediction"] == 0
        assert data["prediction_label"] == "No Stroke"

    def test_invalid_age_rejected(self, client, valid_patient):
        """Age below 18 should be rejected by Pydantic validation."""
        payload = {**valid_patient, "age": 5}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422

    def test_missing_field_rejected(self, client):
        """Incomplete payload should be rejected."""
        res = client.post("/predict", json={"gender": "Male", "age": 50})
        assert res.status_code == 422

    def test_invalid_bmi_rejected(self, client, valid_patient):
        """BMI outside valid range should be rejected."""
        payload = {**valid_patient, "bmi": 500}
        res = client.post("/predict", json=payload)
        assert res.status_code == 422


# ──────────────────────────────────────────────
# Integration Tests: Batch
# ──────────────────────────────────────────────
class TestBatchPredict:

    def test_batch_csv_upload(self, client):
        """Batch endpoint should accept CSV and return predictions."""
        csv_content = (
            "gender,age,hypertension,heart_disease,ever_married,"
            "work_type,Residence_type,avg_glucose_level,bmi,smoking_status\n"
            "Male,67,0,1,Yes,Private,Urban,228.69,36.6,formerly smoked\n"
            "Female,25,0,0,No,Private,Urban,75.0,24.0,never smoked\n"
        )
        res = client.post(
            "/predict/batch",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )
        assert res.status_code == 200
        data = res.json()
        assert data["total_records"] == 2
        assert len(data["predictions"]) == 2
        assert data["stroke_count"] + data["no_stroke_count"] == 2

    def test_batch_rejects_non_csv(self, client):
        """Non-CSV files should be rejected."""
        res = client.post(
            "/predict/batch",
            files={"file": ("test.txt", "not a csv", "text/plain")}
        )
        assert res.status_code == 400

    def test_batch_rejects_missing_columns(self, client):
        """CSV missing required columns should be rejected."""
        csv_content = "gender,age\nMale,67\n"
        res = client.post(
            "/predict/batch",
            files={"file": ("test.csv", csv_content, "text/csv")}
        )
        assert res.status_code == 400