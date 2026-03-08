"""
config.py — Central project configuration.

Non-sensitive settings only. Secrets go in .env file.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# ──────────────────────────────────────────────
# Load .env (secrets)
# ──────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "stroke-data.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
REFERENCE_PATH = PROCESSED_DIR / "reference.csv"
INCOMING_DIR = DATA_DIR / "incoming"
MODEL_DIR = PROJECT_ROOT / "models"

# ──────────────────────────────────────────────
# Feature Definitions
# ──────────────────────────────────────────────
NUMERICAL_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = [
    "gender", "ever_married", "work_type",
    "Residence_type", "smoking_status"
]
TARGET = "stroke"
ALL_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
BMI_IMPUTE_STRATEGY = "median"
MIN_AGE = 18
TEST_SIZE = 0.2
RANDOM_STATE = 42
REFERENCE_SAMPLE_SIZE = 500

# ──────────────────────────────────────────────
# MLflow
# ──────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "stroke-prediction"
MLFLOW_TRACKING_URI = "file:" + str(PROJECT_ROOT / "mlruns")
MLFLOW_REGISTRY_MODEL_NAME = "stroke-classifier"

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
CV_FOLDS = 5
PRIMARY_METRIC = "f1"
MODEL_PROMOTION_THRESHOLD = 0.15  # minimum F1 to promote to Production

# ──────────────────────────────────────────────
# Serving
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ──────────────────────────────────────────────
# Monitoring (Evidently)
# ──────────────────────────────────────────────
DRIFT_THRESHOLD = 0.3  # max drift share before halting inference

# ──────────────────────────────────────────────
# GCP (secrets from .env)
# ──────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION", "asia-south1")
GCP_BUCKET_MODEL = os.getenv("GCP_BUCKET_MODEL")
GCP_BUCKET_INCOMING = os.getenv("GCP_BUCKET_INCOMING")
GCP_BUCKET_SCORED = os.getenv("GCP_BUCKET_SCORED")
GCP_SERVICE_ACCOUNT_KEY = os.getenv("GCP_SERVICE_ACCOUNT_KEY")

# ──────────────────────────────────────────────
# Grafana (secrets from .env, used in Week 4+)
# ──────────────────────────────────────────────
GRAFANA_PUSH_URL = os.getenv("GRAFANA_PUSH_URL")
GRAFANA_API_KEY = os.getenv("GRAFANA_API_KEY")

# ──────────────────────────────────────────────
# Alerting
# ──────────────────────────────────────────────
ALERT_EMAIL = os.getenv("ALERT_EMAIL")