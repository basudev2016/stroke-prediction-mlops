# """
# app.py — FastAPI serving layer for stroke prediction model.

# Usage:
#     uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# Endpoints:
#     GET  /            — Serves the UI
#     GET  /health      — Health check + model info
#     POST /predict     — Single patient prediction
#     POST /predict/batch — Batch prediction (CSV upload)
#     GET  /model-info  — Model metadata
# """

# import sys
# import time
# import json
# import logging
# from pathlib import Path
# from typing import Optional
# from contextlib import asynccontextmanager

# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# import pandas as pd
# import numpy as np
# import mlflow
# import mlflow.sklearn
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import io

# from src.config import (
#     MLFLOW_TRACKING_URI, MLFLOW_REGISTRY_MODEL_NAME,
#     ALL_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES,
#     CATEGORICAL_FEATURES, TARGET, API_HOST, API_PORT
# )

# # ──────────────────────────────────────────────
# # Structured Logging
# # ──────────────────────────────────────────────
# logger = logging.getLogger("stroke-api")
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter("%(message)s"))
# logger.addHandler(handler)

# def log_event(event: str, **kwargs):
#     """Log a structured JSON event."""
#     entry = {"event": event, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
#     entry.update(kwargs)
#     logger.info(json.dumps(entry))


# # ──────────────────────────────────────────────
# # Model loading
# # ──────────────────────────────────────────────
# model = None
# model_version = None
# model_load_time = None

# def load_model():
#     """Load the champion model from MLflow Registry."""
#     global model, model_version, model_load_time

#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     client = mlflow.tracking.MlflowClient()

#     try:
#         # Get champion version
#         version_info = client.get_model_version_by_alias(
#             name=MLFLOW_REGISTRY_MODEL_NAME,
#             alias="champion"
#         )
#         model_version = version_info.version
#         model_uri = f"models:/{MLFLOW_REGISTRY_MODEL_NAME}@champion"
#         model = mlflow.sklearn.load_model(model_uri)
#         model_load_time = time.strftime("%Y-%m-%dT%H:%M:%S")

#         log_event("model_loaded",
#                   model_name=MLFLOW_REGISTRY_MODEL_NAME,
#                   version=model_version)

#     except Exception as e:
#         log_event("model_load_failed", error=str(e))
#         raise RuntimeError(f"Failed to load model: {e}")


# # ──────────────────────────────────────────────
# # Lifespan (startup/shutdown)
# # ──────────────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     log_event("server_starting")
#     load_model()
#     log_event("server_ready", port=API_PORT)
#     yield
#     # Shutdown
#     log_event("server_stopping")


# # ──────────────────────────────────────────────
# # FastAPI app
# # ──────────────────────────────────────────────
# app = FastAPI(
#     title="Stroke Prediction API",
#     description="MLOps demo — Stroke risk prediction with model versioning and monitoring",
#     version="1.0.0",
#     lifespan=lifespan
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve static UI files
# UI_DIR = Path(__file__).resolve().parents[2] / "ui"
# if UI_DIR.exists():
#     app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


# # ──────────────────────────────────────────────
# # Pydantic models
# # ──────────────────────────────────────────────
# class PatientInput(BaseModel):
#     """Single patient input for prediction."""
#     gender: str = Field(..., example="Male", description="Male or Female")
#     age: float = Field(..., ge=18, le=120, example=67.0)
#     hypertension: int = Field(..., ge=0, le=1, example=0)
#     heart_disease: int = Field(..., ge=0, le=1, example=1)
#     ever_married: str = Field(..., example="Yes", description="Yes or No")
#     work_type: str = Field(..., example="Private",
#                            description="Private, Self-employed, Govt_job, Never_worked")
#     Residence_type: str = Field(..., example="Urban", description="Urban or Rural")
#     avg_glucose_level: float = Field(..., ge=0, example=228.69)
#     bmi: float = Field(..., ge=10, le=100, example=36.6)
#     smoking_status: str = Field(..., example="formerly smoked",
#                                 description="formerly smoked, never smoked, smokes, Unknown")

# class PredictionResponse(BaseModel):
#     """Response for a single prediction."""
#     prediction: int
#     prediction_label: str
#     confidence: float
#     model_version: str

# class BatchPredictionResponse(BaseModel):
#     """Response for batch prediction."""
#     total_records: int
#     predictions: list
#     stroke_count: int
#     no_stroke_count: int
#     model_version: str


# # ──────────────────────────────────────────────
# # Endpoints
# # ──────────────────────────────────────────────
# @app.get("/")
# async def root():
#     """Serve the UI."""
#     index_path = UI_DIR / "index.html"
#     if index_path.exists():
#         return FileResponse(str(index_path))
#     return {"message": "Stroke Prediction API is running. UI not found at /ui/"}


# @app.get("/health")
# async def health():
#     """Health check endpoint."""
#     return {
#         "status": "healthy" if model is not None else "unhealthy",
#         "model_name": MLFLOW_REGISTRY_MODEL_NAME,
#         "model_version": model_version,
#         "model_loaded_at": model_load_time,
#     }


# @app.get("/model-info")
# async def model_info():
#     """Return model metadata."""
#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     client = mlflow.tracking.MlflowClient()

#     version_info = client.get_model_version_by_alias(
#         name=MLFLOW_REGISTRY_MODEL_NAME, alias="champion"
#     )

#     # Get the run metrics
#     run = client.get_run(version_info.run_id)
#     metrics = {k: v for k, v in run.data.metrics.items() if k.startswith("cv_")}

#     return {
#         "model_name": MLFLOW_REGISTRY_MODEL_NAME,
#         "version": version_info.version,
#         "alias": "champion",
#         "run_id": version_info.run_id,
#         "registered_at": str(version_info.creation_timestamp),
#         "training_metrics": metrics,
#         "features": ALL_FEATURES,
#     }


# @app.post("/predict", response_model=PredictionResponse)
# async def predict(patient: PatientInput):
#     """Predict stroke risk for a single patient."""
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     start = time.time()

#     # Convert input to DataFrame (model expects DataFrame with feature names)
#     input_data = pd.DataFrame([patient.model_dump()])

#     # Predict
#     prediction = int(model.predict(input_data)[0])
#     proba = model.predict_proba(input_data)[0]
#     confidence = float(round(max(proba), 4))
#     label = "Stroke" if prediction == 1 else "No Stroke"

#     latency_ms = round((time.time() - start) * 1000, 2)

#     # Log
#     log_event("prediction",
#               prediction=prediction,
#               label=label,
#               confidence=confidence,
#               latency_ms=latency_ms,
#               model_version=model_version,
#               age=patient.age,
#               gender=patient.gender)

#     return PredictionResponse(
#         prediction=prediction,
#         prediction_label=label,
#         confidence=confidence,
#         model_version=str(model_version)
#     )


# @app.post("/predict/batch", response_model=BatchPredictionResponse)
# async def predict_batch(file: UploadFile = File(...)):
#     """Batch prediction from CSV upload."""
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")

#     if not file.filename.endswith(".csv"):
#         raise HTTPException(status_code=400, detail="Only CSV files are accepted")

#     start = time.time()

#     try:
#         contents = await file.read()
#         df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

#     # Validate required columns
#     missing = set(ALL_FEATURES) - set(df.columns)
#     if missing:
#         raise HTTPException(status_code=400,
#                             detail=f"Missing columns: {missing}")

#     # Predict
#     predictions = model.predict(df[ALL_FEATURES]).tolist()
#     probas = model.predict_proba(df[ALL_FEATURES])[:, 1].tolist()

#     # Build response
#     results = []
#     for i, (pred, prob) in enumerate(zip(predictions, probas)):
#         results.append({
#             "index": i,
#             "prediction": int(pred),
#             "label": "Stroke" if pred == 1 else "No Stroke",
#             "confidence": round(float(max(prob, 1 - prob)), 4)
#         })

#     stroke_count = sum(predictions)
#     latency_ms = round((time.time() - start) * 1000, 2)

#     log_event("batch_prediction",
#               total_records=len(df),
#               stroke_count=stroke_count,
#               latency_ms=latency_ms,
#               model_version=model_version,
#               filename=file.filename)

#     return BatchPredictionResponse(
#         total_records=len(df),
#         predictions=results,
#         stroke_count=stroke_count,
#         no_stroke_count=len(df) - stroke_count,
#         model_version=str(model_version)
#     )


# # ──────────────────────────────────────────────
# # Run directly
# # ──────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("src.serving.app:app", host=API_HOST, port=API_PORT, reload=True)


# Now we need to update app.py so it can load from either MLflow Registry (local dev) or the exported folder (Docker)

"""
app.py — FastAPI serving layer for stroke prediction model.

Usage:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /            — Serves the UI
    GET  /health      — Health check + model info
    POST /predict     — Single patient prediction
    POST /predict/batch — Batch prediction (CSV upload)
    GET  /model-info  — Model metadata
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import io

from src.config import (
    MLFLOW_TRACKING_URI, MLFLOW_REGISTRY_MODEL_NAME,
    ALL_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES,
    CATEGORICAL_FEATURES, TARGET, API_HOST, API_PORT
)

# ──────────────────────────────────────────────
# Structured Logging
# ──────────────────────────────────────────────
logger = logging.getLogger("stroke-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

def log_event(event: str, **kwargs):
    """Log a structured JSON event."""
    entry = {"event": event, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    entry.update(kwargs)
    logger.info(json.dumps(entry))


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
model = None
model_version = None
model_load_time = None

def load_model():
    """
    Load model with fallback:
      1. Try exported model from models/champion/ (Docker / production)
      2. Fall back to MLflow Registry (local development)
    """
    global model, model_version, model_load_time

    exported_path = Path(__file__).resolve().parents[2] / "models" / "champion"

    # Option 1: Exported model (Docker)
    if exported_path.exists():
        try:
            model = mlflow.sklearn.load_model(str(exported_path))
            model_version = "exported"
            model_load_time = time.strftime("%Y-%m-%dT%H:%M:%S")
            log_event("model_loaded", source="exported", path=str(exported_path))
            return
        except Exception as e:
            log_event("exported_model_load_failed", error=str(e))

    # Option 2: MLflow Registry (local dev)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        version_info = client.get_model_version_by_alias(
            name=MLFLOW_REGISTRY_MODEL_NAME,
            alias="champion"
        )
        model_version = version_info.version
        model_uri = f"models:/{MLFLOW_REGISTRY_MODEL_NAME}@champion"
        model = mlflow.sklearn.load_model(model_uri)
        model_load_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        log_event("model_loaded", source="registry", version=model_version)
    except Exception as e:
        log_event("model_load_failed", error=str(e))
        raise RuntimeError(f"Failed to load model: {e}")


# ──────────────────────────────────────────────
# Lifespan (startup/shutdown)
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log_event("server_starting")
    load_model()
    log_event("server_ready", port=API_PORT)
    yield
    # Shutdown
    log_event("server_stopping")


# ──────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────
app = FastAPI(
    title="Stroke Prediction API",
    description="MLOps demo — Stroke risk prediction with model versioning and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI files
UI_DIR = Path(__file__).resolve().parents[2] / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")


# ──────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────
class PatientInput(BaseModel):
    """Single patient input for prediction."""
    gender: str = Field(..., example="Male", description="Male or Female")
    age: float = Field(..., ge=18, le=120, example=67.0)
    hypertension: int = Field(..., ge=0, le=1, example=0)
    heart_disease: int = Field(..., ge=0, le=1, example=1)
    ever_married: str = Field(..., example="Yes", description="Yes or No")
    work_type: str = Field(..., example="Private",
                           description="Private, Self-employed, Govt_job, Never_worked")
    Residence_type: str = Field(..., example="Urban", description="Urban or Rural")
    avg_glucose_level: float = Field(..., ge=0, example=228.69)
    bmi: float = Field(..., ge=10, le=100, example=36.6)
    smoking_status: str = Field(..., example="formerly smoked",
                                description="formerly smoked, never smoked, smokes, Unknown")

class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    prediction: int
    prediction_label: str
    confidence: float
    model_version: str

class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    total_records: int
    predictions: list
    stroke_count: int
    no_stroke_count: int
    model_version: str


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/")
async def root():
    """Serve the UI."""
    index_path = UI_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Stroke Prediction API is running. UI not found at /ui/"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_name": MLFLOW_REGISTRY_MODEL_NAME,
        "model_version": model_version,
        "model_loaded_at": model_load_time,
    }


@app.get("/model-info")
async def model_info():
    """Return model metadata."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    version_info = client.get_model_version_by_alias(
        name=MLFLOW_REGISTRY_MODEL_NAME, alias="champion"
    )

    # Get the run metrics
    run = client.get_run(version_info.run_id)
    metrics = {k: v for k, v in run.data.metrics.items() if k.startswith("cv_")}

    return {
        "model_name": MLFLOW_REGISTRY_MODEL_NAME,
        "version": version_info.version,
        "alias": "champion",
        "run_id": version_info.run_id,
        "registered_at": str(version_info.creation_timestamp),
        "training_metrics": metrics,
        "features": ALL_FEATURES,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput):
    """Predict stroke risk for a single patient."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Convert input to DataFrame (model expects DataFrame with feature names)
    input_data = pd.DataFrame([patient.model_dump()])

    # Predict
    prediction = int(model.predict(input_data)[0])
    proba = model.predict_proba(input_data)[0]
    confidence = float(round(max(proba), 4))
    label = "Stroke" if prediction == 1 else "No Stroke"

    latency_ms = round((time.time() - start) * 1000, 2)

    # Log
    log_event("prediction",
              prediction=prediction,
              label=label,
              confidence=confidence,
              latency_ms=latency_ms,
              model_version=model_version,
              age=patient.age,
              gender=patient.gender)

    return PredictionResponse(
        prediction=prediction,
        prediction_label=label,
        confidence=confidence,
        model_version=str(model_version)
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from CSV upload."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    start = time.time()

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {str(e)}")

    # Validate required columns
    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        raise HTTPException(status_code=400,
                            detail=f"Missing columns: {missing}")

    # Predict
    predictions = model.predict(df[ALL_FEATURES]).tolist()
    probas = model.predict_proba(df[ALL_FEATURES])[:, 1].tolist()

    # Build response
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probas)):
        results.append({
            "index": i,
            "prediction": int(pred),
            "label": "Stroke" if pred == 1 else "No Stroke",
            "confidence": round(float(max(prob, 1 - prob)), 4)
        })

    stroke_count = sum(predictions)
    latency_ms = round((time.time() - start) * 1000, 2)

    log_event("batch_prediction",
              total_records=len(df),
              stroke_count=stroke_count,
              latency_ms=latency_ms,
              model_version=model_version,
              filename=file.filename)

    return BatchPredictionResponse(
        total_records=len(df),
        predictions=results,
        stroke_count=stroke_count,
        no_stroke_count=len(df) - stroke_count,
        model_version=str(model_version)
    )


# ──────────────────────────────────────────────
# Run directly
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host=API_HOST, port=API_PORT, reload=True)