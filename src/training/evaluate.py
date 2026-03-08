"""
evaluate.py — Evaluate registered model on test set and promote to Production.

Usage:
    python src/training/evaluate.py

Reads:  data/processed/test.csv
        MLflow Model Registry (latest version of stroke-classifier)
Logs:   Evaluation metrics as a new MLflow run
Action: Promotes model to 'Production' alias if F1 >= threshold
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tempfile
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    accuracy_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)

from src.config import (
    TEST_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTRY_MODEL_NAME, ALL_FEATURES, TARGET,
    MODEL_PROMOTION_THRESHOLD
)


def load_test_data():
    """Load the held-out test set."""
    if not TEST_PATH.exists():
        print("ERROR: Test data not found. Run preprocess.py first.")
        sys.exit(1)

    df = pd.read_csv(TEST_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    print(f"Test data: {len(df)} records")
    print(f"Stroke distribution: {y.value_counts().to_dict()}")
    return X, y


def load_registered_model():
    """Load the latest version of the registered model."""
    client = mlflow.tracking.MlflowClient()

    # Get latest version
    versions = client.search_model_versions(f"name='{MLFLOW_REGISTRY_MODEL_NAME}'")
    if not versions:
        print(f"ERROR: No versions found for model '{MLFLOW_REGISTRY_MODEL_NAME}'")
        sys.exit(1)

    latest = max(versions, key=lambda v: int(v.version))
    print(f"\nLoading model: {MLFLOW_REGISTRY_MODEL_NAME} v{latest.version}")
    print(f"Run ID: {latest.run_id}")

    model_uri = f"models:/{MLFLOW_REGISTRY_MODEL_NAME}/{latest.version}"
    model = mlflow.sklearn.load_model(model_uri)

    return model, latest


def evaluate(model, X, y):
    """Run predictions and compute all metrics."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "test_f1": round(f1_score(y, y_pred), 4),
        "test_precision": round(precision_score(y, y_pred), 4),
        "test_recall": round(recall_score(y, y_pred), 4),
        "test_roc_auc": round(roc_auc_score(y, y_proba), 4),
        "test_accuracy": round(accuracy_score(y, y_pred), 4),
    }

    return metrics, y_pred, y_proba


def log_evaluation(metrics, y, y_pred, y_proba, model, model_version_info):
    """Log evaluation results as a new MLflow run."""
    tmp_dir = Path(tempfile.gettempdir())

    with mlflow.start_run(run_name=f"evaluation_v{model_version_info.version}") as run:
        # Log params
        mlflow.log_param("model_name", MLFLOW_REGISTRY_MODEL_NAME)
        mlflow.log_param("model_version", model_version_info.version)
        mlflow.log_param("source_run_id", model_version_info.run_id)
        mlflow.log_param("test_records", len(y))

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Stroke", "Stroke"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"Test Set — Confusion Matrix (v{model_version_info.version})")
        cm_path = tmp_dir / "test_confusion_matrix.png"
        fig.savefig(str(cm_path), bbox_inches="tight", dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        # ROC curve
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y, y_proba, ax=ax)
        ax.set_title(f"Test Set — ROC Curve (v{model_version_info.version})")
        roc_path = tmp_dir / "test_roc_curve.png"
        fig.savefig(str(roc_path), bbox_inches="tight", dpi=100)
        plt.close(fig)
        mlflow.log_artifact(str(roc_path))

        # Classification report as text artifact
        report = classification_report(y, y_pred, target_names=["No Stroke", "Stroke"])
        report_path = tmp_dir / "classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        print(f"\nEvaluation logged as MLflow run: {run.info.run_id}")
        return run.info.run_id


def promote_model(model_version_info, metrics):
    """
    Promote model to 'Production' alias if it meets the threshold.
    Uses MLflow aliases (modern approach) instead of deprecated stages.
    """
    client = mlflow.tracking.MlflowClient()
    f1 = metrics["test_f1"]

    print(f"\n{'─' * 50}")
    print(f"PROMOTION DECISION")
    print(f"  F1 Score:  {f1}")
    print(f"  Threshold: {MODEL_PROMOTION_THRESHOLD}")

    if f1 >= MODEL_PROMOTION_THRESHOLD:
        # Set alias 'champion' on this version
        client.set_registered_model_alias(
            name=MLFLOW_REGISTRY_MODEL_NAME,
            alias="champion",
            version=model_version_info.version
        )
        print(f"  Decision:  ✅ PROMOTED to 'champion' alias")
        print(f"  Model:     {MLFLOW_REGISTRY_MODEL_NAME} v{model_version_info.version}")
        print(f"{'─' * 50}")
        return True
    else:
        print(f"  Decision:  ❌ NOT PROMOTED (below threshold)")
        print(f"  Action:    Improve model or lower threshold in config.py")
        print(f"{'─' * 50}")
        return False


def main():
    print("=" * 60)
    print("STROKE PREDICTION — MODEL EVALUATION")
    print("=" * 60)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Load test data
    X, y = load_test_data()

    # Load registered model
    model, version_info = load_registered_model()

    # Evaluate
    metrics, y_pred, y_proba = evaluate(model, X, y)

    # Print results
    print(f"\n{'─' * 50}")
    print("TEST SET RESULTS")
    print(f"{'─' * 50}")
    for k, v in metrics.items():
        print(f"  {k:<20s}: {v}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["No Stroke", "Stroke"]))

    # Log to MLflow
    log_evaluation(metrics, y, y_pred, y_proba, model, version_info)

    # Promote decision
    promoted = promote_model(version_info, metrics)

    if promoted:
        print(f"\nDone. Model is live as 'champion'.")
        print(f"Load it anywhere with:")
        print(f"  mlflow.sklearn.load_model('models:/{MLFLOW_REGISTRY_MODEL_NAME}@champion')")
    else:
        print(f"\nDone. Model was not promoted. Review results and retrain if needed.")

    print(f"\nView results: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()