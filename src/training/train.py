"""
train.py — Train multiple models with MLflow experiment tracking.

Usage:
    python src/training/train.py

Reads:  data/processed/train.csv
Logs:   MLflow experiments (mlruns/)
Output: Registers best model in MLflow Model Registry
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.config import (
    TRAIN_PATH, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTRY_MODEL_NAME, ALL_FEATURES, TARGET,
    CV_FOLDS, RANDOM_STATE, PRIMARY_METRIC
)
from src.training.preprocess import build_preprocessor


# ──────────────────────────────────────────────
# Model configurations: algorithm + hyperparameter grid
# ──────────────────────────────────────────────
def get_model_configs():
    """
    Returns a list of (name, params) tuples.
    Each will become one MLflow run.
    """
    configs = []

    # Logistic Regression variants
    for C in [0.01, 0.1, 1.0, 10.0]:
        for cw in ["balanced", None]:
            configs.append((
                "LogisticRegression",
                {"classifier__C": C, "classifier__class_weight": cw},
                LogisticRegression(
                    C=C, class_weight=cw,
                    max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs"
                )
            ))

    # Random Forest variants
    for n_est in [100, 200]:
        for depth in [5, 10, None]:
            for cw in ["balanced", None]:
                configs.append((
                    "RandomForest",
                    {
                        "classifier__n_estimators": n_est,
                        "classifier__max_depth": depth,
                        "classifier__class_weight": cw
                    },
                    RandomForestClassifier(
                        n_estimators=n_est, max_depth=depth, class_weight=cw,
                        random_state=RANDOM_STATE, n_jobs=-1
                    )
                ))

    # Gradient Boosting variants
    pos_weight = 16  # approx ratio: 94.2% / 5.8% ≈ 16
    for n_est in [100, 200]:
        for depth in [3, 5]:
            for sample_weight_approach in ["default", "scaled"]:
                configs.append((
                    "GradientBoosting",
                    {
                        "classifier__n_estimators": n_est,
                        "classifier__max_depth": depth,
                        "sample_weight": sample_weight_approach
                    },
                    GradientBoostingClassifier(
                        n_estimators=n_est, max_depth=depth,
                        random_state=RANDOM_STATE
                    )
                ))

    return configs


def build_pipeline(preprocessor, classifier):
    """
    Build imblearn Pipeline: preprocessor → SMOTE → classifier.
    SMOTE only applies during .fit(), skipped during .predict().
    """
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", classifier)
    ])
    return pipeline


def log_confusion_matrix(y_true, y_pred, run_name):
    """Save confusion matrix plot as MLflow artifact."""
    import tempfile

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Stroke", "Stroke"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {run_name}")

    tmp_dir = Path(tempfile.gettempdir())
    path = tmp_dir / f"cm_{run_name.replace(' ', '_')}.png"
    fig.savefig(str(path), bbox_inches="tight", dpi=100)
    plt.close(fig)
    mlflow.log_artifact(str(path))


def run_experiment(train_df):
    """
    Run all model configs with cross-validation, log to MLflow.
    Returns the best run_id and best F1 score.
    """
    X = train_df[ALL_FEATURES]
    y = train_df[TARGET]

    preprocessor = build_preprocessor()
    configs = get_model_configs()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_f1 = -1
    best_run_id = None
    best_run_name = None

    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy"
    }

    print(f"\nRunning {len(configs)} model configurations...")
    print(f"Cross-validation: {CV_FOLDS}-fold stratified")
    print("-" * 60)

    for i, (algo_name, params, classifier) in enumerate(configs, 1):
        run_name = f"{algo_name}_{i:03d}"

        # Build pipeline
        pipeline = build_pipeline(preprocessor, classifier)

        with mlflow.start_run(run_name=run_name) as run:
            # Log algorithm and params
            mlflow.log_param("algorithm", algo_name)
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Cross-validate
            cv_results = cross_validate(
                pipeline, X, y, cv=cv, scoring=scoring,
                return_train_score=False, n_jobs=-1
            )

            # Log mean metrics
            metrics = {}
            for metric_name in scoring:
                mean_val = cv_results[f"test_{metric_name}"].mean()
                std_val = cv_results[f"test_{metric_name}"].std()
                metrics[f"cv_{metric_name}_mean"] = round(mean_val, 4)
                metrics[f"cv_{metric_name}_std"] = round(std_val, 4)
                mlflow.log_metric(f"cv_{metric_name}_mean", round(mean_val, 4))
                mlflow.log_metric(f"cv_{metric_name}_std", round(std_val, 4))

            f1_mean = metrics["cv_f1_mean"]

            # Fit on full training data for artifact logging
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)

            # Log confusion matrix (on training data — for reference)
            log_confusion_matrix(y, y_pred, run_name)

            # Log the full pipeline as model artifact
            mlflow.sklearn.log_model(pipeline, "model")

            # Track best
            if f1_mean > best_f1:
                best_f1 = f1_mean
                best_run_id = run.info.run_id
                best_run_name = run_name

            status = "★ BEST" if run.info.run_id == best_run_id else ""
            print(f"  [{i:2d}/{len(configs)}] {run_name:<30s} F1={f1_mean:.4f}  "
                  f"Prec={metrics['cv_precision_mean']:.4f}  "
                  f"Rec={metrics['cv_recall_mean']:.4f}  "
                  f"AUC={metrics['cv_roc_auc_mean']:.4f}  {status}")

    return best_run_id, best_run_name, best_f1


def register_best_model(best_run_id, best_run_name, best_f1):
    """Register the best model in MLflow Model Registry."""
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(model_uri, MLFLOW_REGISTRY_MODEL_NAME)

    print(f"\n{'=' * 60}")
    print(f"REGISTERED MODEL")
    print(f"  Name:    {MLFLOW_REGISTRY_MODEL_NAME}")
    print(f"  Version: {result.version}")
    print(f"  Run:     {best_run_name}")
    print(f"  F1:      {best_f1:.4f}")
    print(f"  Run ID:  {best_run_id}")
    print(f"{'=' * 60}")

    return result.version


def main():
    print("=" * 60)
    print("STROKE PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # Load training data
    if not TRAIN_PATH.exists():
        print("ERROR: Training data not found. Run preprocess.py first.")
        sys.exit(1)

    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Training data: {len(train_df)} records")
    print(f"Stroke distribution: {train_df[TARGET].value_counts().to_dict()}")

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"\nMLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"MLflow experiment:   {MLFLOW_EXPERIMENT_NAME}")

    # Run experiments
    best_run_id, best_run_name, best_f1 = run_experiment(train_df)

    print(f"\n{'─' * 60}")
    print(f"Best model: {best_run_name} with F1={best_f1:.4f}")
    print(f"{'─' * 60}")

    # Register best model
    version = register_best_model(best_run_id, best_run_name, best_f1)

    print(f"\nDone. Next step: python src/training/evaluate.py")
    print(f"View experiments: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()