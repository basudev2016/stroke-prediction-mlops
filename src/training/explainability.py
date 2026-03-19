"""
explainability.py — Model Explainability using SHAP and LIME.

Explains individual predictions and global feature importance.

Usage:
    python src/training/explainability.py

Reads:  models/champion/model.pkl
        data/processed/test.csv
Writes: reports/shap_global_importance.png
        reports/shap_waterfall_patient_X.png
        reports/lime_explanation_patient_X.html
        reports/lime_explanation_patient_X.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    ALL_FEATURES, NUMERICAL_FEATURES, BINARY_FEATURES,
    CATEGORICAL_FEATURES, TARGET, MODEL_DIR, TEST_PATH, PROJECT_ROOT
)

REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PKL = MODEL_DIR / "champion" / "model.pkl"


def load_model_and_data():
    """Load the champion model and test data."""
    print("Loading model and data...")
    model = joblib.load(MODEL_PKL)
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET]
    print(f"  Model: {MODEL_PKL}")
    print(f"  Test data: {len(X_test)} records")
    return model, X_test, y_test


def get_preprocessed_data(model, X):
    """
    Get preprocessed (transformed) data from the pipeline.
    Returns transformed array and feature names after one-hot encoding.
    """
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)

    # Build feature names after transformation
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERICAL_FEATURES + BINARY_FEATURES + cat_feature_names

    return X_transformed, feature_names


def get_classifier(model):
    """Extract the raw classifier from the imblearn pipeline."""
    return model.named_steps["classifier"]


def explain_shap_global(model, X_test):
    """
    SHAP Global Feature Importance.
    Shows which features matter most across all predictions.
    """
    print("\n" + "=" * 50)
    print("SHAP — Global Feature Importance")
    print("=" * 50)

    X_transformed, feature_names = get_preprocessed_data(model, X_test)
    classifier = get_classifier(model)

    # Create SHAP explainer for tree model
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classification, shap_values is a list [class_0, class_1]
    # We want class 1 (stroke)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    # Global importance bar plot — manual bar chart using mean absolute SHAP values
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[-15:]  # top 15

    plt.figure(figsize=(10, 8))
    plt.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs_shap[sorted_idx],
        color="#60a5fa"
    )
    plt.xlabel("Mean |SHAP Value|", fontsize=12)
    plt.title("SHAP — Global Feature Importance (Top 15)", fontsize=14, pad=15)
    plt.tight_layout()
    path = REPORTS_DIR / "shap_global_importance.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Beeswarm plot — use Explanation object for proper rendering
    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    explanation_all = shap.Explanation(
        values=shap_vals,
        base_values=np.full(shap_vals.shape[0], base_val),
        data=X_transformed,
        feature_names=feature_names
    )

    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(explanation_all, show=False, max_display=15)
    plt.title("SHAP — Feature Impact on Stroke Prediction", fontsize=14, pad=20)
    plt.tight_layout()
    path2 = REPORTS_DIR / "shap_beeswarm.png"
    plt.savefig(str(path2), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path2}")

    return explainer, shap_vals, feature_names


def explain_shap_patient(model, X_test, explainer, shap_vals, feature_names, patient_idx=0):
    """
    SHAP Waterfall for a single patient.
    Shows why THIS patient got THIS prediction.
    """
    print(f"\nSHAP — Patient #{patient_idx} Explanation")

    X_transformed, _ = get_preprocessed_data(model, X_test)
    patient_data = X_test.iloc[patient_idx]
    prediction = model.predict(X_test.iloc[[patient_idx]])[0]
    proba = model.predict_proba(X_test.iloc[[patient_idx]])[0]

    print(f"  Patient: {patient_data.to_dict()}")
    print(f"  Prediction: {'Stroke' if prediction == 1 else 'No Stroke'}")
    print(f"  Confidence: {max(proba):.1%}")

    # Waterfall plot — use class 1 (stroke) values
    sv = shap_vals[patient_idx]

    # Handle multi-output: if sv has 2 columns, take stroke column
    if sv.ndim > 1:
        sv = sv[:, 1]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    explanation = shap.Explanation(
        values=sv,
        base_values=base_val,
        data=X_transformed[patient_idx],
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.title(f"SHAP Waterfall — Patient #{patient_idx} ({'Stroke' if prediction == 1 else 'No Stroke'})", fontsize=12, pad=20)
    plt.tight_layout()
    path = REPORTS_DIR / f"shap_waterfall_patient_{patient_idx}.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def explain_lime_patient(model, X_test, patient_idx=0):
    """
    LIME explanation for a single patient.
    Creates a local interpretable model around the prediction.
    """
    print(f"\nLIME — Patient #{patient_idx} Explanation")

    X_transformed, feature_names = get_preprocessed_data(model, X_test)
    classifier = get_classifier(model)

    patient_data = X_test.iloc[patient_idx]
    prediction = model.predict(X_test.iloc[[patient_idx]])[0]
    proba = model.predict_proba(X_test.iloc[[patient_idx]])[0]

    print(f"  Patient: {patient_data.to_dict()}")
    print(f"  Prediction: {'Stroke' if prediction == 1 else 'No Stroke'}")
    print(f"  Confidence: {max(proba):.1%}")

    # Create LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_transformed,
        feature_names=feature_names,
        class_names=["No Stroke", "Stroke"],
        mode="classification",
        random_state=42
    )

    # Explain single prediction
    explanation = lime_explainer.explain_instance(
        X_transformed[patient_idx],
        classifier.predict_proba,
        num_features=12,
        top_labels=1
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save as HTML
    html_path = REPORTS_DIR / f"lime_explanation_patient_{patient_idx}.html"
    explanation.save_to_file(str(html_path))
    print(f"  Saved HTML: {html_path}")

    # Save as image
    fig = explanation.as_pyplot_figure(label=1 if prediction == 1 else 0)
    fig.set_size_inches(12, 6)
    plt.title(f"LIME — Patient #{patient_idx} ({'Stroke' if prediction == 1 else 'No Stroke'})", fontsize=12)
    plt.tight_layout()
    img_path = REPORTS_DIR / f"lime_explanation_patient_{patient_idx}.png"
    fig.savefig(str(img_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved PNG:  {img_path}")


def find_interesting_patients(model, X_test, y_test):
    """Find a stroke patient and a no-stroke patient for explanation."""
    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    # Find a high-confidence stroke prediction
    stroke_mask = predictions == 1
    if stroke_mask.any():
        stroke_indices = np.where(stroke_mask)[0]
        stroke_confidences = probas[stroke_indices]
        best_stroke = stroke_indices[np.argmax(stroke_confidences)]
    else:
        best_stroke = 0

    # Find a high-confidence no-stroke prediction
    no_stroke_mask = predictions == 0
    if no_stroke_mask.any():
        no_stroke_indices = np.where(no_stroke_mask)[0]
        no_stroke_confidences = 1 - probas[no_stroke_indices]
        best_no_stroke = no_stroke_indices[np.argmax(no_stroke_confidences)]
    else:
        best_no_stroke = 1

    print(f"\nSelected patients for explanation:")
    print(f"  Stroke patient:    index {best_stroke} (prob={probas[best_stroke]:.3f})")
    print(f"  No-stroke patient: index {best_no_stroke} (prob={probas[best_no_stroke]:.3f})")

    return best_stroke, best_no_stroke


def main():
    print("=" * 60)
    print("MODEL EXPLAINABILITY — SHAP + LIME")
    print("=" * 60)

    # Load
    model, X_test, y_test = load_model_and_data()

    # Find interesting patients
    stroke_idx, no_stroke_idx = find_interesting_patients(model, X_test, y_test)

    # SHAP Global
    explainer, shap_vals, feature_names = explain_shap_global(model, X_test)

    # SHAP Individual — Stroke patient
    explain_shap_patient(model, X_test, explainer, shap_vals, feature_names, patient_idx=stroke_idx)

    # SHAP Individual — No-stroke patient
    explain_shap_patient(model, X_test, explainer, shap_vals, feature_names, patient_idx=no_stroke_idx)

    # LIME — Stroke patient
    explain_lime_patient(model, X_test, patient_idx=stroke_idx)

    # LIME — No-stroke patient
    explain_lime_patient(model, X_test, patient_idx=no_stroke_idx)

    print(f"\n{'=' * 60}")
    print("EXPLAINABILITY COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nAll reports saved to: {REPORTS_DIR}/")
    print(f"\nFiles generated:")
    print(f"  SHAP:")
    print(f"    - shap_global_importance.png  (which features matter most)")
    print(f"    - shap_beeswarm.png           (how features impact predictions)")
    print(f"    - shap_waterfall_patient_X.png (why THIS patient got THIS result)")
    print(f"  LIME:")
    print(f"    - lime_explanation_patient_X.html (interactive explanation)")
    print(f"    - lime_explanation_patient_X.png  (feature contribution chart)")
    print(f"\nOpen reports: start reports\\shap_global_importance.png")


if __name__ == "__main__":
    main()