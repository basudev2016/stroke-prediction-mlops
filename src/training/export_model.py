"""
export_model.py — Export the champion model to a standalone directory.

Usage:
    python src/training/export_model.py

Reads:  MLflow Registry (champion alias)
Writes: models/champion/ (self-contained sklearn model)

This exported model is what gets copied into the Docker container.
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
import mlflow.sklearn

from src.config import (
    MLFLOW_TRACKING_URI, MLFLOW_REGISTRY_MODEL_NAME, MODEL_DIR
)


def main():
    print("=" * 50)
    print("EXPORT CHAMPION MODEL")
    print("=" * 50)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Get champion version
    version_info = client.get_model_version_by_alias(
        name=MLFLOW_REGISTRY_MODEL_NAME,
        alias="champion"
    )
    print(f"Champion: {MLFLOW_REGISTRY_MODEL_NAME} v{version_info.version}")
    print(f"Run ID:   {version_info.run_id}")

    # Load the model
    model_uri = f"models:/{MLFLOW_REGISTRY_MODEL_NAME}@champion"
    model = mlflow.sklearn.load_model(model_uri)

    # Export to models/champion/
    export_path = MODEL_DIR / "champion"

    # Clean previous export
    if export_path.exists():
        shutil.rmtree(export_path)

    export_path.mkdir(parents=True, exist_ok=True)

    mlflow.sklearn.save_model(model, str(export_path))

    print(f"\nExported to: {export_path}")
    print(f"Contents:")
    for f in sorted(export_path.rglob("*")):
        if f.is_file():
            size = f.stat().st_size
            print(f"  {f.relative_to(export_path)} ({size:,} bytes)")

    print(f"\nDone. This directory gets copied into the Docker image.")


if __name__ == "__main__":
    main()