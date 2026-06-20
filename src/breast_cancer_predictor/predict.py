from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import ARTIFACTS_DIR, TRAIN_CONFIG
from .data import load_dataset


class ModelArtifactsNotFoundError(FileNotFoundError):
    """Raised when model artifacts are missing."""


def load_artifacts(artifacts_dir: Path | None = None) -> tuple[object, dict]:
    root = artifacts_dir or ARTIFACTS_DIR
    model_path = root / TRAIN_CONFIG.model_filename
    metrics_path = root / TRAIN_CONFIG.metrics_filename

    if not model_path.exists() or not metrics_path.exists():
        raise ModelArtifactsNotFoundError(
            "Model artifacts were not found. Run scripts/train_model.py first."
        )

    model = joblib.load(model_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return model, metrics


def default_input_row() -> pd.DataFrame:
    """Create a sensible default row based on median feature values."""
    features, _ = load_dataset()
    medians = features.median(numeric_only=True)
    return pd.DataFrame([medians.to_dict()])
