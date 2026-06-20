from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import ARTIFACTS_DIR, TRAIN_CONFIG, TrainConfig
from .data import load_dataset
from .modeling import build_pipeline


def train_and_evaluate(output_dir: Path | None = None, config: TrainConfig = TRAIN_CONFIG) -> dict:
    """Train the model, compute metrics, and persist artifacts."""
    features, target = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "config": asdict(config),
    }

    save_dir = output_dir or ARTIFACTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / config.model_filename
    metrics_path = save_dir / config.metrics_filename

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }
