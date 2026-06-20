from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from breast_cancer_predictor.train import train_and_evaluate


def test_training_pipeline_outputs_files(tmp_path: Path) -> None:
    result = train_and_evaluate(output_dir=tmp_path)

    model_path = Path(result["model_path"])
    metrics_path = Path(result["metrics_path"])

    assert model_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
