from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    random_state: int = 42
    test_size: float = 0.2
    model_filename: str = "model.joblib"
    metrics_filename: str = "metrics.json"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRAIN_CONFIG = TrainConfig()
