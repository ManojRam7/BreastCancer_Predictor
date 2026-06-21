from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import sklearn


FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


def _breast_cancer_csv_path() -> Path:
    return Path(sklearn.__file__).resolve().parent / "datasets" / "data" / "breast_cancer.csv"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load breast cancer dataset from sklearn package data files."""
    csv_path = _breast_cancer_csv_path()
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        metadata = next(reader)
        n_features = int(metadata[1])
        rows = [row[: n_features + 1] for row in reader if row]

    values = pd.DataFrame(rows, columns=FEATURE_NAMES + ["target"]).astype(float)

    features = values[FEATURE_NAMES].copy()
    target = values["target"].astype(int).copy()
    return features, target
