from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load sklearn breast cancer dataset as DataFrame/Series."""
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    target = dataset.target.copy()
    return features, target
