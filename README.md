# Breast Cancer Predictor

Production-style machine learning project for binary tumor classification on the Wisconsin Breast Cancer dataset.

## Why This Project

This repository demonstrates a clean, portfolio-grade data science workflow:

- Reproducible training pipeline with persisted artifacts
- Explicit model evaluation and saved metrics
- Streamlit inference dashboard for interactive prediction
- Basic automated test coverage for pipeline reliability
- Lightweight local setup with no cloud dependency

## Project Structure

```
BreastCancer_Predictor/
├── artifacts/                       # Generated model + metrics
├── docs/
│   ├── methodology.md               # Modeling decisions and trade-offs
│   └── runbook.md                   # Run and troubleshooting guide
├── scripts/
│   └── train_model.py               # Training entrypoint
├── src/
│   └── breast_cancer_predictor/
│       ├── config.py                # Paths and train config
│       ├── data.py                  # Dataset loading
│       ├── modeling.py              # Sklearn pipeline definition
│       ├── predict.py               # Artifact loading + defaults
│       └── train.py                 # Train/evaluate/save workflow
├── tests/
│   └── test_training_pipeline.py    # Pipeline smoke test
├── Streamlit_app.py                 # Interactive prediction UI
└── requirements.txt
```

## Quickstart

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train and persist artifacts:

```bash
python scripts/train_model.py
```

4. Start the app:

```bash
streamlit run Streamlit_app.py
```

## Model Summary

- Task: Binary classification (malignant vs benign)
- Dataset: `sklearn.datasets.load_breast_cancer`
- Baseline model: `StandardScaler + LogisticRegression`
- Split strategy: Stratified holdout with fixed random seed
- Metrics tracked: Accuracy, Precision, Recall, F1, ROC AUC

## Reproducibility Notes

- Training configuration is centralized in `src/breast_cancer_predictor/config.py`.
- Model and metrics are versioned as local artifacts under `artifacts/`.
- Tests are runnable with:

```bash
pytest -q
```

## Disclaimer

This project is for education and portfolio demonstration only. It is not a clinical diagnostic system.
