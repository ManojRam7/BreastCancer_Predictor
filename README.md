# Breast Cancer Predictor

### Predicting Tumor Diagnosis with a Reproducible End-to-End ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39.0-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#-quality-checks)
[![Status](https://img.shields.io/badge/status-portfolio--ready-0A7E8C)](#)

This project delivers a professional, portfolio-grade machine learning workflow for classifying breast cancer tumors (malignant vs benign) using the Wisconsin dataset.

From model training to UI inference, every step is reproducible, documented, and production-polished for demonstration quality.

## Highlights

- Reproducible training pipeline with persisted artifacts
- Clean modular architecture under src package structure
- Evaluation-first workflow with saved metrics JSON
- Interactive Streamlit dashboard for real-time prediction
- Automated test coverage for training pipeline integrity
- No cloud lock-in or paid external dependency required

## Project Snapshot

| Category | Details |
|---|---|
| Problem Type | Binary classification |
| Domain | Healthcare / Diagnostic ML (educational use) |
| Dataset | sklearn Wisconsin Breast Cancer |
| Model | StandardScaler + LogisticRegression |
| Evaluation | Accuracy, Precision, Recall, F1, ROC AUC |
| Deployment Style | Local Streamlit inference app |

## Model Performance

Latest run metrics from artifacts/metrics.json:

- Accuracy: 0.9825
- Precision: 0.9861
- Recall: 0.9861
- F1 Score: 0.9861
- ROC AUC: 0.9957

## Architecture

```mermaid
flowchart LR
	A[Load Dataset] --> B[Train/Test Split Stratified]
	B --> C[Pipeline: StandardScaler + LogisticRegression]
	C --> D[Evaluate Metrics]
	D --> E[Save Artifacts: model.joblib + metrics.json]
	E --> F[Streamlit App]
	F --> G[Interactive Prediction + Probabilities]
```

## Repository Structure

```text
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
│       ├── modeling.py              # Pipeline definition
│       ├── predict.py               # Artifact loading + defaults
│       └── train.py                 # Train/evaluate/save workflow
├── tests/
│   └── test_training_pipeline.py    # Pipeline smoke test
├── Streamlit_app.py                 # Inference dashboard
└── requirements.txt
```

## Quick Start

1. Create a virtual environment and activate it.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Train the model and generate artifacts.

```bash
python scripts/train_model.py
```

4. Launch the app.

```bash
streamlit run Streamlit_app.py
```

## Quality Checks

Run automated tests:

```bash
pytest -q
```

## Portfolio Value

This project demonstrates practical data science engineering skills that recruiters look for:

- Structured project layout (not notebook-only)
- Reproducible experiments and deterministic config
- Model evaluation with transparent metric reporting
- Usable front-end for stakeholder-friendly inference
- Test-backed reliability and clean documentation

## Documentation

- docs/methodology.md for modeling rationale and evaluation strategy
- docs/runbook.md for operational steps and troubleshooting

## Disclaimer

This is an educational and portfolio project only. It is not a medical device and should not be used for real clinical diagnosis.
