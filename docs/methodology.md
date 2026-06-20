# Methodology

## Problem Definition

Build a binary classifier that predicts whether a tumor sample is malignant (`0`) or benign (`1`) using structured cell-nuclei measurements.

## Dataset

- Source: `sklearn.datasets.load_breast_cancer`
- Samples: 569
- Features: 30 numeric predictors
- Label distribution: mildly imbalanced, handled with stratified split

## Modeling Approach

Pipeline used:

1. `StandardScaler` for feature scaling
2. `LogisticRegression` (`liblinear`, fixed random state, higher max iterations)

Rationale:

- Strong, interpretable baseline for tabular medical features
- Stable optimization and predictable inference latency
- Compatible probability output for downstream thresholding

## Evaluation Protocol

- Holdout strategy: `train_test_split(..., stratify=y)`
- Random state fixed for reproducibility
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1
  - ROC AUC

## Artifact Strategy

Training writes two deployment artifacts:

- `artifacts/model.joblib`: serialized sklearn pipeline
- `artifacts/metrics.json`: machine-readable evaluation summary

This supports reproducible local serving and transparent model reporting.
