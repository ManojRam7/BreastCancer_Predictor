# Runbook

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train_model.py
```

Expected outputs:

- `artifacts/model.joblib`
- `artifacts/metrics.json`

## Test

```bash
pytest -q
```

## Run Application

```bash
streamlit run Streamlit_app.py
```

## Troubleshooting

- If the app reports missing artifacts, run training once before launching Streamlit.
- If dependency resolution fails, recreate the virtual environment and reinstall from `requirements.txt`.
- If slider values seem extreme, use dataset medians (default values) and adjust incrementally.
