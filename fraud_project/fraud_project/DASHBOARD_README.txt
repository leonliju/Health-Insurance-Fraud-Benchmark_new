# Fraud Research Dashboard

## Install
Run these commands from the project root (the folder containing `data_v3/`, `src/`, and `config/`):

```bash
pip install -r requirements-dashboard.txt
```

## Run
```bash
streamlit run app.py
```

Streamlit will print a local URL (normally http://localhost:8501). Open it in your browser.

## Expected data layout
The dashboard expects the existing project data to remain in these locations:
- `data_v3/claims_fraud_5pct.csv`
- `data_v3/features_5pct.csv`
- `data_v3/labels_5pct.csv`
- `data_v3/metrics_5pct.csv`
- `data_v3/scores_5pct_xgboost_with_district.csv`
- `data_v3/adaptive/5pct/`
- `data_v3/adaptive/5pct_train_adaptation/`

The app uses saved CSV experiment outputs and does not retrain models.

## Notes
- All displayed claims are from the synthetic research benchmark.
- Model scores are not calibrated probabilities.
- The dashboard is intended to support human investigation, not automatic claim denial.
