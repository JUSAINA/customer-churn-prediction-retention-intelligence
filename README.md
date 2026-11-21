# Customer Churn Prediction & Retention Intelligence

An advanced, production-oriented portfolio project for identifying customers at risk of churn and turning model output into retention actions.

## Current implementation

- Validated raw-data loading with a clear dataset contract.
- Leakage-safe numeric imputation/scaling and categorical one-hot encoding.
- Stratified train/test split with Logistic Regression and Random Forest candidates.
- Accuracy, precision, recall, F1, ROC-AUC, and PR-AUC evaluation.
- Persisted model artifact and comparison report.
- Streamlit overview and single-customer prediction flow.
- Automated tests for data validation and preprocessing.

## Run locally

1. Put `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set the package path: ` $env:PYTHONPATH = "src"` (PowerShell).
4. Train: `python -m churn.train`.
5. Launch: `streamlit run dashboard/app.py`.
6. Test: `pytest`.

The project intentionally does not claim model performance until the real dataset is supplied and training is executed.
