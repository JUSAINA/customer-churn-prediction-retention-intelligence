"""Streamlit dashboard for churn risk and retention intelligence."""

from pathlib import Path
import sys


# Streamlit executes this file as a script, so the sibling ``src`` directory
# is not automatically importable when launched from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd
import streamlit as st

from churn.config import ROOT
from churn.data import DataValidationError, load_raw
from churn.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES


st.set_page_config(page_title="Churn Retention Intelligence", page_icon="📉", layout="wide")
st.title("Customer Churn Prediction & Retention Intelligence")
st.caption("Production-oriented baseline: validated data, leakage-safe preprocessing, and risk-focused decisions.")

model_path = ROOT / "models" / "churn_model.joblib"
try:
    frame = load_raw()
except (FileNotFoundError, DataValidationError) as error:
    st.warning(str(error))
    st.info("Add the IBM Telco Customer Churn CSV under data/raw, then run: python -m churn.train")
    st.stop()

if not model_path.exists():
    st.warning("No trained model artifact found.")
    st.info("Run: python -m churn.train")
    st.stop()

model = joblib.load(model_path)
churn_rate = float((frame["Churn"] == "Yes").mean())
overview = st.columns(3)
overview[0].metric("Customers", f"{len(frame):,}")
overview[1].metric("Churn rate", f"{churn_rate:.1%}")
overview[2].metric("Retained", f"{(frame['Churn'] == 'No').sum():,}")

tab_overview, tab_predict = st.tabs(["Overview", "Prediction"])
with tab_overview:
    st.subheader("Churn by contract")
    st.bar_chart(frame.groupby("Contract")["Churn"].apply(lambda values: (values == "Yes").mean()))
    st.subheader("Churn by tenure")
    st.line_chart(frame.assign(tenure_bucket=(frame["tenure"] // 6) * 6).groupby("tenure_bucket")["Churn"].apply(lambda values: (values == "Yes").mean()))

with tab_predict:
    st.subheader("Score a customer")
    input_data = {}
    for column in NUMERIC_FEATURES:
        input_data[column] = st.number_input(column, value=float(frame[column].median()))
    for column in CATEGORICAL_FEATURES:
        options = sorted(frame[column].dropna().astype(str).unique())
        input_data[column] = st.selectbox(column, options)
    if st.button("Predict churn risk", type="primary"):
        probability = float(model.predict_proba(pd.DataFrame([input_data]))[0, 1])
        risk = "High" if probability >= 0.70 else "Medium" if probability >= 0.40 else "Low"
        st.metric("Churn probability", f"{probability:.1%}")
        st.success(f"Risk segment: {risk}")
        st.write("Recommended action: prioritize contract review, service issue resolution, and a targeted retention offer." if risk != "Low" else "Recommended action: maintain engagement and monitor for changes.")
