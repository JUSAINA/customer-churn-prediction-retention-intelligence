import pandas as pd
import pytest

from churn.data import DataValidationError, REQUIRED_COLUMNS, validate_frame
from churn.preprocessing import build_preprocessor, split_features_target


def test_validation_converts_blank_total_charges():
    frame = pd.DataFrame({
        "customerID": ["a"], "gender": ["Female"], "SeniorCitizen": [0], "Partner": ["No"],
        "Dependents": ["No"], "tenure": [1], "PhoneService": ["No"], "MultipleLines": ["No phone service"],
        "InternetService": ["DSL"], "OnlineSecurity": ["No"], "OnlineBackup": ["No"], "DeviceProtection": ["No"],
        "TechSupport": ["No"], "StreamingTV": ["No"], "StreamingMovies": ["No"], "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"], "PaymentMethod": ["Electronic check"], "MonthlyCharges": [29.85],
        "TotalCharges": [" "], "Churn": ["No"],
    })
    result = validate_frame(frame)
    assert pd.isna(result.loc[0, "TotalCharges"])


def test_duplicate_customer_ids_are_rejected():
    frame = pd.DataFrame({column: [None, None] for column in REQUIRED_COLUMNS})
    frame["customerID"] = ["a", "a"]
    with pytest.raises(DataValidationError, match="customerID"):
        validate_frame(frame)


def test_preprocessor_can_transform_mixed_features():
    frame = pd.DataFrame({
        "SeniorCitizen": [0, 1], "tenure": [1, 30], "MonthlyCharges": [30.0, 80.0],
        "TotalCharges": [30.0, 2400.0], "gender": ["Female", "Male"], "Partner": ["No", "Yes"],
        "Dependents": ["No", "Yes"], "PhoneService": ["No", "Yes"], "MultipleLines": ["No phone service", "Yes"],
        "InternetService": ["DSL", "Fiber optic"], "OnlineSecurity": ["No", "Yes"], "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"], "TechSupport": ["No", "Yes"], "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"], "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"], "PaymentMethod": ["Electronic check", "Bank transfer (automatic)"],
        "Churn": ["Yes", "No"],
    })
    features, target = split_features_target(frame)
    transformed = build_preprocessor().fit_transform(features, target)
    assert transformed.shape[0] == 2
    assert target.tolist() == [1, 0]
