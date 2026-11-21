"""Leakage-safe preprocessing for model training and inference."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Churn"
ID_COLUMNS = ["customerID"]
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and a binary target without leaking identifiers."""
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing modeling columns: {', '.join(sorted(missing))}")
    features = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    target = frame[TARGET].map({"Yes": 1, "No": 0})
    if target.isna().any():
        values = sorted(frame.loc[target.isna(), TARGET].astype(str).unique())
        raise ValueError(f"Churn must contain only Yes/No values; found {values}")
    return features, target.astype("int8")


def build_preprocessor() -> ColumnTransformer:
    """Build a fitted-later transformer with explicit numeric/categorical policy."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return stable names after fitting a preprocessor."""
    names = preprocessor.get_feature_names_out()
    return [str(name) for name in names]
