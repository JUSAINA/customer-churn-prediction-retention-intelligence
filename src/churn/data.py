"""Data access and validation utilities for the churn project."""

from pathlib import Path

import pandas as pd

from churn.config import ROOT, load_config


REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


class DataValidationError(ValueError):
    """Raised when the input dataset does not match the project contract."""


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and lightly normalize a raw Telco Churn dataframe."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise DataValidationError(
            "Dataset is missing required columns: " + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise DataValidationError("Dataset contains no rows.")
    if frame["customerID"].duplicated().any():
        raise DataValidationError("customerID must be unique.")

    normalized = frame.copy()
    normalized["TotalCharges"] = pd.to_numeric(
        normalized["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True),
        errors="coerce",
    )
    normalized["MonthlyCharges"] = pd.to_numeric(normalized["MonthlyCharges"], errors="coerce")
    normalized["tenure"] = pd.to_numeric(normalized["tenure"], errors="coerce")
    normalized["Churn"] = normalized["Churn"].astype("string").str.strip()
    if normalized["Churn"].isna().any():
        raise DataValidationError("Churn contains missing values.")
    return normalized


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load and validate the configured raw CSV dataset."""
    cfg = load_config()
    raw_path = _resolve_path(path or cfg["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. "
            "Place the IBM Telco Churn CSV at the configured path."
        )
    return validate_frame(pd.read_csv(raw_path))
