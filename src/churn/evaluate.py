"""Model evaluation helpers with business-relevant classification metrics."""

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


def classification_metrics(y_true: Any, probabilities: Any, threshold: float = 0.5) -> dict[str, float]:
    """Calculate thresholded metrics plus ROC-AUC and PR-AUC."""
    y_true_array = np.asarray(y_true)
    proba_array = np.asarray(probabilities)
    predictions = (proba_array >= threshold).astype(int)
    report = classification_report(y_true_array, predictions, output_dict=True, zero_division=0)
    return {
        "threshold": float(threshold),
        "accuracy": float(report["accuracy"]),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "roc_auc": float(roc_auc_score(y_true_array, proba_array)),
        "pr_auc": float(average_precision_score(y_true_array, proba_array)),
    }
