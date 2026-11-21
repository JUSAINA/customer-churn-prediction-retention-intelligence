"""Training entry point for churn models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from churn.config import ROOT, load_config
from churn.data import load_raw
from churn.evaluate import classification_metrics
from churn.preprocessing import build_preprocessor, split_features_target


def train_models() -> pd.DataFrame:
    """Train baseline production candidates and persist the best model by PR-AUC."""
    cfg = load_config()
    frame = load_raw()
    features, target = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=cfg["test_size"],
        random_state=cfg["seed"],
        stratify=target,
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, random_state=cfg["seed"], class_weight="balanced", n_jobs=-1
        ),
    }
    rows: list[dict[str, float | str]] = []
    trained: dict[str, Pipeline] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(x_train, y_train)
        metrics = classification_metrics(y_test, pipeline.predict_proba(x_test)[:, 1])
        rows.append({"model": name, **metrics})
        trained[name] = pipeline

    results = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best_name = str(results.loc[0, "model"])
    model_dir = ROOT / "models"
    report_dir = ROOT / "reports" / "evaluation_results"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained[best_name], model_dir / "churn_model.joblib")
    results.to_csv(report_dir / "model_comparison.csv", index=False)
    metadata = {"model": best_name, "features": list(features.columns), "rows": len(frame)}
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(train_models().to_string(index=False))
