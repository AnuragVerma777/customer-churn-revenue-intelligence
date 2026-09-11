"""Scikit-learn churn models and explainability utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .evaluation import calculate_metrics
from .preprocessing import build_preprocessor, select_feature_columns


@dataclass
class ModelRun:
    """Fitted model plus its evaluation and metadata."""

    name: str
    pipeline: Pipeline
    metrics: dict[str, object]
    feature_columns: list[str]


def _classifier(name: str, random_state: int = 42):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {name}")


def train_and_evaluate(
    frame: pd.DataFrame,
    target_column: str,
    id_column: str | None = None,
    date_column: str | None = None,
    primary_model: str = "Logistic Regression",
    compare_random_forest: bool = False,
    random_state: int = 42,
) -> tuple[ModelRun, dict[str, ModelRun]]:
    """Train one or two models using a stratified holdout and no leakage."""

    y = frame[target_column].astype(int)
    if y.nunique() != 2:
        raise ValueError("Model training requires exactly two target classes.")
    if len(frame) < 12 or y.value_counts().min() < 2:
        raise ValueError("At least 12 rows and at least two examples of each churn class are required.")

    feature_columns = select_feature_columns(frame, target_column, id_column, date_column)
    X = frame[feature_columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    names = [primary_model]
    if compare_random_forest and "Random Forest" not in names:
        names.append("Random Forest")
    if primary_model == "Random Forest" and compare_random_forest and "Logistic Regression" not in names:
        names.append("Logistic Regression")

    runs: dict[str, ModelRun] = {}
    for name in names:
        preprocessor = build_preprocessor(frame, feature_columns)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", _classifier(name, random_state)),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        runs[name] = ModelRun(
            name=name,
            pipeline=pipeline,
            metrics=calculate_metrics(y_test, predictions, probabilities),
            feature_columns=feature_columns,
        )

    return runs[primary_model], runs


def score_customers(model_run: ModelRun, frame: pd.DataFrame) -> np.ndarray:
    """Return churn probabilities for every input customer."""

    return model_run.pipeline.predict_proba(frame[model_run.feature_columns])[:, 1]


def get_feature_importance(model_run: ModelRun, top_n: int = 15) -> pd.DataFrame:
    """Extract coefficient or tree importances after preprocessing."""

    preprocessor = model_run.pipeline.named_steps["preprocessor"]
    classifier = model_run.pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values = classifier.coef_[0]
        importance = np.abs(values)
        direction = np.where(values >= 0, "increases churn likelihood", "reduces churn likelihood")
    else:
        values = classifier.feature_importances_
        importance = np.abs(values)
        direction = np.repeat("contributes to churn prediction", len(values))
    result = pd.DataFrame(
        {
            "feature": [str(name).replace("numeric__", "").replace("categorical__", "") for name in names],
            "importance": importance,
            "direction": direction,
        }
    )
    return result.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

