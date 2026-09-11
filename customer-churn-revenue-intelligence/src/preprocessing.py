"""Leakage-safe scikit-learn preprocessing for mixed customer data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def select_feature_columns(
    frame: pd.DataFrame,
    target_column: str,
    id_column: str | None = None,
    date_column: str | None = None,
    high_cardinality_limit: int = 50,
) -> list[str]:
    """Select model inputs and remove leakage-prone or unusable columns."""

    excluded = {target_column, id_column, date_column}
    selected: list[str] = []
    for column in frame.columns:
        if column in excluded or column is None:
            continue
        if frame[column].nunique(dropna=False) <= 1:
            continue
        if frame[column].dtype == "object" or pd.api.types.is_string_dtype(frame[column]):
            unique_count = frame[column].nunique(dropna=True)
            if unique_count > high_cardinality_limit or unique_count > max(50, int(len(frame) * 0.5)):
                continue
        selected.append(column)
    if not selected:
        raise ValueError("No usable model features remain after removing identifiers, constants, and high-cardinality fields.")
    return selected


def build_preprocessor(frame: pd.DataFrame, feature_columns: Iterable[str]) -> ColumnTransformer:
    """Build a reusable imputation, encoding, and scaling transformer."""

    features = list(feature_columns)
    numeric_columns = [column for column in features if pd.api.types.is_numeric_dtype(frame[column])]
    categorical_columns = [column for column in features if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

