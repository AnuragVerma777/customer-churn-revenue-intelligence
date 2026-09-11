"""CSV loading, profiling, and automatic schema detection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from typing import BinaryIO

import numpy as np
import pandas as pd


TARGET_NAMES = {
    "churn",
    "churned",
    "exited",
    "attrition",
    "ischurn",
    "customerchurn",
}

REVENUE_NAME_GROUPS = [
    ("monthly", ["monthlyrevenue", "monthlycharges", "monthlycharge"]),
    ("revenue", ["revenue", "annualrevenue", "annualcharges", "charges"]),
    ("total", ["totalrevenue", "totalcharges", "lifetimevalue", "ltv"]),
]


@dataclass(frozen=True)
class ColumnDetection:
    """Result of detecting a semantic column."""

    column: str | None
    confidence: str
    reason: str


def normalize_column_name(name: str) -> str:
    """Normalize a column name for comparison without changing the original."""

    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def read_customer_csv(source: str | Path | bytes | BinaryIO) -> pd.DataFrame:
    """Read a customer CSV and raise clear, user-facing errors for bad input."""

    try:
        if isinstance(source, (bytes, bytearray)):
            data = BytesIO(source)
            frame = pd.read_csv(data)
        else:
            frame = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The CSV could not be parsed. Check delimiters and quoting.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV encoding could not be read. Save it as UTF-8 and retry.") from exc
    except Exception as exc:
        raise ValueError(f"The CSV could not be read: {exc}") from exc

    if frame.empty:
        raise ValueError("The uploaded CSV contains headers but no customer rows.")
    if frame.shape[1] == 0:
        raise ValueError("The uploaded CSV does not contain any columns.")

    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def profile_dataframe(frame: pd.DataFrame) -> dict[str, object]:
    """Return concise profile tables used by the Streamlit overview."""

    missing = (
        frame.isna()
        .sum()
        .rename("missing_values")
        .to_frame()
        .assign(missing_percent=lambda item: (item["missing_values"] / len(frame) * 100).round(2))
        .sort_values("missing_values", ascending=False)
    )
    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": missing,
        "summary": frame.describe(include="all").transpose(),
    }


def detect_target_column(frame: pd.DataFrame) -> ColumnDetection:
    """Detect a likely churn target from common names and binary cardinality."""

    normalized = {column: normalize_column_name(column) for column in frame.columns}
    exact = [column for column, name in normalized.items() if name in TARGET_NAMES]
    if exact:
        return ColumnDetection(exact[0], "high", "The column name matches a standard churn target name.")

    likely = [
        column
        for column, name in normalized.items()
        if any(token in name for token in ("churn", "attrition", "exited"))
    ]
    if likely:
        return ColumnDetection(likely[0], "high", "The column name contains a churn-related keyword.")

    binary_candidates = [
        column for column in frame.columns if frame[column].dropna().nunique() == 2
    ]
    if len(binary_candidates) == 1:
        return ColumnDetection(
            binary_candidates[0],
            "medium",
            "The column is the only binary column; please confirm it is the churn target.",
        )

    return ColumnDetection(None, "low", "No confident churn target was detected.")


def detect_revenue_column(frame: pd.DataFrame) -> ColumnDetection:
    """Find a likely revenue/value column, preferring monthly recurring value."""

    normalized = {column: normalize_column_name(column) for column in frame.columns}
    for group_name, names in REVENUE_NAME_GROUPS:
        matches = [column for column, name in normalized.items() if name in names]
        numeric_matches = [column for column in matches if pd.to_numeric(frame[column], errors="coerce").notna().sum() > 0]
        if numeric_matches:
            return ColumnDetection(numeric_matches[0], "high", f"Matched a {group_name} revenue/value column.")

    fuzzy = [
        column
        for column, name in normalized.items()
        if any(token in name for token in ("revenue", "charge", "income", "value"))
        and pd.to_numeric(frame[column], errors="coerce").notna().sum() > 0
    ]
    if fuzzy:
        return ColumnDetection(fuzzy[0], "medium", "Matched a numeric column with a revenue-related name.")

    return ColumnDetection(None, "low", "No numeric revenue/value column was detected.")


def detect_date_column(frame: pd.DataFrame) -> ColumnDetection:
    """Find a date column that contains at least three distinct calendar months."""

    normalized = {column: normalize_column_name(column) for column in frame.columns}
    candidates = sorted(
        frame.columns,
        key=lambda column: 0 if any(token in normalized[column] for token in ("date", "since", "signup", "join", "start", "time")) else 1,
    )
    for column in candidates:
        parsed = pd.to_datetime(frame[column], errors="coerce")
        valid = parsed.dropna()
        if len(valid) >= max(3, int(len(frame) * 0.6)) and valid.dt.to_period("M").nunique() >= 3:
            confidence = "high" if any(token in normalized[column] for token in ("date", "since", "signup", "join")) else "medium"
            return ColumnDetection(column, confidence, "Contains enough valid dates across multiple months.")

    return ColumnDetection(None, "low", "No usable historical date column was detected.")


def detect_id_column(frame: pd.DataFrame) -> ColumnDetection:
    """Find a customer identifier without assuming one fixed dataset schema."""

    normalized = {column: normalize_column_name(column) for column in frame.columns}
    named = [
        column for column, name in normalized.items()
        if name in {"id", "customerid", "customeridentifier", "accountid", "userid", "uuid"}
        or name.endswith("customerid")
    ]
    if named:
        return ColumnDetection(named[0], "high", "Matched a common customer identifier name.")

    for column in frame.columns:
        if frame[column].nunique(dropna=True) == len(frame) and frame[column].dtype == "object":
            return ColumnDetection(column, "medium", "The column contains unique text values and may be an identifier.")

    return ColumnDetection(None, "low", "No customer identifier was detected.")


POSITIVE_LABELS = {"yes", "true", "1", "1.0", "y", "churn", "churned", "exited", "attrited", "positive"}
NEGATIVE_LABELS = {"no", "false", "0", "0.0", "n", "stay", "stayed", "active", "retained", "notchurn", "negative"}


def normalize_binary_target(series: pd.Series) -> tuple[pd.Series, dict[str, object]]:
    """Convert common churn encodings to 0/1 and document the mapping used."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64"), {"positive": True, "negative": False, "method": "boolean"}

    numeric = pd.to_numeric(series, errors="coerce")
    non_null = series.dropna()
    if len(non_null) > 0 and numeric.loc[non_null.index].notna().all() and set(numeric.dropna().unique()).issubset({0, 1}):
        return numeric.astype("Int64"), {"positive": 1, "negative": 0, "method": "numeric 0/1"}

    clean = series.astype("string").str.strip().str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    unique = [value for value in clean.dropna().unique().tolist()]
    if len(unique) != 2:
        raise ValueError("The churn target must contain exactly two classes after removing missing values.")

    positive = next((value for value in unique if value in POSITIVE_LABELS), None)
    negative = next((value for value in unique if value in NEGATIVE_LABELS and value != positive), None)
    method = "semantic labels"
    if positive is None:
        keyword = next((value for value in unique if any(token in value for token in ("churn", "exit", "attrit"))), None)
        positive = keyword
    if positive is None:
        # Conservative fallback for arbitrary two-label data: treat the less frequent class as churn.
        counts = clean.value_counts()
        positive = counts.idxmin()
        method = "minority class fallback; please validate the target mapping"
    if negative is None:
        negative = next(value for value in unique if value != positive)

    encoded = clean.map({negative: 0, positive: 1}).astype("Int64")
    return encoded, {"positive": positive, "negative": negative, "method": method}


def remove_duplicates_and_empty_targets(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, int, int]:
    """Remove duplicate rows and records with missing target values."""

    deduplicated = frame.drop_duplicates().copy()
    duplicate_count = len(frame) - len(deduplicated)
    missing_target_count = int(deduplicated[target_column].isna().sum())
    cleaned = deduplicated.dropna(subset=[target_column]).reset_index(drop=True)
    return cleaned, duplicate_count, missing_target_count

