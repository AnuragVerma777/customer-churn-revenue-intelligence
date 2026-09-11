"""Customer risk bands and business KPI calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_risk_scores(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    medium_threshold: float = 0.40,
    high_threshold: float = 0.70,
    revenue_column: str | None = None,
) -> pd.DataFrame:
    """Add probability, risk band, and numeric revenue fields to a copy."""

    if not 0 <= medium_threshold < high_threshold <= 1:
        raise ValueError("Risk thresholds must satisfy 0 <= medium < high <= 1.")
    if len(frame) != len(probabilities):
        raise ValueError("There must be one churn probability per customer.")

    scored = frame.copy()
    scored["churn_probability"] = np.asarray(probabilities, dtype=float)
    scored["risk_level"] = np.select(
        [scored["churn_probability"] >= high_threshold, scored["churn_probability"] >= medium_threshold],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )
    if revenue_column:
        scored["revenue_value"] = pd.to_numeric(scored[revenue_column], errors="coerce")
        scored["expected_revenue_at_risk"] = scored["revenue_value"].fillna(0) * scored["churn_probability"]
    else:
        scored["revenue_value"] = np.nan
        scored["expected_revenue_at_risk"] = np.nan
    return scored


def calculate_business_metrics(
    scored: pd.DataFrame,
    target_column: str,
    revenue_column: str | None = None,
    retention_success_rate: float = 0.50,
) -> dict[str, float | int | None]:
    """Calculate transparent KPIs and an explicit retention scenario."""

    total = len(scored)
    churned = int(scored[target_column].sum())
    high_risk = scored[scored["risk_level"] == "High Risk"]
    at_risk = int(scored["risk_level"].isin(["High Risk", "Medium Risk"]).sum())
    revenue_available = revenue_column is not None and scored["revenue_value"].notna().any()
    expected_revenue_at_risk = float(scored["expected_revenue_at_risk"].sum()) if revenue_available else None
    high_risk_expected_churn = float(high_risk["churn_probability"].sum())
    potential_prevented_churn = high_risk_expected_churn * retention_success_rate
    potential_revenue_retained = (
        float((high_risk["expected_revenue_at_risk"].fillna(0).sum()) * retention_success_rate)
        if revenue_available
        else None
    )
    return {
        "total_customers": total,
        "churned_customers": churned,
        "churn_rate": (churned / total * 100) if total else 0.0,
        "at_risk_customers": at_risk,
        "high_risk_customers": int(len(high_risk)),
        "expected_churn": float(scored["churn_probability"].sum()),
        "revenue_at_risk": expected_revenue_at_risk,
        "retention_success_rate": retention_success_rate,
        "potential_prevented_churn": potential_prevented_churn,
        "potential_revenue_retained": potential_revenue_retained,
    }

