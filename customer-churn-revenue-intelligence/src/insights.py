"""Business insights and rule-based retention recommendations."""

from __future__ import annotations

import pandas as pd


def _money(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "unavailable"
    return f"₹{value:,.0f}"


def churn_by_category(frame: pd.DataFrame, target_column: str, column: str, minimum_group_size: int = 3) -> pd.DataFrame:
    """Return supported churn-rate summaries for a categorical column."""

    data = frame[[column, target_column]].dropna().copy()
    if data.empty:
        return pd.DataFrame(columns=[column, "customers", "churn_rate"])
    summary = data.groupby(column, dropna=False)[target_column].agg(customers="size", churn_rate="mean").reset_index()
    summary = summary[summary["customers"] >= minimum_group_size].copy()
    summary["churn_rate"] *= 100
    return summary.sort_values("churn_rate", ascending=False)


def generate_business_insights(
    scored: pd.DataFrame,
    target_column: str,
    metrics: dict[str, float | int | None],
    categorical_columns: list[str],
) -> list[str]:
    """Generate concise statements only from observed customer-level values."""

    insights = [
        f"{metrics['high_risk_customers']:,} customers ({metrics['high_risk_customers'] / max(metrics['total_customers'], 1) * 100:.1f}%) are currently classified as high risk.",
        f"The observed churn rate is {metrics['churn_rate']:.1f}%; the model's total expected churn is {metrics['expected_churn']:.1f} customers.",
    ]
    if metrics.get("revenue_at_risk") is not None:
        insights.append(
            f"Estimated expected revenue at risk is {_money(float(metrics['revenue_at_risk']))}; this is probability-weighted exposure, not guaranteed lost revenue."
        )
    best_segment: tuple[str, str, float] | None = None
    for column in categorical_columns:
        summary = churn_by_category(scored, target_column, column)
        if summary.empty:
            continue
        row = summary.iloc[0]
        candidate = (column, str(row[column]), float(row["churn_rate"]))
        if best_segment is None or candidate[2] > best_segment[2]:
            best_segment = candidate
    if best_segment:
        insights.append(
            f"The highest observed churn segment is {best_segment[0]} = {best_segment[1]} ({best_segment[2]:.1f}% churn among groups with enough records)."
        )
    if metrics.get("potential_revenue_retained") is not None:
        rate = float(metrics["retention_success_rate"]) * 100
        insights.append(
            f"Under the visible scenario that {rate:.0f}% of high-risk customers are retained, estimated revenue retained is {_money(float(metrics['potential_revenue_retained']))}."
        )
    return insights


def retention_recommendations() -> dict[str, list[str]]:
    """Return intentionally rule-based recommendations, separate from ML output."""

    return {
        "High Risk": [
            "Prioritize retention outreach.",
            "Offer targeted incentives or service recovery.",
            "Investigate contract, support, and service issues.",
        ],
        "Medium Risk": [
            "Monitor engagement and recent service activity.",
            "Send personalized offers.",
            "Consider contract or plan upgrades where appropriate.",
        ],
        "Low Risk": [
            "Maintain engagement and satisfaction.",
            "Identify relevant upsell opportunities.",
        ],
    }

