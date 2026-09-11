"""Interpretable six-month forecasts using monthly aggregation and linear regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def build_monthly_history(
    frame: pd.DataFrame,
    date_column: str | None,
    target_column: str | None = None,
    revenue_column: str | None = None,
) -> pd.DataFrame | None:
    """Aggregate valid dated rows into monthly customer, churn, and revenue history."""

    if not date_column or date_column not in frame.columns:
        return None
    dated = frame.copy()
    dated["_date"] = pd.to_datetime(dated[date_column], errors="coerce")
    dated = dated.dropna(subset=["_date"])
    if dated.empty:
        return None
    dated["month"] = dated["_date"].dt.to_period("M").dt.to_timestamp()
    grouped = dated.groupby("month", as_index=False).size().rename(columns={"size": "total_customers"})
    if target_column and target_column in dated.columns:
        churn = dated.groupby("month")[target_column].sum().rename("churned_customers")
        grouped = grouped.merge(churn, on="month", how="left")
        grouped["churn_rate"] = grouped["churned_customers"] / grouped["total_customers"]
    if revenue_column and revenue_column in dated.columns:
        values = pd.to_numeric(dated[revenue_column], errors="coerce")
        if values.notna().any():
            dated["_revenue"] = values
            revenue = dated.groupby("month")["_revenue"].sum(min_count=1).rename("revenue")
            grouped = grouped.merge(revenue, on="month", how="left")
    return grouped.sort_values("month").reset_index(drop=True)


def forecast_series(
    history: pd.DataFrame,
    value_column: str,
    periods: int = 6,
) -> pd.DataFrame | None:
    """Forecast a numeric monthly series with month number -> value linear regression."""

    if history is None or value_column not in history.columns:
        return None
    usable = history[["month", value_column]].dropna().copy()
    if len(usable) < 3:
        return None
    usable["month_number"] = np.arange(len(usable), dtype=float)
    model = LinearRegression().fit(usable[["month_number"]], usable[value_column])
    future_numbers = np.arange(len(usable), len(usable) + periods, dtype=float)
    last_month = pd.Timestamp(usable["month"].max())
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    forecast = model.predict(pd.DataFrame({"month_number": future_numbers}))
    if value_column == "churn_rate":
        forecast = np.clip(forecast, 0, 1)
    elif value_column in {"total_customers", "churned_customers", "revenue"}:
        forecast = np.clip(forecast, 0, None)
    return pd.DataFrame({"month": future_months, value_column: forecast, "series_type": "Forecast"})


def combine_history_and_forecast(history: pd.DataFrame, forecast: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Return a chart-ready frame with a historical/forecast label."""

    actual = history[["month", value_column]].dropna().copy()
    actual["series_type"] = "Historical"
    return pd.concat([actual, forecast[["month", value_column, "series_type"]]], ignore_index=True)
