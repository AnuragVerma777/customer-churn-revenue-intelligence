import pandas as pd

from src.forecasting import build_monthly_history, forecast_series


def test_monthly_forecast_returns_six_future_periods():
    frame = pd.DataFrame(
        {
            "customer_since": pd.date_range("2024-01-01", periods=24, freq="MS"),
            "churn": [int(index % 4 == 0) for index in range(24)],
            "revenue": [100 + index for index in range(24)],
        }
    )
    history = build_monthly_history(frame, "customer_since", "churn", "revenue")
    assert history is not None
    forecast = forecast_series(history, "churn_rate")
    assert forecast is not None
    assert len(forecast) == 6
    assert forecast["series_type"].eq("Forecast").all()

