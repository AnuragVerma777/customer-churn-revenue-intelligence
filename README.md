# Customer Churn & Revenue Intelligence

An end-to-end Python and Streamlit project that turns customer data into churn-risk and revenue-exposure insights.

The application lives in [customer-churn-revenue-intelligence](customer-churn-revenue-intelligence/). It accepts customer CSV files, trains a leakage-safe scikit-learn pipeline, scores churn risk, and presents the results in an interactive dashboard.

## Quick start

From the repository root:

```bash
cd customer-churn-revenue-intelligence
pip install -r requirements.txt
streamlit run app.py
```

The app includes a synthetic demo dataset at `data/sample_customer_churn.csv`. Upload it in the sidebar to explore the dashboard. The sample is generated data and does not represent real customers.

## What it does

- Profiles uploaded CSV data and detects likely churn, customer ID, revenue, and date columns.
- Trains and evaluates a Logistic Regression model, with an optional Random Forest comparison.
- Shows customer churn probabilities, configurable risk bands, and revenue at risk.
- Provides segment analysis, retention scenarios, recommendations, and six-month trend forecasts when usable date history is available.
- Exports scored customers and the trained model.

## Documentation

See the [project README](customer-churn-revenue-intelligence/README.md) for dataset guidance, methodology, assumptions and limitations

