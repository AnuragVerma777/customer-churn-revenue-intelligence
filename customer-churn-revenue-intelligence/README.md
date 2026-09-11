# Customer Churn Revenue Intelligence

Customer Churn Revenue Intelligence is a portfolio-ready Python and Streamlit application for turning a customer CSV into an interpretable churn-risk and revenue-exposure dashboard.

The application accepts common customer churn schemas rather than depending on one hardcoded dataset. It detects likely target, identifier, revenue, and date columns, lets the user correct those choices, trains a leakage-safe scikit-learn pipeline, scores every customer, and presents operational insights.

## Business problem

Customer churn is both a retention problem and a revenue problem. A business needs to know which customers are most likely to leave, which segments have elevated churn, how much revenue is exposed, and where a retention team should focus first. This project connects model probabilities to transparent business metrics while keeping scenario assumptions visible.

## Features

- CSV upload with row, column, missing-value, duplicate, preview, and statistical-summary checks.
- Automatic detection of common churn, customer ID, revenue, and historical date columns.
- Manual target-column selection when automatic detection is not confident.
- Binary target normalization for Yes/No, True/False, 1/0, Churn/Not Churn, and similar values.
- Leakage-safe preprocessing with duplicate removal, constant/high-cardinality/ID/date exclusion, imputation, one-hot encoding, and numeric scaling.
- Logistic Regression as the primary interpretable model, with optional Random Forest comparison.
- Accuracy, precision, recall, F1, ROC-AUC, and confusion matrix evaluation.
- Probability-based Low, Medium, and High customer risk bands with configurable thresholds.
- Expected revenue at risk, retention scenario analysis, top at-risk customers, and CSV/model downloads.
- Risk distribution, supported churn-rate-by-category charts, feature influence, and rule-based recommendations.
- Six-month customer, churn-rate, and revenue forecasts when at least three usable historical months exist.
- Synthetic demo data in `data/sample_customer_churn.csv` so the project can be tested immediately.

## ML methodology

1. The selected churn column is converted to a binary target where 1 means churn.
2. Duplicate rows and unusable target rows are removed.
3. Customer ID, selected date, constant columns, and high-cardinality categoricals are excluded from model features. This reduces memorization and date leakage.
4. Numeric fields use median imputation and `StandardScaler`. Categorical fields use most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")`.
5. The `ColumnTransformer` and classifier are combined in one scikit-learn `Pipeline`.
6. The pipeline is fitted only on the stratified training split. The test split is used only for evaluation.
7. `predict_proba()` produces a churn probability for every customer in the cleaned dataset.

Logistic Regression is the default because its coefficients are easy to explain. Random Forest is available for a nonlinear comparison and uses the same preprocessing design. Feature influence is predictive association, not causation.

## Architecture

```text
customer-churn-revenue-intelligence/
├── app.py                         # Streamlit UI and orchestration
├── requirements.txt
├── README.md
├── generate_sample.py
├── data/
│   └── sample_customer_churn.csv  # synthetic/demo data
├── models/                        # optional saved model location
├── src/
│   ├── data_loader.py             # CSV reading, profiling, schema detection
│   ├── preprocessing.py           # feature selection and transformers
│   ├── churn_model.py             # model training, scoring, explainability
│   ├── evaluation.py              # metrics
│   ├── risk_scoring.py            # risk bands and business KPIs
│   ├── forecasting.py             # monthly history and Linear Regression forecasts
│   ├── revenue_analysis.py        # revenue conversion helpers
│   └── insights.py                # supported insights and recommendations
└── tests/
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_forecasting.py
```

## Dataset requirements

The minimum useful dataset has one row per customer and a binary churn outcome. Recommended optional fields are:

- `customer_id`: stable customer identifier.
- `tenure`: customer tenure in months or another numeric unit.
- `contract`, `payment_method`, `internet_service`: categorical customer attributes.
- `monthly_charges`, `monthly_revenue`, `revenue`, `annual_revenue`, or `total_charges`: numeric customer value.
- `customer_since`, `signup_date`, or another historical date: enables monthly forecasting.
- `churn`, `churned`, `exited`, `attrition`, or `is_churn`: target column.

The app handles missing values and can work without revenue, date, or customer ID. It will explain which optional sections are unavailable.

Example CSV format:

```csv
customer_id,tenure,contract,monthly_charges,total_charges,payment_method,customer_since,churn
CUST-0001,12,Month-to-month,72.50,870.00,Electronic check,2024-01-15,Yes
CUST-0002,42,Two year,55.10,2314.20,Credit card,2022-07-03,No
```

The included `sample_customer_churn.csv` is synthetic and should not be treated as real customer or business data.

## Installation and local run

From this project directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate the demo data:

```bash
python generate_sample.py
```

To run automated tests:

```bash
pytest -q
```

## Model evaluation

The default split is an 80/20 stratified train/test split with `random_state=42`. Accuracy is shown for context, but churn decisions should focus on recall, F1, and ROC-AUC because false negatives represent customers the model failed to flag. Metrics are evaluated on the holdout data, not on the training data.

## Risk and revenue calculations

Risk bands are configurable:

- High Risk: churn probability greater than or equal to the high threshold, default 70%.
- Medium Risk: probability greater than or equal to the medium threshold and below high risk, default 40%.
- Low Risk: probability below the medium threshold.

Expected revenue at risk is calculated as:

```text
sum(customer revenue × churn probability)
```

This is a probability-weighted estimate of exposure, not guaranteed lost revenue. The retention scenario applies a user-visible success-rate assumption to high-risk customers only. It is a planning scenario, not a causal estimate of campaign impact.

## Forecasting methodology

When a usable date column has at least three distinct months, rows are aggregated by calendar month into total customers, churned customers, churn rate, and revenue when available. A simple `sklearn.linear_model.LinearRegression` model maps month number to each historical series and extrapolates six future months. Forecast values are clipped to sensible non-negative ranges, with churn rates constrained to 0–100%.

This is an interpretable baseline, not a production time-series model. A customer-since date may describe acquisition cohorts rather than actual churn event dates, so the resulting monthly pattern should be interpreted accordingly. With no usable date history, the app does not fabricate a time-series forecast; it instead keeps the probability-weighted retention scenario visible.

## Business interpretation

The dashboard helps prioritize outreach, compare observed churn across available customer segments, and quantify revenue exposure. Recommendations are deliberately rule-based and separate from ML predictions. Model feature influence indicates association in this dataset and should be checked with domain knowledge, experiments, and fairness review before operational use.

## Limitations

- A single holdout split can make metrics sensitive to sample size and random variation.
- Probability calibration is not separately fitted; probabilities are model estimates, not guarantees.
- The app does not infer causal retention impact.
- A date column does not prove that rows represent churn events. Historical forecasts depend on the meaning of the supplied date.
- Revenue fields with currency symbols or complex text may need cleaning before upload.
- Thresholds should be set using retention-team capacity, intervention cost, and the relative cost of false positives and false negatives.

## Future improvements

- Cross-validation and probability calibration with a time-aware validation strategy.
- Cost-sensitive threshold optimization based on retention economics.
- More robust event-level time-series forecasting when monthly churn event data is available.
- Model monitoring, drift detection, fairness diagnostics, and experiment tracking.
- Role-based deployment, secure storage, and audit logging for sensitive customer data.

