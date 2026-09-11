"""Streamlit application for Customer Churn Revenue Intelligence."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.churn_model import get_feature_importance, score_customers, train_and_evaluate
from src.data_loader import (
    detect_date_column,
    detect_id_column,
    detect_revenue_column,
    detect_target_column,
    normalize_binary_target,
    profile_dataframe,
    read_customer_csv,
    remove_duplicates_and_empty_targets,
)
from src.forecasting import build_monthly_history, combine_history_and_forecast, forecast_series
from src.insights import churn_by_category, generate_business_insights, retention_recommendations
from src.risk_scoring import add_risk_scores, calculate_business_metrics


st.set_page_config(
    page_title="Customer Churn Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
main {
    background-color: #f7f9fc;
}

[data-testid="stMetric"] {
    background: #1E293B;
    border: 1px solid #334155;
    padding: 12px;
    border-radius: 10px;
}

[data-testid="stMetricLabel"] {
    color: #CBD5E1 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

[data-testid="stMetricDelta"] {
    color: #94A3B8 !important;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


def money(value: float | int | None) -> str:
    """Format a numeric value for an Indian-rupee business dashboard."""

    if value is None or pd.isna(value):
        return "Unavailable"
    return f"₹{float(value):,.0f}"


def make_risk_chart(scored: pd.DataFrame) -> plt.Figure:
    counts = scored["risk_level"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
    colors = ["#42a5a5", "#ffb74d", "#ef5350"]
    figure, axis = plt.subplots(figsize=(5.4, 4.1))
    axis.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=colors, textprops={"fontsize": 10})
    axis.set_title("Customer Risk Distribution", fontweight="bold")
    return figure


def make_category_chart(summary: pd.DataFrame, column: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.5, 3.6))
    axis.bar(summary[column].astype(str), summary["churn_rate"], color="#4568dc")
    axis.set_title(f"Churn Rate by {column.replace('_', ' ').title()}", fontweight="bold")
    axis.set_ylabel("Churn rate (%)")
    axis.set_ylim(0, max(100, float(summary["churn_rate"].max()) * 1.2))
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return figure


def make_forecast_chart(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    value_column: str,
    title: str,
    percent: bool = False,
    currency: bool = False,
) -> plt.Figure:
    combined = combine_history_and_forecast(history, forecast, value_column)
    figure, axis = plt.subplots(figsize=(8, 3.9))
    actual = combined[combined["series_type"] == "Historical"]
    future = combined[combined["series_type"] == "Forecast"]
    axis.plot(actual["month"], actual[value_column], marker="o", color="#4568dc", label="Historical")
    axis.plot(future["month"], future[value_column], marker="o", linestyle="--", color="#f28e2b", label="Forecast")
    axis.axvline(actual["month"].max(), color="#9aa4b2", linestyle=":", linewidth=1.5)
    axis.set_title(title, fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend()
    if percent:
        axis.set_ylabel("Churn rate (%)")
        axis.set_ylim(0, max(100, float(combined[value_column].max()) * 1.2))
        axis.yaxis.set_major_formatter(lambda value, _: f"{value * 100:.0f}%")
    elif currency:
        axis.set_ylabel("Revenue")
        axis.yaxis.set_major_formatter(lambda value, _: f"₹{value:,.0f}")
    else:
        axis.set_ylabel("Customers")
    figure.tight_layout()
    return figure


def display_overview(frame: pd.DataFrame) -> None:
    profile = profile_dataframe(frame)
    columns = st.columns(4)
    columns[0].metric("Rows / Customers", f"{profile['rows']:,}")
    columns[1].metric("Columns", f"{profile['columns']:,}")
    columns[2].metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")
    columns[3].metric("Missing Cells", f"{int(frame.isna().sum().sum()):,}")
    with st.expander("Dataset profile and statistical summary", expanded=False):
        st.subheader("Missing values by column")
        st.dataframe(profile["missing_values"], use_container_width=True)
        st.subheader("Basic statistical summary")
        st.dataframe(profile["summary"], use_container_width=True)
        st.subheader("Dataset preview")
        st.dataframe(frame.head(10), use_container_width=True)


def display_model_results(primary_run, all_runs: dict[str, object]) -> None:
    st.subheader("Model performance")
    metric_rows = []
    for name, run in all_runs.items():
        metrics = run.metrics
        metric_rows.append(
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
            }
        )
    metric_frame = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(metric_frame.style.format("{:.3f}"), use_container_width=True)
    st.caption("Recall, F1, and ROC-AUC are emphasized because missed churners can be costly.")
    cm = primary_run.metrics["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(4.5, 3.4))
    image = axis.imshow(cm, cmap="Blues")
    axis.set_title(f"Confusion Matrix — {primary_run.name}", fontweight="bold")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1], ["Stay", "Churn"])
    axis.set_yticks([0, 1], ["Stay", "Churn"])
    for row in range(2):
        for column in range(2):
            axis.text(column, row, int(cm[row, column]), ha="center", va="center", color="black")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    st.pyplot(figure, use_container_width=False)


def display_forecasts(frame: pd.DataFrame, date_column: str | None, target_column: str, revenue_column: str | None) -> None:
    st.subheader("Six-month forecasts")
    if date_column is None:
        st.info("Six-month time-series forecasting is unavailable because the uploaded dataset does not contain a usable historical date column.")
        st.caption("A customer-level risk scenario is still available in the KPI and retention sections; it is not a time-series forecast.")
        return
    history = build_monthly_history(frame, date_column, target_column, revenue_column)
    if history is None or len(history) < 3:
        st.info("Six-month time-series forecasting is unavailable because the dataset does not contain at least three usable historical months.")
        st.caption("A customer-level risk scenario is still available in the KPI and retention sections; it is not a time-series forecast.")
        return
    st.caption(f"Historical monthly aggregates use `{date_column}`. Linear Regression extrapolates the observed trend for six future months.")
    customer_forecast = forecast_series(history, "total_customers")
    churn_forecast = forecast_series(history, "churn_rate")
    revenue_forecast = forecast_series(history, "revenue")
    if customer_forecast is not None:
        st.markdown("**6-Month Customer Forecast**")
        st.pyplot(make_forecast_chart(history, customer_forecast, "total_customers", "6-Month Customer Forecast"), use_container_width=True)
    else:
        st.info("Customer-count forecasting is unavailable because there are too few monthly observations.")
    if churn_forecast is not None:
        st.markdown("**6-Month Churn Forecast**")
        st.pyplot(make_forecast_chart(history, churn_forecast, "churn_rate", "6-Month Churn Forecast", percent=True), use_container_width=True)
    else:
        st.info("Churn-rate forecasting is unavailable because monthly churn observations could not be built.")
    if revenue_column and revenue_forecast is not None:
        st.markdown("**6-Month Revenue Forecast**")
        st.pyplot(make_forecast_chart(history, revenue_forecast, "revenue", "6-Month Revenue Forecast", currency=True), use_container_width=True)
    else:
        st.info("Revenue forecasting is unavailable because the uploaded dataset does not contain sufficient revenue/time-series information.")


def run_analysis(
    raw_frame: pd.DataFrame,
    target_column: str,
    id_column: str | None,
    revenue_column: str | None,
    date_column: str | None,
    medium_threshold: float,
    high_threshold: float,
    primary_model: str,
    compare_random_forest: bool,
    retention_success_rate: float,
) -> dict[str, object]:
    """Run the full analysis pipeline and return render-ready objects."""

    clean, duplicate_count, missing_target_count = remove_duplicates_and_empty_targets(raw_frame, target_column)
    encoded_target, mapping = normalize_binary_target(clean[target_column])
    valid_target = encoded_target.notna()
    dropped_unmapped = int((~valid_target).sum())
    clean = clean.loc[valid_target].reset_index(drop=True)
    clean[target_column] = encoded_target.loc[valid_target].astype(int).to_numpy()
    if clean[target_column].nunique() != 2:
        raise ValueError("The selected target has only one usable class after cleaning. Choose a binary churn column with both classes.")
    primary_run, all_runs = train_and_evaluate(
        clean,
        target_column,
        id_column=id_column,
        date_column=date_column,
        primary_model=primary_model,
        compare_random_forest=compare_random_forest,
    )
    probabilities = score_customers(primary_run, clean)
    scored = add_risk_scores(clean, probabilities, medium_threshold, high_threshold, revenue_column)
    metrics = calculate_business_metrics(scored, target_column, revenue_column, retention_success_rate)
    categorical_columns = [
        column
        for column in clean.columns
        if column not in {target_column, id_column, date_column}
        and (clean[column].dtype == "object" or pd.api.types.is_string_dtype(clean[column]))
        and clean[column].nunique(dropna=True) <= 12
    ]
    model_bundle = {
        "model": primary_run.pipeline,
        "target_column": target_column,
        "feature_columns": primary_run.feature_columns,
        "id_column": id_column,
        "date_column": date_column,
        "revenue_column": revenue_column,
        "risk_thresholds": {"medium": medium_threshold, "high": high_threshold},
    }
    return {
        "raw_frame": raw_frame,
        "clean_frame": clean,
        "scored": scored,
        "primary_run": primary_run,
        "all_runs": all_runs,
        "metrics": metrics,
        "target_column": target_column,
        "id_column": id_column,
        "revenue_column": revenue_column,
        "date_column": date_column,
        "categorical_columns": categorical_columns,
        "mapping": mapping,
        "duplicate_count": duplicate_count,
        "missing_target_count": missing_target_count + dropped_unmapped,
        "model_bundle": model_bundle,
    }


def main() -> None:
    st.title("Customer Churn Revenue Intelligence")
    st.caption("Upload customer data to estimate churn risk, quantify revenue exposure, and explore six-month trends.")

    with st.sidebar:
        st.header("Analysis setup")
        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
        sample_path = PROJECT_ROOT / "data" / "sample_customer_churn.csv"
        if sample_path.exists():
            st.download_button(
                "Download synthetic demo CSV",
                data=sample_path.read_bytes(),
                file_name="sample_customer_churn.csv",
                mime="text/csv",
            )

    if uploaded_file is None:
        st.info("Upload a CSV from the sidebar to begin. The included synthetic demo CSV is ready for testing.")
        st.markdown("""
        **Expected data:** one row per customer, a binary churn outcome, and optional customer ID, revenue, and date fields.

        The app detects common names such as `churn`, `customer_id`, `monthly_charges`, and `customer_since`, but every important field can be selected manually.
        """)
        return

    file_bytes = uploaded_file.getvalue()
    analysis_key = f"{uploaded_file.name}:{len(file_bytes)}"
    try:
        frame = read_customer_csv(file_bytes)
    except ValueError as error:
        st.error(str(error))
        return

    if st.session_state.get("analysis_key") != analysis_key:
        st.session_state.pop("analysis_result", None)
        st.session_state["analysis_key"] = analysis_key

    display_overview(frame)
    target_detection = detect_target_column(frame)
    revenue_detection = detect_revenue_column(frame)
    date_detection = detect_date_column(frame)
    id_detection = detect_id_column(frame)

    with st.sidebar:
        st.divider()
        st.header("Column selection")
        target_options = ["Select target column"] + list(frame.columns)
        target_default = target_options.index(target_detection.column) if target_detection.column in target_options else 0
        target_column = st.selectbox("Churn / target column", target_options, index=target_default)
        if target_detection.column:
            st.caption(f"Auto-detected: `{target_detection.column}` ({target_detection.reason})")
        revenue_options = ["None"] + list(frame.columns)
        revenue_default = revenue_options.index(revenue_detection.column) if revenue_detection.column in revenue_options else 0
        revenue_choice = st.selectbox("Revenue / value column", revenue_options, index=revenue_default)
        date_options = ["None"] + list(frame.columns)
        date_default = date_options.index(date_detection.column) if date_detection.column in date_options else 0
        date_choice = st.selectbox("Historical date column", date_options, index=date_default)
        id_options = ["None"] + list(frame.columns)
        id_default = id_options.index(id_detection.column) if id_detection.column in id_options else 0
        id_choice = st.selectbox("Customer ID column", id_options, index=id_default)
        st.divider()
        st.header("Model settings")
        primary_model = st.selectbox("Primary model", ["Logistic Regression", "Random Forest"])
        compare_random_forest = st.checkbox("Also compare Random Forest", value=True)
        st.caption("Logistic Regression is the default interpretable baseline; Random Forest is optional.")
        st.header("Risk thresholds")
        medium_threshold = st.slider("Medium risk starts at", 0.0, 0.95, 0.40, 0.05)
        high_threshold = st.slider("High risk starts at", medium_threshold + 0.05, 1.0, max(0.70, medium_threshold + 0.05), 0.05)
        retention_success_rate = st.slider("High-risk retention scenario", 0.0, 1.0, 0.50, 0.05)
        st.caption("The scenario assumes this share of high-risk customers can be retained; it is not a model prediction.")
        run_button = st.button("Run analysis", type="primary", use_container_width=True)

    if run_button:
        if target_column == "Select target column":
            st.error("Select a churn target column before running the analysis.")
        else:
            with st.spinner("Cleaning data, training model, and calculating business metrics..."):
                try:
                    st.session_state["analysis_result"] = run_analysis(
                        frame,
                        target_column,
                        None if id_choice == "None" else id_choice,
                        None if revenue_choice == "None" else revenue_choice,
                        None if date_choice == "None" else date_choice,
                        medium_threshold,
                        high_threshold,
                        primary_model,
                        compare_random_forest,
                        retention_success_rate,
                    )
                except (ValueError, TypeError, KeyError) as error:
                    st.error(f"Analysis could not be completed: {error}")
                except Exception as error:  # pragma: no cover - final UI safety net
                    st.error(f"The model encountered an unexpected training error: {error}")

    result = st.session_state.get("analysis_result")
    if not result:
        st.warning("Choose the columns and click **Run analysis** to generate the dashboard.")
        return

    scored = result["scored"]
    metrics = result["metrics"]
    st.success(
        f"Analysis complete. Target mapping: positive = `{result['mapping']['positive']}`, negative = `{result['mapping']['negative']}` ({result['mapping']['method']})."
    )
    if result["duplicate_count"] or result["missing_target_count"]:
        st.info(f"Data preparation removed {result['duplicate_count']} duplicate rows and {result['missing_target_count']} rows with unusable target values.")

    st.subheader("Executive dashboard")
    kpis = st.columns(6)
    kpis[0].metric("Total Customers", f"{metrics['total_customers']:,}")
    kpis[1].metric("Churn Rate", f"{metrics['churn_rate']:.1f}%")
    kpis[2].metric("At-Risk Customers", f"{metrics['at_risk_customers']:,}")
    kpis[3].metric("High-Risk Customers", f"{metrics['high_risk_customers']:,}")
    kpis[4].metric("Revenue at Risk", money(metrics["revenue_at_risk"]))
    kpis[5].metric("Expected Churn", f"{metrics['expected_churn']:.1f}")
    st.caption("Revenue at Risk is estimated expected revenue at risk = revenue × churn probability; it is not guaranteed lost revenue.")

    risk_left, risk_right = st.columns([1, 1.5])
    with risk_left:
        st.pyplot(make_risk_chart(scored), use_container_width=True)
    with risk_right:
        st.subheader("Retention scenario")
        st.write(
            f"If {metrics['retention_success_rate'] * 100:.0f}% of high-risk customers were successfully retained, the model estimates {metrics['potential_prevented_churn']:.1f} expected churn events could be prevented."
        )
        st.metric("Potential revenue retained", money(metrics["potential_revenue_retained"]))
        st.caption("This is an assumption-based scenario using high-risk customers only; it does not claim a causal treatment effect.")

    st.subheader("Top At-Risk Customers")
    preferred_columns = [
        result["id_column"],
        "churn_probability",
        "risk_level",
        result["revenue_column"],
        "tenure",
        "contract",
        "monthly_charges",
        "monthlycharges",
    ]
    display_columns = []
    for column in preferred_columns:
        if column and column in scored.columns and column not in display_columns:
            display_columns.append(column)
    for column in scored.columns:
        if column not in display_columns and column not in {result["target_column"], "revenue_value", "expected_revenue_at_risk"}:
            display_columns.append(column)
        if len(display_columns) >= 12:
            break
    top_customers = scored.sort_values("churn_probability", ascending=False).head(20)
    st.dataframe(top_customers[display_columns], use_container_width=True)
    csv_bytes = scored.to_csv(index=False).encode("utf-8")
    model_buffer = BytesIO()
    joblib.dump(result["model_bundle"], model_buffer)
    download_left, download_right = st.columns(2)
    with download_left:
        st.download_button("Download scored customer CSV", csv_bytes, "scored_customers.csv", "text/csv", use_container_width=True)
    with download_right:
        st.download_button("Download trained model", model_buffer.getvalue(), "churn_model.joblib", "application/octet-stream", use_container_width=True)

    st.subheader("Churn analysis")
    analysis_columns = [column for column in result["categorical_columns"] if column in scored.columns]
    if analysis_columns:
        chart_columns = st.columns(2)
        for index, column in enumerate(analysis_columns[:4]):
            summary = churn_by_category(scored, result["target_column"], column)
            if summary.empty:
                continue
            with chart_columns[index % 2]:
                st.pyplot(make_category_chart(summary, column), use_container_width=True)
    else:
        st.info("No low-cardinality categorical columns were available for supported churn pattern charts.")

    display_model_results(result["primary_run"], result["all_runs"])
    st.subheader("What drives churn?")
    importance = get_feature_importance(result["primary_run"])
    importance_display = importance.sort_values("importance", ascending=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.barh(importance_display["feature"], importance_display["importance"], color="#4c78a8")
    axis.set_xlabel("Absolute model influence")
    axis.set_title("Top features associated with churn predictions", fontweight="bold")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    st.dataframe(importance, use_container_width=True)
    st.caption("Feature influence is predictive association, not proof that a feature causes churn.")

    display_forecasts(result["clean_frame"], result["date_column"], result["target_column"], result["revenue_column"])

    st.subheader("Business insights")
    insights = generate_business_insights(scored, result["target_column"], metrics, result["categorical_columns"])
    for insight in insights:
        st.markdown(f"- {insight}")
    st.subheader("Recommended actions")
    recommendations = retention_recommendations()
    for level, actions in recommendations.items():
        with st.expander(level, expanded=level == "High Risk"):
            for action in actions:
                st.markdown(f"- {action}")
    st.caption("Recommendations are rule-based playbook suggestions and are intentionally separate from model predictions.")


if __name__ == "__main__":
    main()
