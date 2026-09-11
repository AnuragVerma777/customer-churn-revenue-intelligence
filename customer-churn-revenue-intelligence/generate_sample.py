"""Generate the deterministic synthetic demo dataset used by the Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    rows = []
    start = pd.Timestamp("2023-01-01")
    contracts = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Bank transfer", "Credit card", "Mailed check"]
    internet_services = ["Fiber optic", "DSL", "No"]
    for index in range(360):
        customer_since = start + pd.DateOffset(months=index % 36) + pd.Timedelta(days=int(rng.integers(0, 27)))
        tenure = int(rng.integers(1, 73))
        contract = rng.choice(contracts, p=[0.52, 0.28, 0.20])
        payment = rng.choice(payment_methods, p=[0.40, 0.25, 0.22, 0.13])
        internet = rng.choice(internet_services, p=[0.48, 0.35, 0.17])
        monthly_charges = round(float(rng.normal(78 if internet == "Fiber optic" else 58, 15)), 2)
        monthly_charges = max(20, monthly_charges)
        support_calls = int(rng.poisson(1.6 if contract == "Month-to-month" else 0.9))
        logit = -1.35 + (contract == "Month-to-month") * 1.05 + (internet == "Fiber optic") * 0.35
        logit += (payment == "Electronic check") * 0.35 + support_calls * 0.16 - tenure * 0.018
        probability = 1 / (1 + np.exp(-logit))
        churn = int(rng.random() < probability)
        total_charges = round(monthly_charges * tenure * rng.uniform(0.92, 1.08), 2)
        rows.append(
            {
                "customer_id": f"CUST-{index + 1:04d}",
                "tenure": tenure,
                "contract": contract,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "payment_method": payment,
                "internet_service": internet,
                "support_calls": support_calls,
                "customer_since": customer_since.strftime("%Y-%m-%d"),
                "churn": "Yes" if churn else "No",
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[[17, 143], "support_calls"] = np.nan
    frame.loc[[22], "payment_method"] = np.nan
    output = Path(__file__).resolve().parent / "data" / "sample_customer_churn.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} synthetic customers to {output}")


if __name__ == "__main__":
    main()

