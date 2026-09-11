import pandas as pd

from src.churn_model import score_customers, train_and_evaluate
from src.data_loader import normalize_binary_target


def test_logistic_model_trains_and_scores():
    frame = pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(20)],
            "tenure": list(range(1, 21)),
            "contract": ["Month-to-month" if i % 2 else "Two year" for i in range(20)],
            "monthly_charges": [40 + i * 2 for i in range(20)],
            "churn": ["Yes" if i % 3 == 0 else "No" for i in range(20)],
        }
    )
    frame["churn"] = normalize_binary_target(frame["churn"])[0].astype(int)
    primary, runs = train_and_evaluate(frame, "churn", id_column="customer_id", primary_model="Logistic Regression")
    probabilities = score_customers(primary, frame)
    assert "Logistic Regression" in runs
    assert len(probabilities) == len(frame)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert set(primary.metrics).issuperset({"recall", "f1", "roc_auc"})

