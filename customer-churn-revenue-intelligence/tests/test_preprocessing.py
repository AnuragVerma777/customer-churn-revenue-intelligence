import pandas as pd

from src.data_loader import normalize_binary_target
from src.preprocessing import build_preprocessor, select_feature_columns


def test_target_normalization_and_feature_exclusions():
    frame = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d"],
            "tenure": [1, 10, 2, 20],
            "contract": ["Month", "Year", "Month", "Year"],
            "churn": ["Yes", "No", "Yes", "No"],
            "constant": [1, 1, 1, 1],
        }
    )
    target, mapping = normalize_binary_target(frame["churn"])
    assert target.tolist() == [1, 0, 1, 0]
    assert mapping["positive"] == "yes"
    features = select_feature_columns(frame, "churn", "customer_id")
    assert "customer_id" not in features
    assert "constant" not in features
    transformer = build_preprocessor(frame, features)
    transformed = transformer.fit_transform(frame[features])
    assert transformed.shape[0] == 4

