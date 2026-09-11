"""Revenue detection and presentation helpers."""

from __future__ import annotations

import pandas as pd

from .data_loader import detect_revenue_column


def numeric_revenue_series(frame: pd.DataFrame, revenue_column: str | None) -> pd.Series | None:
    """Return a numeric revenue series or None when conversion is impossible."""

    if not revenue_column or revenue_column not in frame.columns:
        return None
    values = pd.to_numeric(frame[revenue_column], errors="coerce")
    if values.notna().sum() == 0:
        return None
    return values


__all__ = ["detect_revenue_column", "numeric_revenue_series"]

