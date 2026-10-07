"""Dividend and corporate-action accounting for Graham backtests."""

from typing import Callable, List

import numpy as np
import pandas as pd


def corporate_actions(
    code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    load_dividend: Callable,
) -> List[tuple]:
    """Return cash and stock actions after ``start`` through ``end``."""
    dividend = load_dividend(code)
    if dividend is None or dividend.empty or "除权日" not in dividend.columns:
        return []
    frame = dividend.copy()
    frame["action_date"] = pd.to_datetime(frame["除权日"], errors="coerce")
    frame = frame[
        (frame["action_date"] > start) & (frame["action_date"] <= end)
    ].sort_values("action_date")
    actions = []
    for _, row in frame.iterrows():
        cash = pd.to_numeric(row.get("派息比例", np.nan), errors="coerce")
        bonus = pd.to_numeric(row.get("送股比例", np.nan), errors="coerce")
        transfer = pd.to_numeric(row.get("转增比例", np.nan), errors="coerce")
        actions.append((
            row["action_date"],
            0.0 if not np.isfinite(cash) else float(cash) / 10.0,
            1.0 + (0.0 if not np.isfinite(bonus) else float(bonus) / 10.0)
            + (0.0 if not np.isfinite(transfer) else float(transfer) / 10.0),
        ))
    return actions


def dividend_tax_rate(buy_date: pd.Timestamp, pay_date: pd.Timestamp) -> float:
    """Return A-share individual dividend withholding by holding period."""
    days = (pay_date - buy_date).days
    if days <= 30:
        return 0.20
    if days <= 365:
        return 0.10
    return 0.0
