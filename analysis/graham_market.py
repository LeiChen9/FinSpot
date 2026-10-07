"""Graham 策略使用的估值仪表盘和宏观辅助指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.graham import load_10y, load_all_a_pe


GAUGE_LOOKBACK_YEARS = 10
GAUGE_LO_PCT = 0.50
GAUGE_HI_PCT = 0.85
GAUGE_LO_WEIGHT = 0.40


def r10y(date: pd.Timestamp) -> float:
    values = load_10y()["中国国债收益率10年"]
    values = values[values.index <= date]
    return float(values.iloc[-1]) / 100 if not values.empty else np.nan


def market_avg_pe_5y(date: pd.Timestamp) -> float:
    frame = load_all_a_pe()
    values = frame[(frame.index > date - pd.DateOffset(years=5)) & (frame.index <= date)]
    return float(values["averagePETTM"].mean()) if not values.empty else np.nan


def market_gauge(date: pd.Timestamp, lookback_years: int = GAUGE_LOOKBACK_YEARS) -> float:
    values = load_all_a_pe()["averagePETTM"]
    values = values[values.index <= date]
    if len(values) < 60:
        return 0.5
    history = values[values.index > date - pd.DateOffset(years=lookback_years)]
    if len(history) < 60:
        history = values
    return float((history < values.iloc[-1]).mean())


def target_equity_weight(gauge: float) -> float:
    if gauge <= GAUGE_LO_PCT:
        return 1.0
    if gauge >= GAUGE_HI_PCT:
        return GAUGE_LO_WEIGHT
    decline = (gauge - GAUGE_LO_PCT) / (GAUGE_HI_PCT - GAUGE_LO_PCT)
    return 1.0 - decline * (1.0 - GAUGE_LO_WEIGHT)
