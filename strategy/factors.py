"""基于价格序列的横截面因子。"""

from __future__ import annotations

import pandas as pd


PRICE_FACTORS = ["momentum_12_1", "low_vol_12m", "rsi_14_inv"]
PRICE_FACTOR_LABELS = {
    "momentum_12_1": "动量 (12-1月)",
    "low_vol_12m": "低波 (12月)",
    "rsi_14_inv": "RSI 反转 (14日)",
}


def calc_momentum_12_1(close: pd.Series, as_of, **kwargs) -> float | None:
    data = close[close.index <= as_of]
    return data.iloc[-22] / data.iloc[-273] - 1 if len(data) >= 273 else None


def calc_low_vol_12m(close: pd.Series, as_of, **kwargs) -> float | None:
    data = close[close.index <= as_of]
    if len(data) < 252:
        return None
    vol = data.pct_change().dropna().iloc[-252:].std()
    return -vol if vol > 0 else None


def calc_rsi_14_inv(close: pd.Series, as_of, **kwargs) -> float | None:
    data = close[close.index <= as_of]
    if len(data) < 16:
        return None
    changes = data.iloc[-15:].diff().dropna()
    gains = changes.clip(lower=0).sum() / 14
    losses = -changes.clip(upper=0).sum() / 14
    rsi = 100.0 if losses == 0 and gains > 0 else 50.0
    if losses > 0:
        rsi = 100 - 100 / (1 + gains / losses)
    return -rsi


PRICE_CALCS = {
    "momentum_12_1": calc_momentum_12_1,
    "low_vol_12m": calc_low_vol_12m,
    "rsi_14_inv": calc_rsi_14_inv,
}
