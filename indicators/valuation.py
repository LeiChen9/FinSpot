"""估值指标计算 — 纯函数，无状态"""
import pandas as pd
from typing import Dict, Optional


def calculate_percentiles(df: pd.DataFrame, period_years: int = 5) -> Dict:
    """计算估值指标（PE/PB/股息率）的当前值及历史分位数

    参数
    ----------
    df : DataFrame
        需包含 pe_ttm / pb / dividend_yield 列（至少其一）
    period_years : int
        回溯窗口（年）

    返回
    -------
    dict : { indicator: { 'current': float, 'percentile': float } }
    """
    last_date = df.index[-1]
    start_date = last_date - pd.DateOffset(years=period_years)
    window = df[df.index >= start_date].copy()

    cols = [c for c in ['pe_ttm', 'pb', 'dividend_yield'] if c in window.columns]
    if cols:
        window = window.dropna(subset=cols)
    if window.empty:
        return {}

    current = window.iloc[-1]
    metrics = {}
    for indicator in ['pe_ttm', 'pb', 'dividend_yield']:
        if indicator in window.columns and not pd.isna(current.get(indicator)):
            val = current[indicator]
            metrics[indicator] = {
                'current': val,
                'percentile': (window[indicator] < val).mean(),
            }
    return metrics
