"""持有期分析 — "买入到+N%需要多久"、达标天数、回撤"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


def days_to_target(price_series: pd.Series, idx: int, target_pct: float = 0.08):
    """从 idx 买入, 计算达到 target_pct% 收益需要的交易日数"""
    bp = price_series.iloc[idx]
    hit = price_series.iloc[idx + 1:][price_series.iloc[idx + 1:] >= bp * (1 + target_pct)]
    if len(hit):
        nd = price_series.index.get_loc(hit.index[0]) - idx
        return nd, True
    nd = len(price_series) - idx - 1
    return nd, False


def holding_mdd(price_series, start: int, end: int) -> float:
    seg = price_series.iloc[start:end + 1]
    if len(seg) < 2:
        return 0.0
    return (seg / seg.expanding().max() - 1).min()


def holding_vol(price_series, start: int, end: int) -> float:
    seg = price_series.iloc[start:end + 1].pct_change().dropna()
    if len(seg) < 2:
        return 0.0
    return seg.std(ddof=1) * np.sqrt(252)


def generate_entry_positions(dates: pd.DatetimeIndex, days_of_month=None):
    """生成每月 1 日和 15 日的买入位置索引"""
    if days_of_month is None:
        days_of_month = [1, 15]
    positions = []
    seen = set()
    for m_start in pd.date_range(dates[0].replace(day=1), dates[-1], freq='MS'):
        for day in days_of_month:
            cand = m_start.replace(day=min(day, m_start.days_in_month))
            pos = dates.searchsorted(cand, side='left')
            if pos < len(dates) and dates[pos] not in seen:
                seen.add(dates[pos])
                positions.append(pos)
    return positions


def holding_analysis(price_series: pd.Series, target_pct: float = 0.08,
                     days_of_month=None, min_remaining: int = 5) -> pd.DataFrame:
    """对单个标的的完整持有期分析"""
    if days_of_month is None:
        days_of_month = [1, 15]
    dates = price_series.index
    positions = generate_entry_positions(dates, days_of_month)

    rows = []
    for idx in positions:
        if idx >= len(price_series) - min_remaining:
            continue
        nd, ok = days_to_target(price_series, idx, target_pct)
        end = idx + nd if ok else len(price_series) - 1
        rows.append({
            'buy_date': dates[idx],
            'buy_price': price_series.iloc[idx],
            'days_to_target': nd,
            'reached': ok,
            'max_drawdown': holding_mdd(price_series, idx, end),
            'volatility': holding_vol(price_series, idx, end),
        })
    return pd.DataFrame(rows)


def holding_summary(df: pd.DataFrame, name: str = '') -> dict:
    """汇总持有期分析结果"""
    if df.empty:
        return {'name': name, 'n': 0, 'succ_pct': 0.0}
    n = len(df)
    r = df[df['reached']]
    succ = len(r) / n * 100 if n else 0
    result = {'name': name, 'n': n, 'succ_pct': succ}
    if len(r):
        d = r['days_to_target']
        result.update({
            'mean_days': d.mean(), 'median_days': d.median(),
            'min_days': d.min(), 'max_days': d.max(),
            'avg_mdd': r['max_drawdown'].mean(),
            'avg_vol': r['volatility'].mean(),
        })
    return result


def multi_stock_holding_analysis(stock_data: Dict[str, pd.Series],
                                 names: Dict[str, str] = None,
                                 target_pct: float = 0.08,
                                 days_of_month=None) -> pd.DataFrame:
    """多标的同时持有期分析"""
    summaries = []
    for code, series in stock_data.items():
        result = holding_analysis(series, target_pct, days_of_month)
        label = names.get(code, code) if names else code
        s = holding_summary(result, label)
        summaries.append(s)
    return pd.DataFrame(summaries)
