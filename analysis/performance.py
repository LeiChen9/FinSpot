"""绩效诊断 — 年化收益 / 波动 / 夏普 / 回撤 / 胜率"""
from typing import Optional
import pandas as pd
import numpy as np


def annualize_return(daily_returns, periods_per_year: int = 252) -> float:
    if len(daily_returns) == 0:
        return 0.0
    total_ret = (1 + daily_returns).prod()
    n = len(daily_returns)
    return total_ret ** (periods_per_year / n) - 1


def annualize_vol(daily_returns, periods_per_year: int = 252) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return daily_returns.std(ddof=1) * np.sqrt(periods_per_year)


def max_drawdown(price_series) -> float:
    rolling_max = price_series.expanding().max()
    drawdown = price_series / rolling_max - 1
    return drawdown.min()


def calc_sharpe(daily_returns, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    er = annualize_return(daily_returns, periods_per_year) - rf_annual
    vol = annualize_vol(daily_returns, periods_per_year)
    return er / vol if vol > 0 else 0.0


def calc_sortino(daily_returns, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    er = annualize_return(daily_returns, periods_per_year) - rf_annual
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_vol = downside.std(ddof=1) * np.sqrt(periods_per_year)
    return er / downside_vol if downside_vol > 0 else 0.0


def calc_calmar(daily_returns, price_series, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    er = annualize_return(daily_returns, periods_per_year) - rf_annual
    mdd = max_drawdown(price_series)
    return er / abs(mdd) if mdd != 0 else 0.0


def win_rate(daily_returns) -> float:
    return (daily_returns > 0).mean()


def calc_downside_deviation(daily_returns, mar: float = 0, periods_per_year: int = 252) -> float:
    downside = daily_returns[daily_returns < mar]
    if len(downside) < 2:
        return 0.0
    return np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)


def performance_summary(daily_returns, price_series=None, rf_annual: float = 0.02) -> dict:
    """完整的绩效摘要"""
    if price_series is None:
        price_series = (1 + daily_returns).cumsum()

    ann_ret = annualize_return(daily_returns)
    ann_vol = annualize_vol(daily_returns)
    sharpe = calc_sharpe(daily_returns, rf_annual)
    sortino = calc_sortino(daily_returns, rf_annual)
    mdd = max_drawdown(price_series)
    calmar = calc_calmar(daily_returns, price_series, rf_annual)
    win = win_rate(daily_returns)
    cum_ret = (1 + daily_returns).prod() - 1

    return {
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': mdd,
        'win_rate': win,
        'cumulative_return': cum_ret,
        'rf_used': rf_annual,
    }
