"""组合优化 — 有效前沿 / 最大夏普 / 最小波动"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize


def _annualize_returns_and_cov(returns_df: pd.DataFrame, periods_per_year: int = 252):
    ann_ret = returns_df.mean() * periods_per_year
    ann_cov = returns_df.cov() * periods_per_year
    return ann_ret.values, ann_cov.values


def efficient_frontier(returns_df: pd.DataFrame, n_portfolios: int = 5000,
                       rf_annual: float = 0.02, periods_per_year: int = 252,
                       allow_short: bool = False) -> dict:
    """计算有效前沿

    Returns:
        dict with 'portfolios' (list of {'ret','vol','sharpe','weights'}),
              'frontier' (DataFrame of max sharpe per vol level)
    """
    ann_ret, ann_cov = _annualize_returns_and_cov(returns_df, periods_per_year)
    n_assets = len(ann_ret)

    portfolios = []
    for _ in range(n_portfolios):
        w = np.random.random(n_assets)
        if not allow_short:
            w = w / w.sum()
        else:
            w = w - w.mean()
        ret = np.dot(w, ann_ret)
        vol = np.sqrt(np.dot(w.T, np.dot(ann_cov, w)))
        sharpe = (ret - rf_annual) / vol if vol > 0 else 0
        portfolios.append({'ret': ret, 'vol': vol, 'sharpe': sharpe, 'weights': w})

    return {'portfolios': portfolios, 'n_assets': n_assets}


def max_sharpe_portfolio(returns_df: pd.DataFrame, rf_annual: float = 0.02,
                         periods_per_year: int = 252) -> dict:
    """最大夏普比率组合（解析解）"""
    ann_ret, ann_cov = _annualize_returns_and_cov(returns_df, periods_per_year)
    n = len(ann_ret)
    inv_cov = np.linalg.inv(ann_cov)
    ones = np.ones(n)
    mu = ann_ret - rf_annual
    w = np.dot(inv_cov, mu)
    w = w / np.dot(ones, w)
    ret = np.dot(w, ann_ret)
    vol = np.sqrt(np.dot(w.T, np.dot(ann_cov, w)))
    sharpe = (ret - rf_annual) / vol if vol > 0 else 0
    return {'weights': w, 'ret': ret, 'vol': vol, 'sharpe': sharpe}


def min_vol_portfolio(returns_df: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """最小波动率组合"""
    ann_ret, ann_cov = _annualize_returns_and_cov(returns_df, periods_per_year)
    n = len(ann_ret)
    inv_cov = np.linalg.inv(ann_cov)
    ones = np.ones(n)
    w = np.dot(inv_cov, ones)
    w = w / np.dot(ones, w)
    ret = np.dot(w, ann_ret)
    vol = np.sqrt(np.dot(w.T, np.dot(ann_cov, w)))
    sharpe = ret / vol if vol > 0 else 0
    return {'weights': w, 'ret': ret, 'vol': vol, 'sharpe': sharpe}


def mean_variance_optimize(
    mu: np.ndarray, sigma: np.ndarray, lambda_risk: float = 8.0
) -> np.ndarray:
    """求解多因子策略使用的 long-only 均值方差权重。"""
    n = len(mu)
    if n == 0:
        return np.array([])
    if np.all(np.abs(mu) < 1e-10):
        return np.ones(n) / n

    def objective(weights):
        return -(weights @ mu - 0.5 * lambda_risk * weights @ sigma @ weights)

    result = minimize(
        objective,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, 0.3)] * n,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    if not result.success or not np.all(np.isfinite(result.x)) or result.x.sum() <= 0:
        return np.ones(n) / n
    weights = result.x.copy()
    weights[weights < 1e-6] = 0.0
    return weights / weights.sum()
