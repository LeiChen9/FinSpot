"""基金/策略归因分析 — CAPM Beta/Alpha/跟踪误差/信息比率"""
import pandas as pd
import numpy as np
from typing import Optional, List


def align_fund_benchmark(fund_returns, fund_dates, bm_returns, bm_dates, rf_annual: float = 0.02) -> pd.DataFrame:
    """对齐基金和基准日期，计算超额收益"""
    fund_df = pd.DataFrame({'date': fund_dates, 'R_P': fund_returns}).dropna()
    bm_df = pd.DataFrame({'date': bm_dates, 'R_B': bm_returns}).dropna()
    merged = pd.merge(fund_df, bm_df, on='date', how='inner').dropna()
    merged.sort_values('date', inplace=True)
    rf_daily = rf_annual / 252
    merged['r_P'] = merged['R_P'] - rf_daily
    merged['r_B'] = merged['R_B'] - rf_daily
    return merged


def calc_beta(rP, rB) -> float:
    cov = np.cov(rP, rB)
    return cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0


def calc_alpha(rP, rB, beta: Optional[float] = None, periods_per_year: int = 252) -> float:
    if beta is None:
        beta = calc_beta(rP, rB)
    return (np.mean(rP) - beta * np.mean(rB)) * periods_per_year


def tracking_error(rP, rB, beta: Optional[float] = None, periods_per_year: int = 252) -> float:
    if beta is None:
        beta = calc_beta(rP, rB)
    residual = rP - beta * rB
    return np.std(residual, ddof=1) * np.sqrt(periods_per_year)


def information_ratio(alpha_annual: float, te: float) -> float:
    return alpha_annual / te if te > 0 else 0.0


def r_squared(rP, rB, beta: Optional[float] = None) -> float:
    if beta is None:
        beta = calc_beta(rP, rB)
    if np.var(rP) == 0:
        return 0.0
    return (beta ** 2) * np.var(rB) / np.var(rP)


def fund_attribution(aligned: pd.DataFrame, fund_name: str = '', bm_name: str = '', rf_annual: float = 0.02) -> dict:
    """完整归因分析"""
    rP = aligned['r_P'].values
    rB = aligned['r_B'].values

    beta = calc_beta(rP, rB)
    alpha = calc_alpha(rP, rB, beta)
    te = tracking_error(rP, rB, beta)
    ir = information_ratio(alpha, te)
    r2 = r_squared(rP, rB, beta)

    sigma_fund = np.std(rP, ddof=1) * np.sqrt(252)
    sigma_bm = np.std(rB, ddof=1) * np.sqrt(252)
    systematic_risk = beta * sigma_bm
    residual_risk = te

    return {
        'fund_name': fund_name,
        'bm_name': bm_name,
        'beta': beta,
        'alpha_annual': alpha,
        'sigma_fund': sigma_fund,
        'sigma_bm': sigma_bm,
        'tracking_error': te,
        'information_ratio': ir,
        'r_squared': r2,
        'systematic_risk': systematic_risk,
        'residual_risk': residual_risk,
        'risk_residual_pct': residual_risk / sigma_fund if sigma_fund > 0 else 0,
        'n_obs': len(rP),
    }


def rolling_attribution(aligned: pd.DataFrame, window: int = 60, fund_name: str = '') -> pd.DataFrame:
    """滚动计算 Beta 和 Alpha"""
    rP = aligned['r_P'].values
    rB = aligned['r_B'].values
    dates = aligned['date'].values

    result = []
    for i in range(len(aligned) - window + 1):
        wP = rP[i:i + window]
        wB = rB[i:i + window]
        beta = calc_beta(wP, wB)
        alpha = calc_alpha(wP, wB, beta)
        result.append({
            'date': dates[i + window - 1],
            'beta': beta,
            'alpha': alpha,
        })
    return pd.DataFrame(result)
