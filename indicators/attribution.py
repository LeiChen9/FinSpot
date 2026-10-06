"""基金业绩归因与风险分解 — 纯函数，无状态"""
import numpy as np
import pandas as pd


def calc_beta(r_P: pd.Series, r_B: pd.Series) -> float:
    """组合相对基准的 Beta"""
    cov = np.cov(r_P, r_B)
    return cov[0, 1] / cov[1, 1]


def calc_volatility(returns: pd.Series, annualize: int = 252) -> float:
    """年化波动率"""
    return returns.std(ddof=1) * np.sqrt(annualize)


def calc_residual_risk(r_P: pd.Series, r_B: pd.Series, annualize: int = 252) -> float:
    """年化残余风险（主动风险）"""
    beta = calc_beta(r_P, r_B)
    theta = r_P - beta * r_B
    return theta.std(ddof=1) * np.sqrt(annualize)


def calc_realized_alpha(r_P: pd.Series, r_B: pd.Series) -> pd.Series:
    """实现的 Alpha = r_P - β * r_B"""
    beta = calc_beta(r_P, r_B)
    return r_P - beta * r_B


def calc_realized_risk_premium(r_P: pd.Series, r_B: pd.Series) -> pd.Series:
    """实现的风险溢价 = β * r_B"""
    beta = calc_beta(r_P, r_B)
    return beta * r_B


def calc_exceptional_benchmark_return(r_B: pd.Series, f_B: float) -> pd.Series:
    """超常业绩基准收益率 = r_B - f_B"""
    return r_B - f_B


def calc_expected_benchmark_excess_return(r_B: pd.Series) -> float:
    """业绩基准的预期超额收益率 E[r_B]"""
    return r_B.mean()


def calc_info_ratio(alpha: pd.Series, omega: pd.Series) -> pd.Series:
    """信息比率 = Alpha / Omega"""
    return alpha / omega.replace(0, np.nan)


def analyze_rolling(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """滚动计算 Beta 和年化 Alpha

    df 必须包含 r_P（组合超额收益）和 r_B（基准超额收益）
    """
    result = []
    for i in range(len(df) - window + 1):
        w = df.iloc[i: i + window]
        rP, rB = w['r_P'], w['r_B']
        beta = calc_beta(rP, rB)
        alpha_daily = rP.mean() - beta * rB.mean()
        result.append({
            'date': df.iloc[i + window - 1]['date'],
            'beta': beta,
            'alpha': alpha_daily * 252,
        })
    return pd.DataFrame(result)


def decompose_risk(r_P: pd.Series, r_B: pd.Series, annualize: int = 252) -> dict:
    """风险分解：总风险 = 系统性风险 + 残余风险"""
    sigma_fund = calc_volatility(r_P, annualize)
    sigma_bm = calc_volatility(r_B, annualize)
    beta = calc_beta(r_P, r_B)
    omega = calc_residual_risk(r_P, r_B, annualize)
    systematic = (beta ** 2) * (sigma_bm ** 2)

    return {
        'sigma_fund': sigma_fund,
        'sigma_benchmark': sigma_bm,
        'beta': beta,
        'omega': omega,
        'variance_systematic': systematic,
        'variance_residual': omega ** 2,
        'variance_total': sigma_fund ** 2,
    }
