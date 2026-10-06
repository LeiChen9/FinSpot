"""下行风险量化 — VaR / CVaR / 蒙特卡洛 / Bootstrap / 组合模拟"""
import numpy as np
import pandas as pd
from typing import Optional


def rolling_n_day_returns(daily_returns, n_days: int):
    """滚动计算 N 日累计收益率"""
    returns = np.asarray(daily_returns)
    n = len(returns)
    if n <= n_days:
        return np.array([])
    roll_rets = np.array([
        (1 + returns[i:i + n_days]).prod() - 1
        for i in range(n - n_days + 1)
    ])
    return roll_rets


def historical_var_cvar(daily_returns, n_days: int, alpha: float = 0.05):
    """历史模拟法计算 VaR 和 CVaR"""
    roll_rets = rolling_n_day_returns(daily_returns, n_days)
    if len(roll_rets) == 0:
        return 0.0, 0.0, np.array([])
    var = np.percentile(roll_rets, alpha * 100)
    cvar = roll_rets[roll_rets <= var].mean() if np.any(roll_rets <= var) else var
    return var, cvar, roll_rets


def monte_carlo_simulation(daily_returns, n_days: int, n_simulations: int = 100000, seed: int = 42):
    """参数蒙特卡洛模拟（正态分布假设）"""
    np.random.seed(seed)
    mu = daily_returns.mean()
    sigma = daily_returns.std(ddof=1)
    random_returns = np.random.normal(mu, sigma, size=(n_simulations, n_days))
    cumulative_returns = np.prod(1 + random_returns, axis=1) - 1
    return cumulative_returns


def bootstrap_simulation(daily_returns, n_days: int, n_simulations: int = 100000, seed: int = 42):
    """Bootstrap 法模拟（无分布假设）"""
    np.random.seed(seed)
    returns = np.asarray(daily_returns)
    indices = np.random.randint(0, len(returns), size=(n_simulations, n_days))
    sampled = returns[indices]
    cumulative_returns = np.prod(1 + sampled, axis=1) - 1
    return cumulative_returns


def portfolio_bootstrap(df_returns_1, df_returns_2, n_days: int, n_simulations: int = 100000,
                         weight1: float = 0.5, weight2: float = 0.5,
                         date_col: str = 'date', ret_col_1: str = 'R', ret_col_2: str = 'R',
                         seed: int = 42):
    """组合 Bootstrap 模拟（保留相关性）"""
    np.random.seed(seed)
    merged = df_returns_1[[date_col, ret_col_1]].merge(
        df_returns_2[[date_col, ret_col_2]], on=date_col, how='inner'
    ).dropna()
    r1 = merged[ret_col_1].values
    r2 = merged[ret_col_2].values
    n_available = len(r1)
    if n_available == 0:
        return np.array([])
    indices = np.random.randint(0, n_available, size=(n_simulations, n_days))
    sampled_1 = r1[indices]
    sampled_2 = r2[indices]
    cum_1 = np.prod(1 + sampled_1, axis=1) - 1
    cum_2 = np.prod(1 + sampled_2, axis=1) - 1
    portfolio_return = weight1 * cum_1 + weight2 * cum_2
    return portfolio_return


def analyze_simulation(sim_returns, fund_value: float = 1.0, threshold_loss: Optional[float] = None) -> dict:
    """分析模拟结果"""
    if len(sim_returns) == 0:
        return {}
    var_95 = np.percentile(sim_returns, 5)
    var_99 = np.percentile(sim_returns, 1)
    cvar_95 = sim_returns[sim_returns <= var_95].mean()
    median_ret = np.median(sim_returns)
    mean_ret = np.mean(sim_returns)
    upside_prob = (sim_returns > 0).mean()
    downside_prob = (sim_returns <= 0).mean()
    avg_gain = sim_returns[sim_returns > 0].mean() if upside_prob > 0 else 0
    avg_loss = abs(sim_returns[sim_returns <= 0].mean()) if downside_prob > 0 else 1
    ratio = avg_gain / avg_loss if avg_loss > 0 else 0

    result = {
        'var_95': var_95, 'var_99': var_99, 'cvar_95': cvar_95,
        'median_ret': median_ret, 'mean_ret': mean_ret,
        'upside_prob': upside_prob, 'avg_gain': avg_gain,
        'avg_loss': avg_loss, 'ratio': ratio,
    }
    if threshold_loss is not None and fund_value > 0:
        threshold_pct = -threshold_loss / fund_value
        result['prob_hit'] = (sim_returns <= threshold_pct).mean()
        result['threshold_pct'] = threshold_pct
    return result


def risk_summary(daily_returns, n_days_list=None, fund_value: float = 1.0, threshold_loss: Optional[float] = None) -> dict:
    """综合风险摘要（历史+蒙特卡洛+Bootstrap）"""
    if n_days_list is None:
        n_days_list = [21, 63, 126]
    results = {}
    for nd in n_days_list:
        label = f"{nd}d"
        var, cvar, _ = historical_var_cvar(daily_returns, nd)
        mc = monte_carlo_simulation(daily_returns, nd, 10000)
        bs = bootstrap_simulation(daily_returns, nd, 10000)
        results[label] = {
            'historical_var': var,
            'historical_cvar': cvar,
            'monte_carlo': analyze_simulation(mc, fund_value, threshold_loss),
            'bootstrap': analyze_simulation(bs, fund_value, threshold_loss),
        }
    return results
