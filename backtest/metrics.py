"""绩效诊断 — 年化收益 / 波动 / 夏普 / 回撤 / 胜率"""

import numpy as np
import pandas as pd


def perf_metrics(nav: pd.Series, rf: float = 0.02) -> dict:
    if len(nav) < 20:
        return {}
    relative = nav / nav.iloc[0]
    daily = relative.pct_change().dropna()
    years = max((relative.index[-1] - relative.index[0]).days / 365.25, 1 / 365.25)
    annual_return = relative.iloc[-1] ** (1 / years) - 1
    volatility = daily.std() * np.sqrt(252)
    sharpe = (annual_return - rf) / volatility if volatility > 0 else 0.0
    drawdown = (relative / relative.expanding().max() - 1).min()
    return {
        "total_return": relative.iloc[-1] - 1,
        "annual_return": annual_return,
        "annual_vol": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "win_rate": (daily > 0).mean(),
        "trading_days": len(daily),
    }


def annualize_return(daily_returns, periods_per_year: int = 252) -> float:
    if len(daily_returns) == 0:
        return 0.0
    return (1 + daily_returns).prod() ** (periods_per_year / len(daily_returns)) - 1


def annualize_vol(daily_returns, periods_per_year: int = 252) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return daily_returns.std(ddof=1) * np.sqrt(periods_per_year)


def max_drawdown(price_series) -> float:
    return (price_series / price_series.expanding().max() - 1).min()


def calc_sharpe(daily_returns, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    excess = annualize_return(daily_returns, periods_per_year) - rf_annual
    volatility = annualize_vol(daily_returns, periods_per_year)
    return excess / volatility if volatility > 0 else 0.0


def calc_sortino(daily_returns, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    excess = annualize_return(daily_returns, periods_per_year) - rf_annual
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        return 0.0
    deviation = downside.std(ddof=1) * np.sqrt(periods_per_year)
    return excess / deviation if deviation > 0 else 0.0


def calc_calmar(daily_returns, price_series, rf_annual: float = 0.02, periods_per_year: int = 252) -> float:
    drawdown = max_drawdown(price_series)
    excess = annualize_return(daily_returns, periods_per_year) - rf_annual
    return excess / abs(drawdown) if drawdown != 0 else 0.0


def win_rate(daily_returns) -> float:
    return (daily_returns > 0).mean()


def calc_downside_deviation(daily_returns, mar: float = 0, periods_per_year: int = 252) -> float:
    downside = daily_returns[daily_returns < mar]
    if len(downside) < 2:
        return 0.0
    return np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year)


def performance_summary(daily_returns, price_series=None, rf_annual: float = 0.02) -> dict:
    price_series = price_series if price_series is not None else (1 + daily_returns).cumsum()
    return {
        "annual_return": annualize_return(daily_returns),
        "annual_vol": annualize_vol(daily_returns),
        "sharpe": calc_sharpe(daily_returns, rf_annual),
        "sortino": calc_sortino(daily_returns, rf_annual),
        "calmar": calc_calmar(daily_returns, price_series, rf_annual),
        "max_drawdown": max_drawdown(price_series),
        "win_rate": win_rate(daily_returns),
        "cumulative_return": (1 + daily_returns).prod() - 1,
        "rf_used": rf_annual,
    }


def nav_summary(nav: pd.Series, rf_annual: float = 0.02) -> dict:
    if len(nav) < 2:
        return {}
    returns = nav.pct_change().dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1 / 365.25)
    annual_return = nav.iloc[-1] ** (1 / years) - 1
    annual_vol = annualize_vol(returns)
    drawdown = max_drawdown(nav)
    return {
        "total_return": f"{nav.iloc[-1] - 1:.2%}",
        "annual_return": f"{annual_return:.2%}",
        "annual_vol": f"{annual_vol:.2%}",
        "sharpe": f"{(annual_return - rf_annual) / annual_vol:.2f}" if annual_vol else "0.00",
        "calmar": f"{(annual_return - rf_annual) / abs(drawdown):.2f}" if drawdown else "0.00",
        "max_drawdown": f"{drawdown:.2%}",
        "win_rate": f"{(returns > 0).mean():.2%}",
        "n_trading_days": len(returns),
    }


def calc_perf(nav, name: str) -> dict:
    """单条净值曲线的完整绩效行 (含 Sortino)。"""
    ret = nav.pct_change().dropna()
    if len(ret) == 0:
        return {}
    ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    dd = nav / nav.expanding().max() - 1
    mdd = dd.min()
    calmar = (ann_ret - 0.02) / abs(mdd) if mdd != 0 else 0
    downside = ret[ret < 0]
    down_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0
    sortino = (ann_ret - 0.02) / down_vol if down_vol > 0 else 0
    return {
        'name': name,
        'cumulative': nav.iloc[-1] - 1,
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_drawdown': mdd,
    }


def formatted_perf_row(nav_series: pd.Series, name: str) -> dict:
    """中文表格口径: 按实际日历年化, 用于银行 PB notebook 报告。"""
    returns = nav_series.pct_change().dropna()
    total = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    annual_return = (1 + total) ** (1 / years) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    drawdown = (nav_series - nav_series.cummax()) / nav_series.cummax()
    max_drawdown = drawdown.min()
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    return {
        '策略': name,
        '累计收益': f'{total * 100:.2f}%',
        '年化收益': f'{annual_return * 100:.2f}%',
        '年化波动': f'{annual_vol * 100:.2f}%',
        'Sharpe': f'{sharpe:.2f}',
        '最大回撤': f'{max_drawdown * 100:.2f}%',
        'Calmar': f'{calmar:.2f}',
    }


def monthly_returns(nav) -> pd.DataFrame:
    """月度收益透视表 (index=年, columns=月)。"""
    ret = nav.pct_change().dropna()
    ret.index = pd.to_datetime(ret.index)
    monthly = ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
    table = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })
    pivot = table.pivot(index='year', columns='month', values='return')
    pivot.columns = [f'{m}月' for m in pivot.columns]
    return pivot
