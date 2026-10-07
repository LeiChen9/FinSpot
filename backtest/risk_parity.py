"""全天候风险平价策略回测引擎

等风险贡献 (ERC) — 使每项资产的边际风险贡献相等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from backtest.risk_parity_math import calc_erc_weights
from backtest.portfolio_nav import build_nav_from_weights as _build_nav_from_weights


@dataclass
class BacktestResult:
    weights_history: List[dict] = field(default_factory=list)
    daily_nav: pd.Series = field(default_factory=pd.Series)
    daily_weights: pd.DataFrame = field(default_factory=pd.DataFrame)


class RiskParityBacktest:
    """风险平价策略回测引擎

    每个调仓日：用过去 cov_window 天日收益率估算协方差矩阵，
    求解 ERC 权重，构建每日 NAV。
    """

    def __init__(self, assets: List[str], cov_window: int = 126,
                 max_weight: float = 1.0):
        """
        Args:
            assets: 资产代码列表
            cov_window: 协方差估计窗口（交易日数），默认 126 ≈ 6 个月
            max_weight: 单资产最大权重上限
        """
        self.assets = assets
        self.cov_window = cov_window
        self.max_weight = max_weight

    def run(self, rebalance_dates: List[datetime],
            data_loader: Callable) -> BacktestResult:
        result = BacktestResult()

        all_data = {}
        for a in self.assets:
            df = data_loader(a, rebalance_dates[-1])
            if df is not None and len(df) > 20:
                all_data[a] = df
            else:
                print(f'  [WARN] no data for {a}')

        if len(all_data) < 2:
            print('  [ERROR] fewer than 2 assets have data')
            return result

        for t, rb_date in enumerate(rebalance_dates):
            prices = {}
            for a, df in all_data.items():
                sliced = df[df.index <= pd.Timestamp(rb_date)]
                if len(sliced) > max(20, self.cov_window // 4):
                    prices[a] = sliced['close']

            if len(prices) < 2:
                continue

            price_df = pd.DataFrame(prices)
            returns = price_df.pct_change().dropna()
            if len(returns) < 2:
                weights = pd.Series(1.0 / len(prices), index=list(prices.keys()))
            else:
                if len(returns) < self.cov_window:
                    cov = returns.cov()
                else:
                    cov = returns.iloc[-self.cov_window:].cov()
                cov = pd.DataFrame(cov.values + np.eye(len(cov)) * 1e-10,
                                   index=cov.index, columns=cov.columns)
                weights = calc_erc_weights(cov, self.max_weight)

            result.weights_history.append({
                'date': rb_date,
                'weights': weights.to_dict(),
            })

        if result.weights_history:
            result.daily_nav, result.daily_weights = self._build_daily_nav(
                result.weights_history, all_data)

        return result

    def _build_daily_nav(self, weights_history: List[dict],
                         all_data: Dict[str, pd.DataFrame]) -> Tuple[pd.Series, pd.DataFrame]:
        return _build_nav_from_weights(weights_history, all_data)


def _legacy_build_nav_from_weights(
    weights_history: List[dict],
    all_data: Dict[str, pd.DataFrame],
) -> Tuple[pd.Series, pd.DataFrame]:
    """按调仓权重构建每日净值和实际漂移权重。

    ``weights_history`` 的格式与 ``RiskParityBacktest`` 输出一致，
    因此固定权重组合也可以复用同一套调仓和净值口径。
    """
    if not weights_history:
        return pd.Series(), pd.DataFrame()

    all_dates = sorted(set(d for df in all_data.values() for d in df.index))
    shares: Dict[str, float] = {}
    nav_value = 1.0
    nav_records = []
    weight_records = []
    rebalance_map = {e['date']: e['weights'] for e in weights_history}

    for d in all_dates:
        dt = pd.Timestamp(d).to_pydatetime()

        if dt in rebalance_map:
            weights = rebalance_map[dt]
            if shares:
                nav_value = sum(
                    sh * float(all_data[a].loc[d, 'close'])
                    for a, sh in shares.items()
                    if a in all_data and d in all_data[a].index
                ) or nav_value

            nav_records.append({'date': dt, 'nav': nav_value})
            weight_records.append({'date': dt, **weights})

            new_shares = {}
            total_weight = 0.0
            for asset, weight in weights.items():
                if weight <= 0 or asset not in all_data or d not in all_data[asset].index:
                    continue
                price = float(all_data[asset].loc[d, 'close'])
                if price > 0:
                    new_shares[asset] = weight / price
                    total_weight += weight
            if new_shares and total_weight > 0:
                shares = {
                    asset: quantity / total_weight * nav_value
                    for asset, quantity in new_shares.items()
                }
            else:
                shares = {}
            continue

        if not shares:
            continue

        total = sum(
            sh * float(all_data[a].loc[d, 'close'])
            for a, sh in shares.items()
            if a in all_data and d in all_data[a].index
        )
        if total <= 0:
            continue

        nav_records.append({'date': dt, 'nav': total})
        weight_records.append({
            'date': dt,
            **{
                asset: sh * float(all_data[asset].loc[d, 'close']) / total
                for asset, sh in shares.items()
                if asset in all_data and d in all_data[asset].index
            },
        })

    if not nav_records:
        return pd.Series(), pd.DataFrame()

    df_nav = pd.DataFrame(nav_records).set_index('date')
    df_nav['nav'] = df_nav['nav'] / df_nav['nav'].iloc[0]
    df_weights = pd.DataFrame(weight_records).set_index('date')
    return df_nav['nav'], df_weights


# Compatibility export; the canonical implementation lives in portfolio_nav.
build_nav_from_weights = _build_nav_from_weights
