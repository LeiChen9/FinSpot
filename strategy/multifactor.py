"""多因子量化选股策略 — 核心模块

Grinold & Kahn 主动投资框架:
  因子计算 → 横截面 Rank IC / IR → IC_IR 加权信号合成
  → Rank 打分加权 → 组合权重分配 → 调仓执行
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable
from datetime import datetime
from analysis.performance import nav_summary
from data.financial import FinancialDataLoader
from strategy.factors import PRICE_FACTORS
from strategy.fundamental_factors import FINANCIAL_FACTORS
from strategy.model_types import BacktestResult, PeriodSnapshot
from strategy.factor_model import FactorModel as CanonicalFactorModel

ALL_FACTORS = PRICE_FACTORS + FINANCIAL_FACTORS

# Compatibility export; the implementation lives in strategy.factor_model.
FactorModel = CanonicalFactorModel

class MultiFactorBacktest:
    """多因子策略回测引擎"""

    def __init__(self, stocks: List[str], factors: List[str] = None,
                 lambda_risk: float = 8.0, warmup_periods: int = 8,
                 fin_loader: FinancialDataLoader = None):
        self.fin_loader = fin_loader
        self.factor_model = CanonicalFactorModel(factors, fin_loader=fin_loader)
        self.stocks = stocks
        self.lambda_risk = lambda_risk
        self.warmup_periods = warmup_periods
        self.factors = factors or ALL_FACTORS

    def run(self, rebalance_dates: List[datetime],
            data_loader: Callable) -> BacktestResult:
        result = BacktestResult()
        warmup: List[PeriodSnapshot] = []
        ic_history: Dict[str, List[float]] = {f: [] for f in self.factors}
        ir: Dict[str, float] = {f: 0.0 for f in self.factors}
        n_dates = len(rebalance_dates)

        # Pre-load all market data to speed up
        all_market = {}
        for s in self.stocks:
            df = data_loader(s, rebalance_dates[-1])
            if df is not None and len(df) > 250:
                all_market[s] = df

        for t, rb_date in enumerate(rebalance_dates):
            # Slice market data up to rb_date
            market_data = {}
            for s, df in all_market.items():
                sliced = df[df.index <= pd.Timestamp(rb_date)]
                if len(sliced) > 250:
                    market_data[s] = sliced

            # Compute factors
            factor_df = self.factor_model.compute_factors(market_data, rb_date)
            if factor_df.empty or len(factor_df) < 10:
                continue
            active_stocks = factor_df.index.tolist()

            # Forward return to next rebalance date
            fwd_ret = pd.Series(dtype=float)
            if t < n_dates - 1:
                next_date = rebalance_dates[t + 1]
                rets = {}
                for s in active_stocks:
                    df = all_market.get(s)
                    if df is None:
                        continue
                    p0_data = df[df.index <= pd.Timestamp(rb_date)]
                    p1_data = df[df.index <= pd.Timestamp(next_date)]
                    if len(p0_data) > 0 and len(p1_data) > 0:
                        p0 = p0_data['close'].iloc[-1]
                        p1 = p1_data['close'].iloc[-1]
                        rets[s] = p1 / p0 - 1
                fwd_ret = pd.Series(rets)

            # Estimate factor returns
            f_ret, resid_var = self.factor_model.estimate_factor_returns(
                factor_df, fwd_ret)

            snap = PeriodSnapshot(
                date=rb_date, stocks=active_stocks,
                factor_df=factor_df, forward_ret=fwd_ret,
                factor_returns=f_ret, residual_var=resid_var,
            )
            warmup.append(snap)

            # Accumulate warmup before signal generation
            if len(warmup) < self.warmup_periods + 1:
                continue

            # Keep window size
            if len(warmup) > self.warmup_periods + 1:
                warmup = warmup[-(self.warmup_periods + 1):]

            # Update IC history from warmup
            ic_history = self.factor_model.compute_ic_history(
                warmup[:-1], self.factors)
            ir = self.factor_model.compute_ir(ic_history)

            # Rank-based signal weighting: top 100, weighted by signal value
            signal = self.factor_model.synthesize_signal(ir, factor_df)
            if np.all(np.abs(signal) < 1e-10):
                w = np.ones(len(active_stocks)) / len(active_stocks)
            else:
                signal_series = pd.Series(signal, index=active_stocks)
                top_n = min(100, len(active_stocks))
                selected = signal_series.nlargest(top_n)
                selected = selected[selected > 0]
                if len(selected) == 0:
                    w = np.ones(len(active_stocks)) / len(active_stocks)
                else:
                    w_sub = selected.values / selected.values.sum()
                    w = np.zeros(len(active_stocks))
                    for s, wi in zip(selected.index, w_sub):
                        idx = active_stocks.index(s)
                        w[idx] = wi

            result.weights_history.append({
                'date': rb_date,
                'weights': dict(zip(active_stocks, w)),
                'ir': dict(ir),
                'n_ic_obs': {f: len(v) for f, v in ic_history.items()},
            })

        # Build daily NAV
        if result.weights_history:
            result.daily_nav = self._build_daily_nav(
                result.weights_history, all_market)

        return result

    def _build_daily_nav(self, weights_history: List[dict],
                          all_market: Dict[str, pd.DataFrame]) -> pd.Series:
        if not weights_history:
            return pd.Series()

        all_dates = sorted(set(
            d for df in all_market.values() for d in df.index
        ))

        shares: Dict[str, float] = {}
        nav_value = 1.0
        result = []

        rebalance_map = {}
        for e in weights_history:
            rebalance_map[e['date']] = e['weights']

        for d in all_dates:
            dt = datetime(d.year, d.month, d.day)

            if dt in rebalance_map:
                w = rebalance_map[dt]
                # Compute NAV from old shares before rebalancing
                if shares:
                    total = 0.0
                    for s, sh in shares.items():
                        df = all_market.get(s)
                        if df is not None and d in df.index:
                            total += sh * df.loc[d, 'close']
                    if total > 0:
                        nav_value = total
                # Record rebalance-day NAV using old (or initial) value
                result.append({'date': dt, 'nav': nav_value})
                # Build new shares; skip stocks without price data and renormalize
                new_shares = {}
                total_weight = 0.0
                for s, wi in w.items():
                    if wi <= 0:
                        continue
                    df = all_market.get(s)
                    if df is not None and d in df.index:
                        price = float(df.loc[d, 'close'])
                        if price > 0:
                            new_shares[s] = wi / price
                            total_weight += wi
                if new_shares and total_weight > 0:
                    for s in new_shares:
                        new_shares[s] = (new_shares[s] / total_weight) * nav_value
                shares = new_shares
                continue

            if not shares:
                continue

            total = 0.0
            for s, sh in shares.items():
                df = all_market.get(s)
                if df is not None and d in df.index:
                    total += sh * float(df.loc[d, 'close'])
            if total > 0:
                result.append({'date': dt, 'nav': total})

        if not result:
            return pd.Series()

        df = pd.DataFrame(result).set_index('date')
        df['nav'] = df['nav'] / df['nav'].iloc[0]
        return df['nav']


# ─── 绩效评估 ───

def perf_summary(nav: pd.Series, rf_annual: float = 0.02) -> dict:
    return nav_summary(nav, rf_annual)
