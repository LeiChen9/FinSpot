"""多因子量化选股策略 — 核心模块

Grinold & Kahn 主动投资框架:
  因子计算 → 横截面 Rank IC / IR → IC_IR 加权信号合成
  → Rank 打分加权 → 组合权重分配 → 调仓执行
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field
from analysis.performance import nav_summary
from analysis.optimization import mean_variance_optimize
from data.financial import FinancialDataLoader
from strategy.factors import (
    PRICE_CALCS, PRICE_FACTORS, PRICE_FACTOR_LABELS,
    calc_low_vol_12m, calc_momentum_12_1, calc_rsi_14_inv,
)


# ─── 工具函数 ───

def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    return (s - s.mean()) / std if std > 0 else s * 0


def _winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    return float(stats.spearmanr(x, y)[0])


# ─── 基本面因子计算 ───

FINANCIAL_FACTORS = ['pe_inv', 'pb_inv', 'roe', 'profit_margin', 'rev_growth']

FINANCIAL_FACTOR_LABELS = {
    'pe_inv': 'PE 倒数',
    'pb_inv': 'PB 倒数',
    'roe': 'ROE',
    'profit_margin': '销售净利率',
    'rev_growth': '营收增长率',
}


def _find_fin_value(fin: dict, candidates: List[str]) -> Optional[float]:
    for k in candidates:
        val = fin.get(k)
        if val is not None and np.isfinite(val):
            return float(val)
    return None


def _compute_ttm_eps(fin_data: dict, loader: FinancialDataLoader,
                      stock: str, as_of: datetime) -> Optional[float]:
    """计算 TTM EPS = latest_annual + latest_cumulative - same_period_last_year"""
    lag_months = 3
    cutoff_ym = as_of.year * 12 + as_of.month - lag_months

    eps = _find_fin_value(fin_data, ['基本每股收益', '每股收益', '每股盈余'])
    if eps is None:
        return None

    if as_of.month >= 4:
        prev_quarter = f'{as_of.year - 1}0930'
        prev_cutoff = (as_of.year - 1) * 12 + 9 - lag_months
    else:
        prev_quarter = f'{as_of.year - 2}0930'
        prev_cutoff = (as_of.year - 2) * 12 + 9 - lag_months

    fin_prev = loader.get_latest_financial(stock, datetime(prev_cutoff // 12, max(1, prev_cutoff % 12), 1))
    if fin_prev is None:
        annual_date = f'{as_of.year - 1}1231'
        prev_cutoff2 = (as_of.year - 1) * 12 + 12 - lag_months
        fin_prev2 = loader.get_latest_financial(stock, datetime(prev_cutoff2 // 12, 1, 1))
        if fin_prev2:
            eps_prev_annual = _find_fin_value(fin_prev2, ['基本每股收益', '每股收益', '每股盈余'])
            if eps_prev_annual is not None and eps is not None:
                return (eps + eps_prev_annual - eps)  # fallback to latest annual
        return eps
    eps_prev = _find_fin_value(fin_prev, ['基本每股收益', '每股收益', '每股盈余'])
    if eps_prev is None:
        return eps

    return eps  # simplified: use latest EPS value directly


def calc_pe_inv(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    """PE 倒数 = EPS / Price, 值越高越便宜"""
    fin = kwargs.get('financial_data')
    loader = kwargs.get('fin_loader')
    stock = kwargs.get('stock')
    if fin is None or loader is None or stock is None:
        return None
    close_val = close[close.index <= as_of].iloc[-1] if len(close[close.index <= as_of]) > 0 else None
    if close_val is None or close_val <= 0:
        return None
    eps = _find_fin_value(fin, ['基本每股收益', '每股收益', '每股盈余'])
    if eps is None or eps <= 0:
        return None
    return float(eps) / float(close_val)


def calc_pb_inv(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    """PB 倒数 = BVPS / Price, 值越高越便宜"""
    fin = kwargs.get('financial_data')
    if fin is None:
        return None
    close_val = close[close.index <= as_of].iloc[-1] if len(close[close.index <= as_of]) > 0 else None
    if close_val is None or close_val <= 0:
        return None
    bvps = _find_fin_value(fin, ['每股净资产', '每股净资产_最新股数', '摊薄每股净资产_期末股数'])
    if bvps is None or bvps <= 0:
        return None
    return float(bvps) / float(close_val)


def calc_roe(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    """ROE, 越高越好"""
    fin = kwargs.get('financial_data')
    if fin is None:
        return None
    roe = _find_fin_value(fin, ['净资产收益率(ROE)', '净资产收益率', '摊薄净资产收益率'])
    if roe is None or not np.isfinite(roe) or abs(roe) > 200:
        return None
    return float(roe)


def calc_profit_margin(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    """销售净利率, 越高越好"""
    fin = kwargs.get('financial_data')
    if fin is None:
        return None
    pm = _find_fin_value(fin, ['销售净利率', '营业利润率'])
    if pm is None or not np.isfinite(pm) or abs(pm) > 200:
        return None
    return float(pm)


def calc_rev_growth(close: pd.Series, as_of: datetime, **kwargs) -> Optional[float]:
    """营业总收入增长率, 越高越好"""
    fin = kwargs.get('financial_data')
    if fin is None:
        return None
    g = _find_fin_value(fin, ['营业总收入增长率', '归属母公司净利润增长率'])
    if g is None or not np.isfinite(g) or abs(g) > 1000:
        return None
    return float(g)


FUNDAMENTAL_CALCS = {
    'pe_inv': calc_pe_inv,
    'pb_inv': calc_pb_inv,
    'roe': calc_roe,
    'profit_margin': calc_profit_margin,
    'rev_growth': calc_rev_growth,
}


ALL_FACTORS = PRICE_FACTORS + FINANCIAL_FACTORS
ALL_LABELS = {**PRICE_FACTOR_LABELS, **FINANCIAL_FACTOR_LABELS}


# ─── 因子模型 ───

@dataclass
class PeriodSnapshot:
    """单个调仓时点的数据快照"""
    date: datetime
    stocks: List[str]                     # 当期有数据的股票
    factor_df: pd.DataFrame               # index=stock, columns=factor, values=zscore
    forward_ret: pd.Series                # index=stock, forward return to next period
    factor_returns: np.ndarray            # [K] 因子收益率 (from OLS)
    residual_var: float                   # average residual variance


class FactorModel:
    """多因子模型: 因子计算 + IC/IR + 信号合成 + Rank 打分加权"""

    def __init__(self, factors: List[str] = None, fin_loader: FinancialDataLoader = None):
        self.factors = factors or ALL_FACTORS
        self.fin_loader = fin_loader
        self.calcs = {}
        for f in self.factors:
            if f in PRICE_CALCS:
                self.calcs[f] = PRICE_CALCS[f]
            elif f in FUNDAMENTAL_CALCS:
                self.calcs[f] = FUNDAMENTAL_CALCS[f]

    def compute_factors(self, market_data: Dict[str, pd.DataFrame],
                        as_of: datetime) -> pd.DataFrame:
        """计算所有股票的因子值, 返回 DataFrame [stock, factor]"""
        rows = {}
        for stock, md in market_data.items():
            if md is None or md.empty:
                continue
            close = md['close']
            fin = None
            if self.fin_loader is not None:
                fin = self.fin_loader.get_latest_financial(stock, as_of)
            vals = {}
            for fname, fn in self.calcs.items():
                v = fn(close, as_of, financial_data=fin,
                       fin_loader=self.fin_loader, stock=stock)
                if v is not None and np.isfinite(v):
                    vals[fname] = v
            if len(vals) >= 3:
                rows[stock] = vals
        df = pd.DataFrame.from_dict(rows, orient='index')
        for f in self.factors:
            if f not in df.columns:
                continue
            s = _winsorize(df[f].dropna())
            df[f] = _zscore(s)
        return df.dropna(how='any')

    @staticmethod
    def estimate_factor_returns(factor_df: pd.DataFrame,
                                 forward_ret: pd.Series
                                 ) -> Tuple[np.ndarray, float]:
        """横截面回归: R_i = α + Σ_k β_{k,i} × f_k + ε_i

        返回 (因子收益率向量, 平均残差方差)
        """
        common = factor_df.index.intersection(forward_ret.dropna().index)
        if len(common) < len(factor_df.columns) * 5:
            return np.zeros(len(factor_df.columns)), 0.01

        X = factor_df.loc[common].values
        y = forward_ret.loc[common].values
        X = np.column_stack([np.ones(len(X)), X])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            return beta[1:], float(np.var(residuals))
        except np.linalg.LinAlgError:
            return np.zeros(len(factor_df.columns)), 0.01

    @staticmethod
    def compute_ic_history(warmup: List[PeriodSnapshot], factors: List[str]
                           ) -> Dict[str, List[float]]:
        """计算每个因子的横截面 Rank IC 时间序列

        对每个 time t: IC_t[k] = Spearman(factor_k_values, forward_returns)
        """
        ic_history = {f: [] for f in factors}
        for snap in warmup:
            if snap.forward_ret is None or snap.forward_ret.empty:
                continue
            for i, f in enumerate(factors):
                vals = snap.factor_df[f]
                rets = snap.forward_ret.reindex(vals.index).dropna()
                common = vals.index.intersection(rets.index)
                if len(common) < 10:
                    continue
                ic = _spearmanr(vals.loc[common].values, rets.loc[common].values)
                ic_history[f].append(ic)
        return ic_history

    @staticmethod
    def compute_ir(ic_history: Dict[str, List[float]]) -> Dict[str, float]:
        """IR = mean(IC) / std(IC) 对每个因子"""
        ir = {}
        for f, vals in ic_history.items():
            vals = [v for v in vals if np.isfinite(v) and abs(v) < 1]
            if len(vals) >= 3:
                m, s = np.mean(vals), np.std(vals)
                ir[f] = m / s if s > 0 else 0.0
            else:
                ir[f] = 0.0
        return ir

    @staticmethod
    def synthesize_signal(ir: Dict[str, float],
                          factor_df: pd.DataFrame) -> np.ndarray:
        """IC_IR 加权合成预测信号

        S_i = Σ_k IR_k × zscore(F_{k,i})
        """
        signals = np.zeros(len(factor_df))
        for f, ir_val in ir.items():
            if f not in factor_df.columns:
                continue
            signals += ir_val * factor_df[f].values
        return signals

    @staticmethod
    def build_covariance(warmup: List[PeriodSnapshot],
                          factor_df: pd.DataFrame) -> np.ndarray:
        """结构化协方差: Σ = B @ Σ_f @ B^T + diag(D)"""
        stocks = factor_df.index.tolist()
        n, k = len(stocks), len(factor_df.columns)

        if len(warmup) < 3:
            return np.eye(n) * 0.02 ** 2

        # factor return time series
        f_ret_list = []
        resid_vars = []
        for snap in warmup:
            if snap.factor_returns is not None and len(snap.factor_returns) == k:
                f_ret_list.append(snap.factor_returns)
                resid_vars.append(snap.residual_var)
        if len(f_ret_list) < 3:
            return np.eye(n) * 0.02 ** 2

        F = np.array(f_ret_list)
        Sigma_f = np.cov(F, rowvar=False) + np.eye(k) * 1e-8

        B = factor_df.values
        avg_resid = np.mean(resid_vars) if resid_vars else 0.02 ** 2

        return B @ Sigma_f @ B.T + np.eye(n) * max(avg_resid, 1e-8)


# ─── 回测引擎 ───

@dataclass
class BacktestResult:
    weights_history: List[dict] = field(default_factory=list)
    daily_nav: pd.Series = field(default_factory=pd.Series)
    performance: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)


class MultiFactorBacktest:
    """多因子策略回测引擎"""

    def __init__(self, stocks: List[str], factors: List[str] = None,
                 lambda_risk: float = 8.0, warmup_periods: int = 8,
                 fin_loader: FinancialDataLoader = None):
        self.fin_loader = fin_loader
        self.factor_model = FactorModel(factors, fin_loader=fin_loader)
        self.stocks = stocks
        self.lambda_risk = lambda_risk
        self.warmup_periods = warmup_periods
        self.factors = factors or ALL_FACTORS

    def _get_market_data(self, data_loader, date, stocks):
        """加载该日期所有股票的行情"""
        md = {}
        for s in stocks:
            df = data_loader(s, date)
            if df is not None and len(df) > 250:
                md[s] = df
        return md

    def run(self, rebalance_dates: List[datetime],
            data_loader: Callable) -> BacktestResult:
        result = BacktestResult()
        warmup: List[PeriodSnapshot] = []
        ic_history: Dict[str, List[float]] = {f: [] for f in self.factors}
        ir: Dict[str, float] = {f: 0.0 for f in self.factors}
        current_weights: Dict[str, float] = {}

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

            current_weights = dict(zip(active_stocks, w))
            result.weights_history.append({
                'date': rb_date,
                'weights': current_weights,
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
