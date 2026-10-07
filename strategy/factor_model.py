"""多因子模型：因子计算、IC/IR 估计和信号合成。"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from data.financial import FinancialDataLoader
from strategy.factors import PRICE_CALCS, PRICE_FACTORS
from strategy.fundamental_factors import FINANCIAL_FACTORS, FUNDAMENTAL_CALCS
from backtest.types import PeriodSnapshot


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    return (series - series.mean()) / std if std > 0 else series * 0


def _winsorize(series: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    return series.clip(series.quantile(lo), series.quantile(hi))


def _spearmanr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 4:
        return 0.0
    return float(stats.spearmanr(left, right)[0])


class FactorModel:
    """计算横截面因子并将其合成为选股信号。"""

    def __init__(self, factors: List[str] | None = None,
                 fin_loader: FinancialDataLoader | None = None):
        self.factors = factors or PRICE_FACTORS + FINANCIAL_FACTORS
        self.fin_loader = fin_loader
        self.calcs = {
            **{name: PRICE_CALCS[name] for name in self.factors if name in PRICE_CALCS},
            **{name: FUNDAMENTAL_CALCS[name] for name in self.factors if name in FUNDAMENTAL_CALCS},
        }

    def compute_factors(self, market_data: Dict[str, pd.DataFrame],
                        as_of: datetime) -> pd.DataFrame:
        rows = {}
        for stock, frame in market_data.items():
            if frame is None or frame.empty:
                continue
            financial = (
                self.fin_loader.get_latest_financial(stock, as_of)
                if self.fin_loader else None
            )
            values = {}
            for name, calculation in self.calcs.items():
                value = calculation(
                    frame["close"], as_of, financial_data=financial,
                    fin_loader=self.fin_loader, stock=stock,
                )
                if value is not None and np.isfinite(value):
                    values[name] = value
            if len(values) >= 3:
                rows[stock] = values

        factors = pd.DataFrame.from_dict(rows, orient="index")
        for name in self.factors:
            if name in factors:
                factors[name] = _zscore(_winsorize(factors[name].dropna()))
        return factors.dropna(how="any")

    @staticmethod
    def estimate_factor_returns(
        factor_df: pd.DataFrame, forward_ret: pd.Series
    ) -> Tuple[np.ndarray, float]:
        common = factor_df.index.intersection(forward_ret.dropna().index)
        if len(common) < len(factor_df.columns) * 5:
            return np.zeros(len(factor_df.columns)), 0.01
        design = np.column_stack([np.ones(len(common)), factor_df.loc[common].values])
        beta = np.linalg.lstsq(design, forward_ret.loc[common].values, rcond=None)[0]
        residuals = forward_ret.loc[common].values - design @ beta
        return beta[1:], float(np.var(residuals))

    @staticmethod
    def compute_ic_history(warmup: List[PeriodSnapshot], factors: List[str]) -> Dict[str, List[float]]:
        history = {name: [] for name in factors}
        for snapshot in warmup:
            if snapshot.forward_ret is None or snapshot.forward_ret.empty:
                continue
            for name in factors:
                values = snapshot.factor_df[name]
                returns = snapshot.forward_ret.reindex(values.index).dropna()
                common = values.index.intersection(returns.index)
                if len(common) >= 10:
                    history[name].append(_spearmanr(values.loc[common], returns.loc[common]))
        return history

    @staticmethod
    def compute_ir(ic_history: Dict[str, List[float]]) -> Dict[str, float]:
        result = {}
        for name, values in ic_history.items():
            valid = [value for value in values if np.isfinite(value) and abs(value) < 1]
            result[name] = np.mean(valid) / np.std(valid) if len(valid) >= 3 and np.std(valid) > 0 else 0.0
        return result

    @staticmethod
    def synthesize_signal(ir: Dict[str, float], factor_df: pd.DataFrame) -> np.ndarray:
        signal = np.zeros(len(factor_df))
        for name, weight in ir.items():
            if name in factor_df:
                signal += weight * factor_df[name].values
        return signal

    @staticmethod
    def build_covariance(warmup: List[PeriodSnapshot], factor_df: pd.DataFrame) -> np.ndarray:
        n, k = len(factor_df), len(factor_df.columns)
        if len(warmup) < 3:
            return np.eye(n) * 0.02 ** 2
        returns = [s.factor_returns for s in warmup if len(s.factor_returns) == k]
        residuals = [s.residual_var for s in warmup if len(s.factor_returns) == k]
        if len(returns) < 3:
            return np.eye(n) * 0.02 ** 2
        factor_cov = np.cov(np.array(returns), rowvar=False) + np.eye(k) * 1e-8
        residual = max(np.mean(residuals), 1e-8) if residuals else 0.02 ** 2
        return factor_df.values @ factor_cov @ factor_df.values.T + np.eye(n) * residual


__all__ = ["FactorModel"]
