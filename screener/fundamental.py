"""财务面选股筛选器"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from screener.base import Screener, FilterResult
from data.manager import DataManager


class FinancialDataCache:
    """财务数据缓存管理器（避免重复下载）"""
    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._dm = DataManager()

    def get(self, code: str) -> pd.DataFrame:
        if code not in self._cache:
            self._cache[code] = self._dm.get_financial_data(code)
        return self._cache[code]

    def prefetch(self, codes: list):
        for code in codes:
            _ = self.get(code)


class FundamentalScreener(Screener):
    """财务因子筛选器

    基于连续 N 年的财务指标过滤股票。
    默认策略：
      - 经营现金流净额 / 净利润 > 80%（连续3年）
      - 资本开支 / 经营现金流净额 < 50%（连续3年）
      - 净利润增速 ≥ 0
    """

    def __init__(
        self,
        cashflow_to_profit_min: float = 0.8,
        capex_to_cashflow_max: float = 0.5,
        profit_growth_min: float = 0.0,
        consecutive_years: int = 3,
    ):
        super().__init__()
        self.cashflow_to_profit_min = cashflow_to_profit_min
        self.capex_to_cashflow_max = capex_to_cashflow_max
        self.profit_growth_min = profit_growth_min
        self.consecutive_years = consecutive_years
        self._data = FinancialDataCache()
        self._result_df: Optional[pd.DataFrame] = None

    def run(self, as_of_date, pool: Optional[List[str]] = None) -> FilterResult:
        as_of = pd.Timestamp(as_of_date)

        if pool is None or len(pool) == 0:
            return FilterResult()

        all_rows = []
        for code in pool:
            try:
                df = self._data.get(code)
            except Exception:
                continue
            if df.empty:
                continue
            usable = df[df['publish_date'] <= as_of].copy()
            if len(usable) >= self.consecutive_years:
                usable['code'] = code
                all_rows.append(usable)

        if not all_rows:
            return FilterResult()

        combined = pd.concat(all_rows, ignore_index=True)
        self._result_df = combined
        return self._filter(combined)

    def _filter(self, df: pd.DataFrame) -> FilterResult:
        codes = df['code'].unique()
        passed = []
        info = {}

        for code in codes:
            rows = df[df['code'] == code].sort_values('report_date')
            if len(rows) < self.consecutive_years:
                continue

            latest = rows.tail(self.consecutive_years)
            ratios_cp = latest['operating_cashflow'] / latest['net_profit'].replace(0, np.nan)
            ratios_cc = latest['capex'] / latest['operating_cashflow'].replace(0, np.nan)

            if ratios_cp.isna().any() or ratios_cc.isna().any():
                continue

            cp_ok = (ratios_cp > self.cashflow_to_profit_min).all()
            cc_ok = (ratios_cc < self.capex_to_cashflow_max).all()

            profits = latest['net_profit'].values
            growth_ok = True
            if self.profit_growth_min is not None and len(profits) >= 2:
                growth_ok = all(
                    (profits[i] / profits[i - 1] - 1) >= self.profit_growth_min
                    for i in range(1, len(profits))
                    if profits[i - 1] != 0
                )

            if cp_ok and cc_ok and growth_ok:
                passed.append(code)
                info[code] = {
                    'ratios_cp': [round(float(r), 4) for r in ratios_cp],
                    'ratios_cc': [round(float(r), 4) for r in ratios_cc],
                    'latest_profit': float(profits[-1]),
                }

        return FilterResult(codes=passed, info=info)


class SimpleFinancialScreener(FundamentalScreener):
    """简化版：直接从预计算好的评分 DataFrame 中过滤"""

    def __init__(self, df_ratios: pd.DataFrame):
        super().__init__()
        self.df = df_ratios

    def run(self, as_of_date, pool: Optional[List[str]] = None) -> FilterResult:
        df = self.df.copy()
        if pool:
            df = df[df['code'].isin(pool)]

        for col in ['cashflow_to_profit_1y', 'cashflow_to_profit_2y', 'cashflow_to_profit_3y']:
            if col in df.columns:
                df = df[df[col] > self.cashflow_to_profit_min]

        for col in ['capex_to_cashflow_1y', 'capex_to_cashflow_2y', 'capex_to_cashflow_3y']:
            if col in df.columns:
                df = df[df[col] < self.capex_to_cashflow_max]

        for col in ['profit_growth_1y', 'profit_growth_2y']:
            if col in df.columns:
                df = df[df[col] >= self.profit_growth_min]

        return FilterResult(
            codes=df['code'].tolist(),
            info=df.set_index('code').to_dict(orient='index'),
        )
