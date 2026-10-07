"""格雷厄姆「大型财务稳健」+ 红利 池筛选 (point-in-time)

数据源:
  - data/financial/{code}_fin.csv   同花顺财务摘要宽表 (指标 × 报告期)
  - data/dividend/{code}_dividend.csv  分红送配历史 (巨潮)
  - data/{code}_market.csv          前复权行情

Point-in-time 惯例:
  - 年报 (12-31 报告期) 视为次年 4-30 披露
  - 分红以「实施方案公告日期」为准可见
  - 市值 = 股价 × (股东权益 ÷ 每股净资产)  [用最新已披露年报]
"""
from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
from screener.graham_data_access import (
    DIV_CACHE as _DIV_CACHE, FIN_CACHE as _FIN_CACHE, MKT_CACHE as _MKT_CACHE,
    indicator_series as _indicator_series, parse_value,
    read_dividend as _read_dividend, read_fin as _read_fin, read_market as _read_market,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIN_DIR = os.path.join(DATA_DIR, 'financial')
DIV_DIR = os.path.join(DATA_DIR, 'dividend')

ANNUAL_REPORT_LAG = pd.DateOffset(months=4, days=30)  # 12-31 报告期 → 次年 4-30

# ── 本地只读文件缓存 (同一进程内复用, 避免每次筛选重复读盘) ──
_GH_CACHE: Dict[Tuple[str, pd.Timestamp], 'GrahamHolding'] = {}


class GrahamHolding:
    """单只股票在 as_of 日的基本面状态 (point-in-time)"""

    def __init__(self, code: str, as_of: pd.Timestamp):
        self.code = code
        self.as_of = pd.Timestamp(as_of)
        self.valid = False
        self.reason = ''
        self.market_cap = np.nan
        self.current_ratio = np.nan
        self.debt_ratio = np.nan
        self.last_5y_profits: List[float] = []
        self.dividend_yield = np.nan
        self._load()

    # ─── 财务 ───
    def _load(self):
        df_full = _read_fin(self.code)
        if df_full.empty or '指标' not in df_full.columns:
            self.reason = '无财务数据'
            return

        cols = df_full.columns[2:]
        dates = pd.to_datetime(cols, format='%Y%m%d', errors='coerce')
        mask = pd.notna(dates) & (dates + ANNUAL_REPORT_LAG <= self.as_of)
        disclosed_cols = [c for c, ok in zip(cols, mask) if ok]
        if not disclosed_cols:
            self.reason = '无已披露财报'
            return
        # 列序可能是降序, 统一按日期升序
        asc = sorted(zip(pd.to_datetime(disclosed_cols, format='%Y%m%d'), disclosed_cols))
        disclosed_cols = [c for _, c in asc]
        df = df_full[['指标'] + disclosed_cols].set_index('指标')
        df = df[~df.index.duplicated(keep='first')]

        # 最新披露年报 (12-31 报告期)
        annual_cols = [c for c in disclosed_cols
                       if pd.to_datetime(c, format='%Y%m%d').month == 12]
        if not annual_cols:
            self.reason = '无年报'
            return
        latest_annual = annual_cols[-1]

        # 市值 = 股价 × (净资产 / 每股净资产)
        equity = parse_value(df.loc['股东权益合计(净资产)', latest_annual])
        bps = parse_value(df.loc['每股净资产', latest_annual])
        if equity and bps and bps > 0:
            shares = equity / bps
            price = self._close_price()
            if price and price > 0:
                self.market_cap = shares * price

        self.current_ratio = parse_value(df.loc['流动比率', latest_annual]) \
            if '流动比率' in df.index else np.nan
        self.debt_ratio = parse_value(df.loc['资产负债率', latest_annual]) \
            if '资产负债率' in df.index else np.nan

        # 连续 5 年年报归母净利润
        if '归母净利润' in df.index:
            self.last_5y_profits = [
                parse_value(df.loc['归母净利润', c])
                for c in annual_cols[-5:]
            ]

        self._load_dividend()
        self.valid = True

    # ─── 分红 ───
    def _load_dividend(self):
        path = os.path.join(DIV_DIR, f'{self.code}_dividend.csv')
        if not os.path.exists(path):
            return
        df = _read_dividend(self.code)
        if df.empty or '实施方案公告日期' not in df.columns:
            return
        df['公告日'] = pd.to_datetime(df['实施方案公告日期'], errors='coerce')
        df['派息比例'] = pd.to_numeric(df['派息比例'], errors='coerce')
        # 最近一次已宣告的年度分红 (公告日≤as_of), 18 个月内才算有效
        annual = df[
            df['公告日'].notna()
            & (df['公告日'] <= self.as_of)
            & df['报告时间'].astype(str).str.endswith('年报')
            & (df['派息比例'].notna()) & (df['派息比例'] > 0)
        ]
        if annual.empty:
            return
        latest = annual.sort_values('公告日').iloc[-1]
        if latest['公告日'] < self.as_of - pd.Timedelta(days=550):
            return  # 长期未分红
        dps = latest['派息比例'] / 10.0  # 每10股派X元 → 每股
        price = self._close_price()
        if price and price > 0:
            self.dividend_yield = dps / price
        self.recent_dividend = True

    # ─── 行情 ───
    def _close_price(self) -> Optional[float]:
        df = _read_market(self.code)
        df = df[df.index <= self.as_of]
        if df.empty:
            return None
        return float(df['close'].iloc[-1])


class GrahamScreener:
    """格雷厄姆-红利池 (若不启用红利则跳过分红约束)

    条件: 市值≥池中位 | 流动比率≥2 | 资产负债率≤60% | 连续5年盈利 | 5年利润增速>0
    红利用 require_dividend 开关: 近12月有现金分红 且 股息率≥池中位
    """

    def __init__(self, require_dividend: bool = True):
        self.require_dividend = require_dividend

    def run(self, as_of: pd.Timestamp, codes: List[str]) -> Tuple[List[str], Dict]:
        """返回 (通过的 codes, 各股状态详情)"""
        as_of = pd.Timestamp(as_of)
        states: Dict[str, GrahamHolding] = {}
        for code in codes:
            key = (code, as_of)
            g = _GH_CACHE.get(key)
            if g is None:
                g = GrahamHolding(code, as_of)
                _GH_CACHE[key] = g
            if g.valid:
                states[code] = g

        # 市值 / 股息率 的池内中位基准
        caps = pd.Series({c: s.market_cap for c, s in states.items()}).dropna()
        yields = pd.Series({c: s.dividend_yield for c, s in states.items()
                            if getattr(s, 'recent_dividend', False)}).dropna()

        passed, details = [], {}
        for code, g in states.items():
            ok = True
            note = []
            cap_ok = (not caps.empty) and pd.notna(g.market_cap) and g.market_cap >= caps.median()
            cr_ok = pd.notna(g.current_ratio) and g.current_ratio >= 2.0
            dr_ok = pd.notna(g.debt_ratio) and g.debt_ratio <= 60.0  # THS 资产负债率为百分数
            profits = g.last_5y_profits
            eps_ok = len(profits) >= 5 and all(p is not None and p > 0 for p in profits)
            grow_ok = False
            if len(profits) >= 5 and all(p is not None and p > 0 for p in profits):
                grow_ok = (profits[-1] / profits[0] - 1) > 0
            div_ok = False
            if getattr(g, 'recent_dividend', False) and (not yields.empty) \
                    and pd.notna(g.dividend_yield) and g.dividend_yield >= yields.median():
                div_ok = True

            if not cap_ok:
                ok, note = False, note + ['市值']
            if not cr_ok:
                ok, note = False, note + ['流动比率']
            if not dr_ok:
                ok, note = False, note + ['负债率']
            if not eps_ok:
                ok, note = False, note + ['盈利连续性']
            if not grow_ok:
                ok, note = False, note + ['盈利增长']
            if self.require_dividend and not div_ok:
                ok, note = False, note + ['红利']

            details[code] = {
                'market_cap': g.market_cap,
                'current_ratio': g.current_ratio,
                'debt_ratio': g.debt_ratio,
                'dividend_yield': g.dividend_yield,
                'last_5y_profits': g.last_5y_profits,
                'fail': note,
            }
            if ok:
                passed.append(code)
        return passed, details
