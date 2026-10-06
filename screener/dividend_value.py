"""高息价值池筛选器 — 4 选 3 (股息率/PE_TTM/ROE连续三年/市值) — point-in-time 无泄漏

口径与 screener/pool_screener.py 一致:
  - 财报/分红仅使用 公告日期 ≤ as_of 的记录 (公告日期缺失视为可见)
  - 股息率 = 最近一次已公告年度每股派息 ÷ 不复权收盘价 (18 个月内有效)
  - PE_TTM = 不复权收盘价 ÷ TTM 每股归母净利 (公告可见)
  - ROE 连续三年 = 最近 3 个已公告年报 (报告日12月) 的 归母净利/归母权益 均 > roe_min
  - 市值 = 实收资本(或股本) × 不复权收盘价 (亿元)

候选域: 全部有行情+分红+利润+资产负债数据的股票, 每月按市值取 Top top_n 再筛 4 选 3。
"""
import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from screener.pool_screener import (
    load_balance,
    load_profit,
    raw_close,
    dividend_yield,
    pe_ttm,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIN_DIR = os.path.join(DATA_DIR, 'financial')
DIV_DIR = os.path.join(DATA_DIR, 'dividend')
META_DIR = os.path.join(DATA_DIR, 'meta')

_NAME_MAP: Optional[Dict[str, str]] = None


def _load_names() -> Dict[str, str]:
    global _NAME_MAP
    if _NAME_MAP is None:
        p = os.path.join(META_DIR, 'graham_universe.csv')
        try:
            df = pd.read_csv(p, dtype={'code': str})
            _NAME_MAP = dict(zip(df['code'].str.strip(), df['name'].str.strip()))
        except Exception:
            _NAME_MAP = {}
    return _NAME_MAP


def stock_name(code: str) -> str:
    return _load_names().get(str(code), f'股票{code}')


def universe_codes() -> List[str]:
    """有行情 + 分红 + 利润 + 资产负债数据的全部股票 (交集)。"""
    qfq = {f[:-8] for f in os.listdir(DATA_DIR) if f.endswith('_qfq.csv')}
    div = {f[:-13] for f in os.listdir(DIV_DIR) if f.endswith('_dividend.csv')}
    pro = {f[:-11] for f in os.listdir(FIN_DIR) if f.endswith('_profit.csv')}
    bal = {f[:-12] for f in os.listdir(FIN_DIR) if f.endswith('_balance.csv')}
    return sorted(qfq & div & pro & bal)


def market_cap(code: str, as_of: pd.Timestamp) -> Optional[float]:
    """市值 (亿元) = 最新已披露实收资本(股本,元) × 不复权收盘价。"""
    bal = load_balance(code)
    as_of = pd.Timestamp(as_of)
    if bal is None or '实收资本(或股本)' not in bal.columns:
        return None
    bd = bal[bal['公告日期'].fillna(pd.Timestamp.max) <= as_of].sort_values('报告日')
    if bd.empty:
        return None
    shares = pd.to_numeric(bd['实收资本(或股本)'], errors='coerce').iloc[-1]
    if not np.isfinite(shares) or shares <= 0:
        return None
    price = raw_close(code, as_of)
    if price is None or price <= 0:
        return None
    return shares * price / 1e8


def roe_3y(code: str, as_of: pd.Timestamp) -> Optional[tuple]:
    """(最近一年 ROE, 近三年 ROE 列表) 或 None; None 表示近 3 年 ROE 不可判定。"""
    pro = load_profit(code)
    bal = load_balance(code)
    as_of = pd.Timestamp(as_of)
    if pro is None or bal is None:
        return None
    cols = ('归属于母公司所有者的净利润', '归属于母公司股东权益合计')
    if cols[0] not in pro.columns or cols[1] not in bal.columns:
        return None
    mask = (pro['报告日'].dt.month == 12) & (pro['公告日期'].fillna(pd.Timestamp.max) <= as_of)
    ap = pro[mask][['报告日', cols[0]]].dropna(subset=['报告日']).sort_values('报告日')
    mask_b = (bal['报告日'].dt.month == 12) & (bal['公告日期'].fillna(pd.Timestamp.max) <= as_of)
    ab = bal[mask_b][['报告日', cols[1]]].dropna(subset=['报告日']).sort_values('报告日')
    merged = ap.merge(ab, on='报告日', how='inner').sort_values('报告日')
    if len(merged) < 3:
        return None
    last3 = merged.tail(3)
    npf = pd.to_numeric(last3[cols[0]], errors='coerce')
    eq = pd.to_numeric(last3[cols[1]], errors='coerce')
    roes = npf / eq.replace(0, np.nan)
    if roes.isna().any() or (roes <= 0).any():
        return None
    values = [float(v) for v in roes]
    return (values[-1], values)


def factor_detail(code: str, as_of: pd.Timestamp,
                  div_min: float = 0.04, pe_max: float = 20.0,
                  roe_min: float = 0.10, mc_min_yi: float = 500.0) -> Optional[dict]:
    """单个标的 4 因子 + 逐项判定 + 是否通过 (≥3/4)。"""
    dy = dividend_yield(code, as_of)
    pe = pe_ttm(code, as_of)
    r3 = roe_3y(code, as_of)
    mc = market_cap(code, as_of)
    if all(v is None for v in (dy, pe, r3, mc)):
        return None
    c_div = dy is not None and dy > div_min
    c_pe = pe is not None and pe < pe_max
    c_roe = r3 is not None and all(v > roe_min for v in r3[1])
    c_mc = mc is not None and mc > mc_min_yi
    n_pass = int(c_div) + int(c_pe) + int(c_roe) + int(c_mc)
    return {
        'name': stock_name(code),
        'div_yield': dy,
        'pe_ttm': pe,
        'roe_latest': r3[0] if r3 else None,
        'roe_3y': r3[1] if r3 else None,
        'mc_yi': mc,
        'c_div': c_div, 'c_pe': c_pe, 'c_roe': c_roe, 'c_mc': c_mc,
        'n_pass': n_pass,
        'pass': n_pass >= 3,
    }


def build_value_screener(top_n: int = 300, div_min: float = 0.04,
                         pe_max: float = 20.0, roe_min: float = 0.10,
                         mc_min_yi: float = 500.0):
    """返回 sc(as_of) -> {code: detail}; 每月按市值取 Top top_n 再筛 4 选 3。

    结果按 (as_of, 结果字典) 缓存, 回测中重复调用为 O(1) 查表。
    """
    codes = sorted(universe_codes())

    @lru_cache(maxsize=None)
    def _screen(as_of: str) -> Dict[str, dict]:
        ts = pd.Timestamp(as_of)
        mcs = {c: mc for c in codes if (mc := market_cap(c, ts)) is not None}
        top = [c for c, _ in sorted(mcs.items(), key=lambda kv: -kv[1])[:top_n]]
        out = {}
        for code in top:
            d = factor_detail(code, ts, div_min=div_min, pe_max=pe_max,
                              roe_min=roe_min, mc_min_yi=mc_min_yi)
            if d is not None:
                out[code] = d
        return out

    def sc(as_of) -> Dict[str, dict]:
        return _screen(str(pd.Timestamp(as_of)))

    return sc


__all__ = ['universe_codes', 'market_cap', 'roe_3y', 'factor_detail',
           'build_value_screener', 'stock_name']