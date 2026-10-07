"""Point-in-time data accessors for the dividend-growth stock pool."""

from typing import Optional
import numpy as np
import pandas as pd
from data.pool import load_balance, load_dividend, load_fin_summary, load_raw


def raw_close(code: str, as_of: pd.Timestamp) -> Optional[float]:
    df = load_raw(code)
    if df is None or df.empty:
        return None
    hist = df[df.index <= pd.Timestamp(as_of)]
    return float(hist['close'].iloc[-1]) if not hist.empty else None


def dividend_yield(code: str, as_of: pd.Timestamp) -> Optional[float]:
    df = load_dividend(code)
    if df is None or '实施方案公告日期' not in df.columns:
        return None
    as_of = pd.Timestamp(as_of)
    ann = df[pd.to_datetime(df['实施方案公告日期'], errors='coerce').notna()].copy()
    ann['公告日'] = pd.to_datetime(ann['实施方案公告日期'], errors='coerce')
    ann['派息比例'] = pd.to_numeric(ann['派息比例'], errors='coerce')
    annual = ann[(ann['公告日'] <= as_of)
                 & ann['报告时间'].astype(str).str.endswith('年报')
                 & (ann['派息比例'] > 0)]
    if annual.empty:
        return None
    latest = annual.sort_values('公告日').iloc[-1]
    if latest['公告日'] < as_of - pd.Timedelta(days=550):
        return None
    price = raw_close(code, as_of)
    return latest['派息比例'] / 10.0 / price if price and price > 0 else None


def ttm_eps_profit_balance(code: str, as_of: pd.Timestamp) -> Optional[float]:
    pro, bal = load_profit(code), load_balance(code)
    if pro is None or bal is None or '归属于母公司所有者的净利润' not in pro.columns:
        return None
    as_of = pd.Timestamp(as_of)
    pf = pro[pro['公告日期'].fillna(pd.Timestamp.max) <= as_of].sort_values('报告日')
    if pf.empty:
        return None
    latest = pf.iloc[-1]
    cum = latest['归属于母公司所有者的净利润']
    prev_annual = pf[(pf['报告日'].dt.month == 12) & (pf['报告日'] < latest['报告日'])]
    prior = pf[pf['报告日'] == latest['报告日'] - pd.DateOffset(years=1)]
    if not np.isfinite(cum) or prev_annual.empty or prior.empty:
        return None
    ttm = cum + prev_annual.iloc[-1]['归属于母公司所有者的净利润'] - prior.iloc[-1]['归属于母公司所有者的净利润']
    bd = bal[bal['公告日期'].fillna(pd.Timestamp.max) <= as_of].sort_values('报告日')
    if bd.empty or '实收资本(或股本)' not in bd.columns:
        return None
    shares = float(bd.iloc[-1]['实收资本(或股本)'])
    return ttm / shares if np.isfinite(shares) and shares > 0 else None


def ttm_eps_fin(code: str, as_of: pd.Timestamp) -> Optional[float]:
    fin = load_fin_summary(code)
    if fin is None or '指标' not in fin.columns:
        return None
    row = fin[fin['指标'] == '基本每股收益']
    if row.empty:
        return None
    values = row.iloc[0, 2:].astype(object)
    values.index = pd.to_datetime(values.index, format='%Y%m%d', errors='coerce')
    values = pd.to_numeric(values, errors='coerce')
    values = values[(values.index.notna()) & (values.index <= pd.Timestamp(as_of))].sort_index()
    if values.empty:
        return None
    annual = values[values.index.month == 12]
    prior = annual[annual.index < values.index[-1]]
    previous = values.loc[values.index[-1] - pd.DateOffset(years=1)] if values.index[-1] - pd.DateOffset(years=1) in values.index else np.nan
    return float(values.iloc[-1] + prior.iloc[-1] - previous) if not prior.empty and np.isfinite(previous) else None


def eps_ttm(code: str, as_of: pd.Timestamp) -> Optional[float]:
    value = ttm_eps_profit_balance(code, as_of)
    return ttm_eps_fin(code, as_of) if value is None else value


def pe_ttm(code: str, as_of: pd.Timestamp) -> Optional[float]:
    price, eps = raw_close(code, as_of), eps_ttm(code, as_of)
    return price / eps if price and eps and eps > 0 else None
