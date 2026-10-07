"""高息成长池 point-in-time 筛选取证 (28 只 A 股)

数据源与口径:
  - data/{code}_qfq.csv       前复权行情 (688036 无 qfq, 其 market.csv 已复权)
  - data/{code}_market.csv    不复权行情 (用于股息率/PE 的价格)
  - data/financial/{code}_profit.csv / {code}_balance.csv  利润/资产负债 (公告日可见)
  - data/financial/{code}_fin.csv     同花顺财务摘要 (仅 688036 用于 PE 兜底)
  - data/dividend/{code}_dividend.csv 分红送配 (实施方案公告日期可见)

Point-in-time 惯例 (与 strategy/graham_dodd.py 一致):
  - 财报/分红以「公告日期 ≤ as_of」为准可见
  - 股息率 = 最近一次已公告年度每股派息 ÷ 不复权收盘价 (18 个月内有效)
  - PE_TTM = 不复权收盘价 ÷ TTM 每股归母净利 (公告可见)
  - MA120 / 波动率基于前复权价
"""
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dataload.readers import (
    load_balance, load_dividend, load_fin_summary, load_profit, load_qfq, load_raw,
)

# ── 股票池 (28 只 A 股; 远东宏信/巨子生物为港股, 无本地数据, 剔除) ──
POOL: Dict[str, str] = {
    '000651': '格力电器', '000333': '美的集团', '688036': '传音控股',
    '600941': '中国移动',
    '000895': '双汇发展', '600887': '伊利股份',
    '600938': '中国海油', '601088': '中国神华', '000937': '冀中能源',
    '600096': '云天化', '601233': '桐昆股份', '600273': '嘉化能源',
    '600219': '南山铝业',
    '300498': '温氏股份', '000048': '京基智农', '002749': '国光股份',
    '600377': '宁沪高速', '600012': '皖通高速', '601766': '中国中车',
    '600023': '浙能电力',
    '601318': '中国平安', '600901': '江苏金租', '600036': '招商银行',
    '601229': '上海银行',
    '603529': '爱玛科技', '600000': '浦发银行', '603508': '思维列控',
    '301004': '嘉益股份',
}

# 预计算信号序列缓存 (按日索引, 避免回测中逐日重算)
_signal_frames: Dict[str, Optional[pd.DataFrame]] = {}


# ── 收盘价 (不复权, 仅 D 日有成交时返回) ──
def raw_close(code: str, as_of: pd.Timestamp) -> Optional[float]:
    df = load_raw(code)
    if df is None or df.empty:
        return None
    hist = df[df.index <= pd.Timestamp(as_of)]
    return float(hist['close'].iloc[-1]) if not hist.empty else None


# ── 股息率 (单日) ──
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


# ── TTM EPS (单日) ──
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


# ── MA120 偏差 (前复权) ──
def ma120_dev(code: str, as_of: pd.Timestamp) -> Optional[float]:
    df = load_qfq(code)
    if df is None or df.empty:
        return None
    hist = df[df.index <= pd.Timestamp(as_of)]
    if len(hist) < 120 + 1:
        return None
    close = float(hist['close'].iloc[-1])
    ma120 = float(hist['close'].tail(120).mean())
    if ma120 <= 0:
        return None
    return (close - ma120) / ma120


# ── 120 日年化波动率 (前复权日收益) ──
def vol120(code: str, as_of: pd.Timestamp) -> Optional[float]:
    df = load_qfq(code)
    if df is None or df.empty:
        return None
    hist = df[df.index <= pd.Timestamp(as_of)]
    ret = hist['close'].pct_change().dropna()
    if len(ret) < 60:
        return None
    return float(ret.tail(120).std() * np.sqrt(252))


def pool_vol_median(as_of: pd.Timestamp) -> Optional[float]:
    """池内 120 日波动率中位数 (用于补仓档位分档)"""
    vals = [v for c in POOL if (v := vol120(c, as_of)) is not None]
    return float(np.median(vals)) if vals else None


# ── 一次性筛查 (信号日) ──
def screen_pool(as_of: pd.Timestamp,
                div_min: float = 0.03, pe_max: float = 20.0,
                dev_buy: float = -0.12) -> List[Tuple[str, Dict]]:
    """返回 (code, detail) 列表, detail 含 股息率/PE_TTM/MA120偏差/波动率档位"""
    as_of = pd.Timestamp(as_of)
    out = []
    med = pool_vol_median(as_of)
    for code, name in POOL.items():
        dy = dividend_yield(code, as_of)
        pe = pe_ttm(code, as_of)
        dev = ma120_dev(code, as_of)
        vol = vol120(code, as_of)
        thr = add_threshold_from(vol, med)
        ok = (dy is not None and dy > div_min
              and pe is not None and pe < pe_max
              and dev is not None and dev <= dev_buy)
        out.append((code, {
            'name': name, 'div_yield': dy, 'pe_ttm': pe, 'dev': dev,
            'vol120': vol, 'add_thr': thr, 'pass': ok,
        }))
    return out


def add_threshold_from(vol: Optional[float], med: Optional[float]) -> float:
    """补仓档位: 波动率 > 池中位数 → -20cm (保守), 否则 → -10cm"""
    if vol is None or med is None:
        return 0.20
    return 0.20 if vol > med else 0.10


def build_pool_screener(div_min: float = 0.03, pe_max: float = 20.0,
                        dev_buy: float = -0.12) -> Callable[[pd.Timestamp], Dict[str, Dict]]:
    """预计算全部个股信号帧与池中位数波动率序列, 返回每日 O(1) 查表 screener。

    返回值: {code: {'name','div_yield','pe_ttm','dev','vol120','add_thr','pass'}}
    """
    frames = {c: signal_frame(c) for c in POOL}
    frames = {c: f for c, f in frames.items() if f is not None}
    # 池中位数波动率序列 (全部可用交易日并集)
    all_vol = pd.DataFrame({c: f['vol120'] for c, f in frames.items()})
    med_vol = all_vol.median(axis=1)

    def sc(as_of):
        as_of = pd.Timestamp(as_of)
        med = med_vol.get(as_of, np.nan) if as_of in med_vol.index else np.nan
        out = {}
        for code, f in frames.items():
            if as_of not in f.index:
                continue
            r = f.loc[as_of]
            dy = r['div_yield'] if pd.notna(r['div_yield']) else None
            pe = r['pe_ttm'] if pd.notna(r['pe_ttm']) else None
            dev = r['dev'] if pd.notna(r['dev']) else None
            vol = r['vol120'] if pd.notna(r['vol120']) else None
            thr = add_threshold_from(vol, med)
            ok = (dy is not None and dy > div_min
                  and pe is not None and pe < pe_max
                  and dev is not None and dev <= dev_buy)
            out[code] = {
                'name': POOL[code], 'div_yield': dy, 'pe_ttm': pe, 'dev': dev,
                'vol120': vol, 'add_thr': thr, 'pass': ok,
            }
        return out

    return sc


# ── 预计算信号序列 (每日索引, 回测用; 一次构建, 全程查表) ──
def signal_frame(code: str) -> Optional[pd.DataFrame]:
    """返回该股逐日 DataFrame: index=交易日, columns=[div_yield,pe_ttm,dev,vol120]"""
    if code in _signal_frames:
        return _signal_frames[code]

    q = load_qfq(code)
    raw = load_raw(code)
    if q is None or q.empty:
        _signal_frames[code] = None
        return None
    days = q.index
    out = pd.DataFrame(index=days)

    # 股息率: 仅在公告日更新的 DPS, 前向填充, 再 ÷ 不复权收盘价
    div = load_dividend(code)
    if div is not None and '实施方案公告日期' in div.columns:
        ann = pd.to_datetime(div['实施方案公告日期'], errors='coerce')
        dps = pd.to_numeric(div['派息比例'], errors='coerce')
        is_annual = div['报告时间'].astype(str).str.endswith('年报')
        mask = ann.notna() & is_annual & (dps.notna()) & (dps > 0)
        s = pd.Series(dps[mask].values / 10.0, index=ann[mask]).sort_index()
        # 18 个月内有效: 仅保留公告日后 550 天内
        s = s[~s.index.duplicated(keep='last')]
        if not s.empty:
            effective = s.reindex(days.union(s.index)).ffill()
            effective[effective.index > s.index[-1] + pd.Timedelta(days=550)] = np.nan
            rawp = raw['close'].reindex(effective.index) if raw is not None else None
            if rawp is not None:
                out['div_yield'] = effective / rawp
    if 'div_yield' not in out.columns:
        out['div_yield'] = np.nan

    # PE_TTM: eps 为公告步进序列, 前向填充, ÷ 不复权收盘价
    es = _eps_ttm_series(code, days)
    if es.notna().any():
        rawp = raw['close'].reindex(days) if raw is not None else None
        if rawp is not None:
            out['pe_ttm'] = rawp / es
    if 'pe_ttm' not in out.columns:
        out['pe_ttm'] = np.nan

    # MA120 偏差
    out['dev'] = _ma120_dev_series(code, days)

    # 120 日年化波动率
    ret = q['close'].pct_change()
    out['vol120'] = ret.rolling(120).std() * np.sqrt(252)

    _signal_frames[code] = out
    return out


def _eps_ttm_series(code: str, days: pd.DatetimeIndex) -> pd.Series:
    """全部披露日 TTM eps 的步进序列, 前向填充到 days"""
    pro = load_profit(code)
    bal = load_balance(code)
    if pro is not None and bal is not None and '归属于母公司所有者的净利润' in pro.columns:
        pf = pro[pro['公告日期'].notna()].sort_values('报告日')
        bd = bal[bal['公告日期'].notna()].sort_values('报告日')
        recs = []
        for _, r in pf.iterrows():
            disc = r['公告日期']
            if not pd.notna(disc):
                continue
            cum = pd.to_numeric(r['归属于母公司所有者的净利润'], errors='coerce')
            if not np.isfinite(cum):
                continue
            rdate = r['报告日']
            prev_annual = pf[(pf['报告日'].dt.month == 12) & (pf['报告日'] < rdate)]
            lyr = float(pd.to_numeric(prev_annual.iloc[-1]['归属于母公司所有者的净利润'], errors='coerce')) \
                if not prev_annual.empty else np.nan
            sq_prev = pf[pf['报告日'] == rdate - pd.DateOffset(years=1)]
            sq = float(pd.to_numeric(sq_prev.iloc[-1]['归属于母公司所有者的净利润'], errors='coerce')) \
                if not sq_prev.empty else np.nan
            if not (np.isfinite(lyr) and np.isfinite(sq)):
                continue
            ttm = cum + lyr - sq
            ticks = float(pd.to_numeric(bd[bd['公告日期'] <= disc].sort_values('报告日').iloc[-1]['实收资本(或股本)'], errors='coerce')) \
                if not bd[bd['公告日期'] <= disc].empty and '实收资本(或股本)' in bd.columns else np.nan
            if not np.isfinite(ticks) or ticks <= 0:
                continue
            recs.append((disc, ttm / ticks))
        if recs:
            s = pd.Series([v for _, v in recs], index=pd.DatetimeIndex([d for d, _ in recs]))
            s = s[~s.index.duplicated(keep='last')].sort_index()
            return s.reindex(days.union(s.index)).ffill().reindex(days)
    # fin 兜底
    fin = load_fin_summary(code)
    if fin is not None and '指标' in fin.columns:
        row = fin[fin['指标'] == '基本每股收益']
        if not row.empty:
            s = row.iloc[0, 2:].astype(object)
            s.index = pd.to_datetime(s.index, format='%Y%m%d', errors='coerce')
            s = pd.to_numeric(s, errors='coerce')
            s = s[(s.index.notna())].sort_index()
            annual = s[s.index.month == 12]
            recs = []
            for i, (d, v) in enumerate(s.items()):
                prev_annual = annual[annual.index < d]
                lyr = float(prev_annual.iloc[-1]) if not prev_annual.empty else np.nan
                sq = s.loc[d - pd.DateOffset(years=1)] if d - pd.DateOffset(years=1) in s.index else np.nan
                if not (np.isfinite(lyr) and np.isfinite(sq)):
                    continue
                recs.append((d, float(v + lyr - sq)))
            if recs:
                ser = pd.Series([v for _, v in recs], index=pd.DatetimeIndex([d for d, _ in recs]))
                ser = ser[~ser.index.duplicated(keep='last')].sort_index()
                return ser.reindex(days.union(ser.index)).ffill().reindex(days)
    return pd.Series(np.nan, index=days)


def _ma120_dev_series(code: str, days: pd.DatetimeIndex) -> pd.Series:
    q = load_qfq(code)
    ma120 = q['close'].rolling(120).mean()
    dev = (q['close'] - ma120) / ma120
    return dev.reindex(days)
