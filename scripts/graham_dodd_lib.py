#!/usr/bin/env python
"""Graham & Dodd 防御型 10 条 - A股 回测核心逻辑 (无泄漏)

数据均为 scripts/graham_data.py 缓存, 全部为本地文件:
    data/meta/graham_universe.csv     全A股清单
    data/financial/{code}_{balance,profit}.csv   sina 报表(含公告日期)
    data/dividend/{code}_dividend.csv  巨潮分红(实施方案公告日期)
    data/{code}_market.csv             不复权日线
    data/{code}_qfq.csv                前复权日线
    data/zh_10y_treasury.csv           中债10Y国债收益率 (AAA代理)
    data/meta/all_a_pe.csv             全部A股平均/中位 PE

无泄漏约定:
  - 财报/分红仅使用 公告日期 ≤ 调仓日 D 的记录
  - 收盘价形成信号 + 下一交易日成交
  - 全部A股为资产池; D 无成交价 → 判不可交易
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIN_DIR = os.path.join(DATA_DIR, 'financial')
DIV_DIR = os.path.join(DATA_DIR, 'dividend')
META_DIR = os.path.join(DATA_DIR, 'meta')

START = pd.Timestamp('2023-01-01')
END = pd.Timestamp('2026-07-31')

# 成本假设 (A股现实)
COMMISSION = 0.0003        # 佣金 万3 双向
COMMISSION_MIN = 5.0       # 单笔最低 5 元
TRANSFER_FEE = 0.00001     # 过户费 双向
STAMP_TAX_HI = 0.001       # 印花税 卖出 (2023-08-28 前)
STAMP_TAX_LO = 0.0005      # 印花税 卖出 (2023-08-28 起)
STAMP_CUTOFF = pd.Timestamp('2023-08-28')

MAX_HOLDINGS = 30          # 最大持仓数
MAX_WEIGHT = 0.15          # 单只最大仓位
YIELD_AUTHORITY = '中债10年期国债收益率(AAA代理)'
MIN_PASS_DEF = 8           # 防御型纳入口径: 至少 8/10，且核心防御条件不得缺失

# —— 防御型纪律 (格雷厄姆) ——
SIZE_QUANTILE = 0.30       # 市值门禁: 需 ≥ 当日全A可交易市值分布的该分位 (温和档, 排除底部30%)
SELL_FLOOR = 5             # 两档卖出: 持有中仍通过 ≥5/10 且门禁达标 → 继续持有
SECTOR_WEIGHT_CAP = 0.25   # 单行业权重上限 (占 NAV)
SECTOR_MAX_STOCKS = 4      # 单行业最大持仓只数
GAUGE_LO_PCT = 0.50        # 估值仪表盘: 低于该百分位 → 满仓权益
GAUGE_HI_PCT = 0.85        # 估值仪表盘: 高于该百分位 → 陡降至保守权益仓位
GAUGE_LO_WEIGHT = 0.40     # 仪表盘高位时保留的最低权益比例
GAUGE_LOOKBACK_YEARS = 10  # 全A平均PE的分位回看年数

# ─────────────────────────────────────────────────────────────
# 数据加载 (本地缓存)
# ─────────────────────────────────────────────────────────────

_cache: Dict[str, object] = {}


def _cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def load_universe() -> pd.DataFrame:
    def _f():
        return pd.read_csv(os.path.join(META_DIR, 'graham_universe.csv'),
                           dtype={'code': str})
    return _cached('universe', _f)


def load_balance(code: str) -> Optional[pd.DataFrame]:
    p = os.path.join(FIN_DIR, f'{code}_balance.csv')
    if not os.path.exists(p):
        return None
    def _f():
        df = pd.read_csv(p, dtype={'报告日': str, '公告日期': str})
        df['报告日'] = pd.to_datetime(df['报告日'], errors='coerce')
        df['公告日期'] = pd.to_datetime(df['公告日期'], errors='coerce')
        return df.dropna(subset=['报告日']).sort_values('报告日')
    return _cached(f'bal_{code}', _f)


def load_profit(code: str) -> Optional[pd.DataFrame]:
    p = os.path.join(FIN_DIR, f'{code}_profit.csv')
    if not os.path.exists(p):
        return None
    def _f():
        df = pd.read_csv(p, dtype={'报告日': str, '公告日期': str})
        df['报告日'] = pd.to_datetime(df['报告日'], errors='coerce')
        df['公告日期'] = pd.to_datetime(df['公告日期'], errors='coerce')
        return df.dropna(subset=['报告日']).sort_values('报告日')
    return _cached(f'prof_{code}', _f)


def load_dividend(code: str) -> Optional[pd.DataFrame]:
    p = os.path.join(DIV_DIR, f'{code}_dividend.csv')
    if not os.path.exists(p):
        return None
    def _f():
        df = pd.read_csv(p, dtype={'实施方案公告日期': str})
        return df
    return _cached(f'div_{code}', _f)


def _corporate_actions(code: str, start: pd.Timestamp, end: pd.Timestamp) -> List[tuple]:
    """Cash/stock actions after ``start`` and on or before ``end``.

    Raw prices are used throughout the backtests.  This keeps dividends explicit
    instead of treating a forward-adjusted price series as tax-free immediate
    reinvestment.  The vendor expresses all action ratios per ten shares.
    """
    div = load_dividend(code)
    if div is None or div.empty or '除权日' not in div.columns:
        return []
    d = div.copy()
    d['action_date'] = pd.to_datetime(d['除权日'], errors='coerce')
    d = d[(d['action_date'] > start) & (d['action_date'] <= end)].sort_values('action_date')
    actions = []
    for _, row in d.iterrows():
        cash = pd.to_numeric(row.get('派息比例', np.nan), errors='coerce')
        bonus = pd.to_numeric(row.get('送股比例', np.nan), errors='coerce')
        transfer = pd.to_numeric(row.get('转增比例', np.nan), errors='coerce')
        actions.append((row['action_date'],
                        0.0 if not np.isfinite(cash) else float(cash) / 10.0,
                        1.0 + (0.0 if not np.isfinite(bonus) else float(bonus) / 10.0)
                            + (0.0 if not np.isfinite(transfer) else float(transfer) / 10.0)))
    return actions


def _dividend_tax_rate(buy_date: pd.Timestamp, pay_date: pd.Timestamp) -> float:
    """A-share individual dividend withholding, using the holding period at payout."""
    days = (pay_date - buy_date).days
    if days <= 30:
        return 0.20
    if days <= 365:
        return 0.10
    return 0.0


def load_market(code: str, qfq: bool = False) -> Optional[pd.DataFrame]:
    name = f'{code}_qfq.csv' if qfq else f'{code}_market.csv'
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p):
        return None
    def _f():
        df = pd.read_csv(p, index_col='date', parse_dates=True).sort_index()
        df = df[~df.index.duplicated(keep='last')]
        return df
    return _cached(f'mkt_{qfq}_{code}', _f)


def load_10y() -> pd.DataFrame:
    def _f():
        df = pd.read_csv(os.path.join(DATA_DIR, 'zh_10y_treasury.csv'),
                         index_col='date', parse_dates=True)
        return df
    return _cached('10y', _f)


def load_all_a_pe() -> pd.DataFrame:
    def _f():
        df = pd.read_csv(os.path.join(META_DIR, 'all_a_pe.csv'),
                         index_col='date', parse_dates=True)
        return df
    return _cached('allpe', _f)


def load_index(code: str) -> Optional[pd.DataFrame]:
    def _f():
        p = os.path.join(DATA_DIR, f'{code}_market.csv')
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p, index_col='date', parse_dates=True).sort_index()
        return df
    return _cached(f'idx_{code}', _f)


# ─────────────────────────────────────────────────────────────
# 宏观辅助
# ─────────────────────────────────────────────────────────────

def r10y(D: pd.Timestamp) -> float:
    """调仓日 D 可观测的中债10Y国债收益率(≤D 最近值)"""
    df = load_10y()
    s = df['中国国债收益率10年']
    s = s[s.index <= D]
    if s.empty:
        return np.nan
    return float(s.iloc[-1]) / 100.0  # 百分比 → 小数


def market_avg_pe_5y(D: pd.Timestamp) -> float:
    """过去5年内全部A股等权平均PE (TTM) 的均值, 仅用 ≤D 数据"""
    df = load_all_a_pe()
    win = df[(df.index > D - pd.DateOffset(years=5)) & (df.index <= D)]
    if win.empty:
        return np.nan
    return float(win['averagePETTM'].mean())


def market_gauge(D: pd.Timestamp, lookback_years: int = GAUGE_LOOKBACK_YEARS) -> float:
    """估值仪表盘: 当日全部A股平均PE 相对自身历史(过去 lookback_years 年)的百分位。

    0 → 处于历史地量(便宜), 1 → 处于历史最高分位(昂贵)。仅用 ≤D 数据, 无前视。
    数据不足时返回 0.5 中性值。
    """
    df = load_all_a_pe()
    s = df['averagePETTM']
    s = s[s.index <= D]
    if len(s) < 60:
        return 0.5
    hist = s[s.index > D - pd.DateOffset(years=lookback_years)]
    if hist.empty or len(hist) < 60:
        hist = s
    cur = float(s.iloc[-1])
    return float((hist < cur).mean())


def target_equity_weight(g: float) -> float:
    """仪表盘百分位 → 目标权益仓位比例"""
    if g <= GAUGE_LO_PCT:
        return 1.0
    if g >= GAUGE_HI_PCT:
        return GAUGE_LO_WEIGHT
    return 1.0 - (1.0 - GAUGE_LO_WEIGHT) * (g - GAUGE_LO_PCT) / (GAUGE_HI_PCT - GAUGE_LO_PCT)


# ─────────────────────────────────────────────────────────────
# 单股 point-in-time 快照
# ─────────────────────────────────────────────────────────────

@dataclass
class Snap:
    code: str = ''
    name: str = ''
    D: pd.Timestamp = None
    market_cap: float = np.nan       # 依据最新股本 × 收盘价
    shares: float = np.nan           # 股数 (实收资本)
    bvps: float = np.nan             # 归母净资产/股
    tbvps: float = np.nan            # 有形净资产/股
    ncavps: float = np.nan           # 净流动资产/股
    current_ratio: float = np.nan
    debt_ratio: float = np.nan       # 负债/归母净资产 (产权比率)
    cond8: float = np.nan            # 负债 / (2×NCA) <1 判定用
    eps_ttm: float = np.nan
    eps_annual: List[float] = field(default_factory=list)
    eps_years: List[int] = field(default_factory=list)
    np_annual: List[float] = field(default_factory=list)   # 年报归母净利 (公告日≤D)
    np_years: List[int] = field(default_factory=list)
    roe_annual: List[float] = field(default_factory=list)  # 年报 ROE (净利 ÷ 当年末净资产)
    ttm_yoy: float = np.nan                                # TTM 归母净利同比
    dividend_yield: float = np.nan
    div_years: List[int] = field(default_factory=list)  # 已公告(≤D)现金分红对应的财年
    industry: str = ''
    price: float = np.nan            # 不复权收盘
    tradable: bool = False
    has_10y: bool = False
    reason: str = ''


def _close_at(df, at: pd.Timestamp) -> float:
    prev = df.index[df.index <= at]
    if len(prev) == 0:
        return np.nan
    return float(df.loc[prev[-1], 'close'])


def _attach_name(s, pre: Dict[str, str]):
    s.code = s.code
    if not hasattr(s, 'name'):
        s.name = pre.get(s.code, s.code)
    return s


_name_cache: Optional[Dict[str, str]] = None


def _name_of(code: str) -> str:
    global _name_cache
    if _name_cache is None:
        u = load_universe()
        _name_cache = dict(zip(u['code'], u['name']))
    return _name_cache.get(code, code)


def is_st_name(name: str) -> bool:
    """Conservative fallback for the cached security name.

    The local universe does not retain historical name changes, so this cannot
    certify a historical ST state; it does prevent a currently labelled ST
    security from entering a defensive portfolio.
    """
    return bool(name and 'ST' in str(name).upper())


_industry_cache: Optional[Dict[str, str]] = None


def _industry_of(code: str) -> str:
    """申万一级行业 (以东方财富行业板块近似, 见 scripts/graham_data.py fetch-industry)。
    缺失时返回空串, 视为不参与行业封顶。"""
    global _industry_cache
    if _industry_cache is None:
        u = load_universe()
        _industry_cache = {}
        if 'industry' in u.columns:
            for c, ind in zip(u['code'], u['industry'].fillna('')):
                _industry_cache[str(c)] = str(ind)
    return _industry_cache.get(code, '')


def _mo_score(s: 'Snapshot') -> float:
    """安全边际排序: 价格/有形净资产 与 价格/(20×EPS) 中的更差者, 值越小=越便宜优先买入."""
    tbv = s.tbvps if np.isfinite(s.tbvps) and s.tbvps > 0 else np.nan
    pe = s.price / s.eps_ttm if (np.isfinite(s.eps_ttm) and s.eps_ttm > 0) else np.nan
    if np.isfinite(tbv) and np.isfinite(pe):
        return (s.price / tbv) * (pe / 20.0)
    if np.isfinite(tbv):
        return s.price / tbv
    return np.inf if not np.isfinite(pe) else pe


def profit_trend_components(s: 'Snap') -> Optional[Dict[str, float]]:
    """盈利趋势分量 (全部用时点数据, 无前视)。缺失某分量则相应降权。

    分量:
      ttm_yoy     TTM 归母净利同比 (盈利是否仍在变好的直接度量)
      slope       近6年年报归母净利 对数线性斜率 (避免 c9 的首尾两点失真)
      roe         最新年报 ROE 水平
      roe_trend   最新 ROE - 3 年前 ROE 变化
      run         连续盈利增长年数 (从最近一年往前数)
    """
    if not s.tradable:
        return None
    comps = {}
    if np.isfinite(s.ttm_yoy):
        comps['ttm_yoy'] = s.ttm_yoy
    np_ = s.np_annual[-6:]
    if len(np_) >= 4 and all(v > 0 for v in np_):
        ys = np.log(np.asarray(np_, dtype=float))
        xs = np.arange(len(ys), dtype=float)
        comps['slope'] = float(np.polyfit(xs, ys, 1)[0])
    roes = [r for r in s.roe_annual if np.isfinite(r)]
    if roes:
        comps['roe'] = roes[-1]
        if len(roes) >= 3:
            comps['roe_trend'] = roes[-1] - roes[-3]
    run_len = 0
    for i in range(len(s.np_annual) - 1, 0, -1):
        if s.np_annual[i] <= s.np_annual[i - 1]:
            break
        run_len += 1
    if len(s.np_annual) >= 2:
        comps['run'] = float(run_len)
    return comps or None


def rank_profit_trend(candidates: List['Snap']) -> Dict[int, float]:
    """对候选池做盈利趋势横截面排名: 每个分量现值排百分位 (高者优), 等权求和, 越小越好.

    候选无任何可测分量为 inf (排最后, 只在仓位置富余时买入)。
    """
    keys = ('ttm_yoy', 'slope', 'roe', 'roe_trend', 'run')
    comps_list = [profit_trend_components(s) for s in candidates]
    out = {}
    for idx, comps in enumerate(comps_list):
        if comps is None:
            out[id(candidates[idx])] = float('inf')
            continue
        total = 0.0
        for k in keys:
            if k not in comps:
                continue
            vals = [c[k] for c in comps_list if c is not None and k in c]
            if not vals:
                continue
            rank = pd.Series(vals).rank(ascending=False, method='min').iloc[vals.index(comps[k])]
            total += float(rank)
        out[id(candidates[idx])] = total
    return out


def snapshot(code: str, D: pd.Timestamp, bal=None, pro=None, div=None,
             mkt_raw=None) -> Snap:
    s = Snap(code=code, D=D)
    s.name = _name_of(code)
    bal = bal if bal is not None else load_balance(code)
    pro = pro if pro is not None else load_profit(code)
    div = div if div is not None else load_dividend(code)
    mkt_raw = mkt_raw if mkt_raw is not None else load_market(code, qfq=False)

    if bal is None or pro is None or mkt_raw is None or mkt_raw.empty:
        s.reason = '缺数据'
        return s

    # 价格 (不复权, 仅可用时)
    if D in mkt_raw.index:
        price = float(mkt_raw.loc[D, 'close'])
    else:
        price = np.nan
    if not np.isfinite(price) or price <= 0:
        s.reason = 'D日无成交'
        return s
    s.price = price
    s.tradable = True
    s.industry = _industry_of(code)

    # 最新已披露资产负债表 (公告日 ≤ D)
    bd = bal[bal['公告日期'].fillna(pd.Timestamp.max) <= D]
    if bd.empty:
        s.reason = '无已披露报表'
        return s
    b = bd.sort_values('报告日').iloc[-1]

    ta = b.get('资产总计', np.nan)
    ca = b.get('流动资产合计', np.nan)
    cl = b.get('流动负债合计', np.nan)
    tl = b.get('负债合计', np.nan)
    ias = b.get('无形资产', np.nan)
    gw = b.get('商誉', np.nan)
    eq = b.get('归属于母公司股东权益合计', np.nan)
    ticks = b.get('实收资本(或股本)', np.nan)

    for col in ('流动资产合计', '流动负债合计', '负债合计', '归属于母公司股东权益合计'):
        pass
    # 用"所有者权益合计"兜底归母权益
    if not np.isfinite(eq):
        eq = b.get('所有者权益(或股东权益)合计', np.nan)

    if not np.isfinite(ticks) or ticks <= 0:
        s.reason = '缺股本'
        return s
    s.shares = ticks          # 实收资本(元)→股 (面值1元假设)

    s.market_cap = price * ticks
    if np.isfinite(eq) and eq > 0:
        s.bvps = eq / ticks
        s.tbvps = (eq - (ias if np.isfinite(ias) else 0.0)
                   - (gw if np.isfinite(gw) else 0.0)) / ticks
    if np.isfinite(ca) and np.isfinite(tl):
        nca = ca - tl
        s.ncavps = nca / ticks
        s.cond8 = tl / (2 * nca) if nca > 0 else np.inf
    if np.isfinite(ca) and np.isfinite(cl) and cl > 0:
        s.current_ratio = ca / cl
    if np.isfinite(tl) and np.isfinite(eq) and eq > 0:
        s.debt_ratio = tl / eq

    # 最新已披露利润表 (公告日 ≤ D) → TTM 归母净利
    pd_ = pro[pro['公告日期'].fillna(pd.Timestamp.max) <= D]
    if pd_.empty:
        s.reason = '无已披露利润表'
        return s
    pf = pd_.sort_values('报告日')
    latest = pf.iloc[-1]
    cum_profit = latest['归属于母公司所有者的净利润']
    if not np.isfinite(cum_profit):
        cum_profit = np.nan

    # TTM = 最新累计 + 上年报 - 去年同期累计
    rdate = latest['报告日']
    same_q_prev = rdate - pd.DateOffset(years=1)
    prev_annual = pf[
        (pf['报告日'].dt.month == 12) & (pf['报告日'] < rdate)
    ]
    lyr_val = np.nan
    if not prev_annual.empty:
        lyr_val = float(prev_annual.iloc[-1]['归属于母公司所有者的净利润'])
    sq_prev = pf[pf['报告日'] == same_q_prev]
    sq_val = float(sq_prev.iloc[-1]['归属于母公司所有者的净利润']) if not sq_prev.empty else np.nan

    if np.isfinite(cum_profit) and np.isfinite(lyr_val) and np.isfinite(sq_val):
        ttm = cum_profit + lyr_val - sq_val
        s.eps_ttm = ttm / ticks if ticks > 0 else np.nan

        # TTM 归母净利同比: 一年前同报告期口径的 TTM (盈利趋势的分量之一)
        rdate_p = rdate - pd.DateOffset(years=1)
        blk_p = pf[pf['报告日'] <= rdate_p]
        if not blk_p.empty:
            lp = blk_p.iloc[-1]
            cum_prev = lp['归属于母公司所有者的净利润']
            pa2 = pf[(pf['报告日'].dt.month == 12) & (pf['报告日'] < lp['报告日'])]
            lyr2 = float(pa2.iloc[-1]['归属于母公司所有者的净利润']) if not pa2.empty else np.nan
            sq2 = pf[pf['报告日'] == lp['报告日'] - pd.DateOffset(years=1)]
            sq2_v = float(sq2.iloc[-1]['归属于母公司所有者的净利润']) if not sq2.empty else np.nan
            if (np.isfinite(cum_prev) and np.isfinite(lyr2) and np.isfinite(sq2_v)):
                ttm_prev = cum_prev + lyr2 - sq2_v
                if ttm_prev > 0:
                    s.ttm_yoy = ttm / ttm_prev - 1.0

    # 年度EPS/净利序列 (公告≤D, 报告日为12-31), 用于十年增长/降年 + 盈利趋势
    annuals = pf[pf['报告日'].dt.month == 12]
    for _, r in annuals.iterrows():
        v = pd.to_numeric(r['基本每股收益'], errors='coerce')
        if np.isfinite(v):
            s.eps_annual.append(float(v))
            s.eps_years.append(int(r['报告日'].year))
        npv = pd.to_numeric(r.get('归属于母公司所有者的净利润', np.nan), errors='coerce')
        if np.isfinite(npv):
            s.np_annual.append(float(npv))
            s.np_years.append(int(r['报告日'].year))
    if len(s.eps_annual) >= 10:
        s.has_10y = True

    # 年度 ROE 序列: 净利 Y ÷ 最早披露≤D 的 Y-12-31 归母净资产 (公告≤D, 无前视)
    if s.np_annual:
        balD = bal[bal['公告日期'].fillna(pd.Timestamp.max) <= D].dropna(subset=['报告日'])
        balD = balD.sort_values('报告日')
        for npv, y in zip(s.np_annual, s.np_years):
            eq_row = balD[balD['报告日'] <= pd.Timestamp(year=y, month=12, day=31)]
            if eq_row.empty:
                continue
            b = eq_row.iloc[-1]
            eq = b.get('归属于母公司股东权益合计', np.nan)
            if not np.isfinite(eq):
                eq = b.get('所有者权益(或股东权益)合计', np.nan)
            if np.isfinite(eq) and eq > 0 and npv > 0:
                s.roe_annual.append(npv / eq)
            else:
                s.roe_annual.append(np.nan)

    # 分红收益率
    if div is not None and not div.empty and '实施方案公告日期' in div.columns:
        dv = div.copy()
        dv['公告日'] = pd.to_datetime(dv['实施方案公告日期'], errors='coerce')
        dv['派息比例'] = pd.to_numeric(dv['派息比例'], errors='coerce')
        annual = dv[(dv['公告日'].fillna(pd.Timestamp.max) <= D)
                    & dv['报告时间'].astype(str).str.endswith('年报')
                    & (dv['派息比例'].fillna(0) > 0)]
        if not annual.empty:
            latest_div = annual.sort_values('公告日').iloc[-1]
            if latest_div['公告日'] >= D - pd.DateOffset(days=550):
                dps = latest_div['派息比例'] / 10.0
                s.dividend_yield = dps / price
        # 近若干财年现金分红记录 (财年 = 报告年度, 公告日 ≤ D)
        dvec = dv[(dv['公告日'].fillna(pd.Timestamp.max) <= D)]
        dvec = dvec[pd.to_numeric(dvec['派息比例'], errors='coerce').fillna(0) > 0]
        yrs = pd.to_numeric(dvec['报告时间'].astype(str).str.extract(r'^(\d{4})')[0],
                            errors='coerce')
        s.div_years = sorted(int(y) for y in yrs.dropna().unique())
    return s


# ─────────────────────────────────────────────────────────────
# 10 条件 评估
# ─────────────────────────────────────────────────────────────

COND_NAMES = [
    ('c1', '收益价格比 ≥ 2×AAA'),
    ('c2', 'PE ≤ 60%×市场5年均PE'),
    ('c3', '股息率 ≥ 2/3×AAA'),
    ('c4', '价格 ≤ 2/3 有形净资产'),
    ('c5', '价格 ≤ 2/3 净流动资产'),
    ('c6', '负债/净资产 < 1'),
    ('c7', '流动比率 ≥ 2'),
    ('c8', '总负债 < 2×净流动资产'),
    ('c9', '十年EPS CAGR > 7%'),
    ('c10', '十年中EPS下滑年 ≤ 2'),
]

GATE_NAMES = [
    ('earnings', '连续盈利: 近5年报 ≤1次为负 且最近一年为正'),
    ('dividend', '近5财年现金分红 ≥4 年'),
    ('size', '市值 ≥ 当日全A可交易分位下限'),
]


@dataclass
class Eval:
    passed: bool = False
    npass: int = 0
    gates_ok: bool = False
    values: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, Tuple[bool, str, str]] = field(default_factory=dict)  # id -> (pass, cur, req)
    gates: Dict[str, Tuple[bool, str, str]] = field(default_factory=dict)   # id -> (pass, cur, req)


def evaluate(s: Snap, D: pd.Timestamp, r=None, mpe=None, min_pass: int = 7,
             size_floor: Optional[float] = None) -> Eval:
    e = Eval()
    if r is None:
        r = r10y(D)
    if mpe is None:
        mpe = market_avg_pe_5y(D)
    if not s.tradable or not np.isfinite(r) or not np.isfinite(mpe):
        e.detail['c1'] = (False, 'NA', f'不可交易/缺宏观({r=:},{mpe=:})')
        return e

    pe_ttm = s.price / s.eps_ttm if np.isfinite(s.eps_ttm) and s.eps_ttm > 0 else np.nan

    def c1():
        ep = s.eps_ttm / s.price
        need = 2 * r
        return ep >= need, f'{ep:.4f}', f'≥{need:.4f}'

    def c2():
        if not np.isfinite(pe_ttm):
            return False, 'PE<0(无)', f'≤{0.6*mpe:.1f}'
        return pe_ttm <= 0.6 * mpe, f'{pe_ttm:.1f}', f'≤{0.6*mpe:.1f}'

    def c3():
        need = (2 / 3) * r
        y = s.dividend_yield
        return (np.isfinite(y) and y >= need), f'{y:.4f}' if np.isfinite(y) else '无分红', f'≥{need:.4f}'

    def c4():
        if not np.isfinite(s.tbvps):
            return False, 'NA', f'≤{2/3*0:.2f}'
        return s.price <= (2 / 3) * s.tbvps, f'{s.price:.1f} ', f'≤{2/3*s.tbvps:.1f}'

    def c5():
        if not np.isfinite(s.ncavps):
            return False, 'NA', '≤2/3×NCA'
        return s.price <= (2 / 3) * s.ncavps, f'{s.price:.1f}', f'≤{2/3*s.ncavps:.1f}'

    def c6():
        return np.isfinite(s.debt_ratio) and s.debt_ratio < 1.0, \
            f'{s.debt_ratio:.2f}' if np.isfinite(s.debt_ratio) else 'NA', '<1'

    def c7():
        return np.isfinite(s.current_ratio) and s.current_ratio >= 2.0, \
            f'{s.current_ratio:.2f}' if np.isfinite(s.current_ratio) else 'NA', '≥2'

    def c8():
        return np.isfinite(s.cond8) and s.cond8 < 1.0, \
            f'{s.cond8:.2f}' if np.isfinite(s.cond8) else 'NA', '<1'

    def c9():
        if not s.has_10y:
            return False, '不足10年', 'CAGR>7%'
        eps = s.eps_annual[-10:]
        yrs = s.eps_years[-10:]
        e0, e1 = eps[0], eps[-1]
        if e0 <= 0 or e1 <= 0:
            return False, f'{e1:e}', 'CAGR>7%'
        cagr = (e1 / e0) ** (1 / (yrs[-1] - yrs[0])) - 1
        return cagr > 0.07, f'{cagr:.1%}', '>7%'

    def c10():
        if not s.has_10y:
            return False, '不足10年', '≤2'
        eps = s.eps_annual[-10:]
        decl = sum(1 for i in range(1, len(eps)) if eps[i] < eps[i - 1])
        return decl <= 2, f'{decl}年', '≤2'

    checks = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
    npass = 0
    for (cid, _lbl), f in zip(COND_NAMES, checks):
        try:
            ok_bool, cur, req = f()
        except Exception:
            ok_bool, cur, req = False, 'ERR', '?'
        e.detail[cid] = (bool(ok_bool), str(cur), str(req))
        if ok_bool:
            npass += 1
    e.npass = npass
    defensive_core = ('c1', 'c2', 'c3', 'c6', 'c7', 'c8', 'c9', 'c10')
    e.passed = npass >= min_pass and all(e.detail[k][0] for k in defensive_core)

    # —— 硬性门禁 (格雷厄姆防御纪律, 独立于 10 条件评分) ——
    # 盈利: 近 5 年报 ≤1 次为负 且最近一年为正
    eps, yrs = s.eps_annual, s.eps_years
    if len(eps) >= 5:
        tail_e, tail_y = eps[-5:], yrs[-5:]
        neg = sum(1 for v in tail_e if not np.isfinite(v) or v < 0)
        ok_e = (neg <= 1) and (np.isfinite(tail_e[-1]) and tail_e[-1] > 0)
        e.gates['earnings'] = (ok_e, f'{tail_e[-1]:.4f} (近{len(tail_e)}年, 负{neg}次)',
                               '≥0 且负次数≤1')
    else:
        e.gates['earnings'] = (False, f'仅{len(eps)}年报', '需≥3年报')

    # 分红: 近 5 个财年 (截至 D.year-1) 现金分红公告 ≥4 年
    fy_end = int(D.year) - 1
    need_yrs = [fy_end - k for k in range(5)]
    got = [y for y in need_yrs if y in s.div_years]
    ok_d = len(got) >= 4
    e.gates['dividend'] = (ok_d, f'{len(got)}/5 财年', f'≥4 财年 ({need_yrs[0]}~{need_yrs[-1]})')

    # 市值: 需 ≥ 全A可交易市值分位下限 (由 screen_all/run 传入; None=不参与判定)
    if size_floor is None or not np.isfinite(s.market_cap):
        ok_s = True if size_floor is None else False
        e.gates['size'] = (ok_s, f'{s.market_cap/1e8:.0f}亿', '未评估' if size_floor is None
                           else f'≥{size_floor/1e8:.0f}亿')
    else:
        e.gates['size'] = (s.market_cap >= size_floor, f'{s.market_cap/1e8:.0f}亿',
                           f'≥{size_floor/1e8:.0f}亿')

    e.gates_ok = bool(e.gates['earnings'][0] and e.gates['dividend'][0] and e.gates['size'][0])
    return e


# ─────────────────────────────────────────────────────────────
# 调仓日历 & 市场交易日
# ─────────────────────────────────────────────────────────────

def trading_days() -> pd.DatetimeIndex:
    """以 000300 指数日期为交易日历 (宽基准)"""
    df = load_index('000300')
    if df is None or df.empty:
        return pd.DatetimeIndex([])
    return df.index


def rebalance_dates() -> List[pd.Timestamp]:
    cal = trading_days()
    out = []
    q = pd.Timestamp(START)
    end = pd.Timestamp(END)
    while q <= end:
        nxt = cal[cal >= q]
        if len(nxt):
            out.append(pd.Timestamp(nxt[0]))
        q = (pd.Timestamp(year=q.year, month=q.month, day=1)
             + pd.DateOffset(months=3))  # 下一季度首日
    return out


# ─────────────────────────────────────────────────────────────
# 回测
# ─────────────────────────────────────────────────────────────

@dataclass
class Pos:
    code: str
    name: str
    shares: float
    buy_price: float
    invested: float          # 买入毛资金 (不含费)
    anchor_qfq: float        # legacy field; raw-price valuation no longer uses qfq
    buy_date: pd.Timestamp


def stamp_tax(D: pd.Timestamp) -> float:
    return STAMP_TAX_LO if D >= STAMP_CUTOFF else STAMP_TAX_HI


def buy_fee(amount: float) -> float:
    return max(amount * COMMISSION, COMMISSION_MIN) + amount * TRANSFER_FEE


def sell_fee(D: pd.Timestamp, amount: float) -> float:
    return (max(amount * COMMISSION, COMMISSION_MIN)
            + amount * TRANSFER_FEE + amount * stamp_tax(D))


FUNNEL: Dict[pd.Timestamp, dict] = {}   # screen_all 每次全市场扫描后记录门禁漏斗


def screen_all(D: pd.Timestamp, min_pass: int = 7,
               size_quantile: float = SIZE_QUANTILE) -> List[Tuple[Snap, Eval]]:
    uni = load_universe()
    r = r10y(D)
    mpe = market_avg_pe_5y(D)
    results = []
    caps = []
    scored = 0
    for code in uni['code']:
        s = snapshot(code, D)
        if not s.tradable:
            continue
        if is_st_name(s.name):
            continue
        # Missing industry labels cannot be allowed to bypass concentration caps.
        if not s.industry:
            continue
        if not np.isfinite(s.market_cap) or s.market_cap <= 0:
            continue
        scored += 1
        ev = evaluate(s, D, r=r, mpe=mpe, min_pass=min_pass, size_floor=None)
        caps.append(float(s.market_cap))
        if ev.passed:
            results.append((s, ev))
    # 市值分位下限 (当日全A可交易样本)
    if caps:
        floor = float(pd.Series(caps).quantile(size_quantile))
    else:
        floor = np.inf
    score_ok = len(results)
    div_ok = ear_ok = 0
    out = []
    for s, ev in results:
        ev.gates['size'] = (s.market_cap >= floor, f'{s.market_cap/1e8:.0f}亿',
                            f'≥{floor/1e8:.0f}亿')
        ev.gates_ok = bool(ev.gates['earnings'][0] and ev.gates['dividend'][0]
                           and ev.gates['size'][0])
        div_ok += int(ev.gates['dividend'][0])
        ear_ok += int(ev.gates['earnings'][0])
        if ev.gates_ok:
            out.append((s, ev))
    FUNNEL[D] = {'tradable': scored, 'score': score_ok, 'div': div_ok,
                 'earn': ear_ok, 'final': len(out)}
    return out


class GrahamBacktest:
    def __init__(self, initial_capital: float = 1_000_000.0,
                 rank_by: str = 'margin_of_safety'):
        self.initial_capital = initial_capital
        # rank_by: 'margin_of_safety'= 安全边际(现状) | 'profit_trend'= 盈利趋势排序
        self.rank_by = rank_by
        self.cash = initial_capital
        self.positions: Dict[str, Pos] = {}
        self.trades: List[dict] = []
        self.nav_curve: pd.Series = pd.Series(dtype=float)
        self.holdings_curve: pd.Series = pd.Series(dtype=float)
        self._snapshots: List[tuple] = []   # (date, cash, list[(code,invested,anchor,buy_price)])
        self.periods: List[dict] = []       # 每调仓日: 仪表盘/门禁/现金/行业分布 记录

    # —— 内部工具 ——
    @staticmethod
    def _next_trading_day(D: pd.Timestamp) -> Optional[pd.Timestamp]:
        cal = trading_days()
        nxt = cal[cal > D]
        return pd.Timestamp(nxt[0]) if len(nxt) else None

    @staticmethod
    def _position_value(pos: Pos, t: pd.Timestamp) -> Tuple[float, float]:
        """Return market value and net cash dividends using raw prices."""
        raw = load_market(pos.code, qfq=False)
        if raw is None:
            return pos.invested, 0.0
        prev = raw.index[raw.index <= t]
        if len(prev) == 0:
            return pos.invested, 0.0
        shares = pos.shares
        dividends = 0.0
        for pay_date, dps, share_factor in _corporate_actions(pos.code, pos.buy_date, t):
            dividends += shares * dps * (1.0 - _dividend_tax_rate(pos.buy_date, pay_date))
            shares *= share_factor
        return shares * float(raw.loc[prev[-1], 'close']), dividends

    @classmethod
    def _liquidation_value(cls, pos: Pos, t: pd.Timestamp) -> float:
        market, dividends = cls._position_value(pos, t)
        return market + dividends

    def _mark_values(self, t: pd.Timestamp) -> Tuple[float, float]:
        """返回 (总权益, 持仓市值)"""
        val = self.cash
        held = 0.0
        for pos in self.positions.values():
            pv, cash_div = self._position_value(pos, t)
            val += pv + cash_div
            held += pv
        return val, held

    def _mark(self, t: pd.Timestamp) -> float:
        return self._mark_values(t)[0]

    def _held(self, t: pd.Timestamp) -> float:
        return self._mark_values(t)[1]

    def run(self, dates: List[pd.Timestamp],
            screening_map: Optional[Dict[pd.Timestamp, List[Snap]]] = None,
            verbose: bool = True, end: Optional[pd.Timestamp] = None) -> Tuple[pd.Series, list, list]:
        i = 0
        n = len(dates)
        while i < n:
            signal_D = dates[i]
            D = self._next_trading_day(signal_D)
            if D is None:
                break
            # 1. 筛选 (全市场) —— 允许复用预计算结果; passed 已含防御核心条件与门禁
            if screening_map is not None and signal_D in screening_map:
                passed = [s for s, _ in screening_map[signal_D]]
            else:
                screened = screen_all(signal_D)
                passed = [s for s, _ in screened]

            passed_codes = {s.code for s in passed}
            # 2. 卖出两档: 仍在候选 → 持有; 否则按 5/10 下限 + 门禁 判定
            sell_specs = []
            for code in list(self.positions.keys()):
                if code in passed_codes:
                    continue
                s = snapshot(code, signal_D)
                if not s.tradable:
                    sell_specs.append((code, '停牌/退市/无成交'))
                    continue
                ev = evaluate(s, signal_D, min_pass=SELL_FLOOR, size_floor=None)
                if ev.npass >= SELL_FLOOR and ev.gates['earnings'][0] and ev.gates['dividend'][0]:
                    continue  # 两档: 仍通过 ≥5/10 且盈利/分红门禁达标 → 持有
                if ev.npass < SELL_FLOOR:
                    sell_specs.append((code, f'break {ev.npass}/10 跌破卖出下限{SELL_FLOOR}/10'))
                else:
                    fails = [k for k in ('earnings', 'dividend') if not ev.gates[k][0]]
                    sell_specs.append((code, '门禁不达标: ' + '、'.join(fails)))

            # 3. 执行卖出
            for code, reason in sell_specs:
                pos = self.positions.pop(code)
                proceed = self._liquidation_value(pos, D)
                fee = sell_fee(D, proceed)
                self.cash += proceed - fee
                self.trades.append({'date': D, 'code': pos.code, 'name': pos.name,
                                    'action': 'sell', 'shares': pos.shares,
                                    'price': np.nan, 'proceeds': proceed, 'fee': fee,
                                    'reason': reason})

            # 3.5 主动降仓至权益目标 (估值仪表盘真正落地的现金缓冲):
            #   现有权益比例 > 目标 → 从"安全边际最差"的持仓开始整仓卖出, 回收现金;
            #   与两档卖出不同, 这里不要求跌破 5/10, 只遵循"市场贵了就把仓位降下来"。
            total_nav = self._mark(D)
            held_val = self._mark_values(D)[1]
            g_pct = market_gauge(signal_D)
            eq_w = target_equity_weight(g_pct)
            trim_value = held_val - total_nav * eq_w
            if total_nav > 0 and trim_value > max(total_nav * 0.02, 0.0):
                held_sorted = sorted(self.positions.items(),
                                     key=lambda kv: (_mo_score(snapshot(kv[0], signal_D)), kv[0]))
                for code, pos in held_sorted:
                    if trim_value <= 0:
                        break
                    proceed = self._liquidation_value(pos, D)
                    fee = sell_fee(D, proceed)
                    self.positions.pop(code)
                    self.cash += proceed - fee
                    trim_value -= proceed
                    self.trades.append({'date': D, 'code': code, 'name': pos.name,
                                        'action': 'sell', 'shares': pos.shares,
                                        'price': np.nan, 'proceeds': proceed, 'fee': fee,
                                        'reason': f'主动降仓: 仪表盘高位(PE分位{g_pct:.0%}→目标{eq_w:.0%})'})

            # 4. 决定买入 (等权, ≤30, 单只≤15%, 行业封顶, 估值仪表盘现金缓冲)
            #    选股顺序: rank_by 决定 — 安全边际(现状) 或 盈利趋势(P0 改进)。
            keep = len(self.positions)
            budget_slots = max(0, MAX_HOLDINGS - keep)
            total_nav = self._mark(D)
            held_val = self._mark_values(D)[1]
            g_pct = market_gauge(signal_D)
            eq_w = target_equity_weight(g_pct)
            # 现金缓冲: 权益目标 = NAV×eq_w; 高于目标已在上一步 3.5 主动降仓, 这里只约束新增买入
            buy_budget = max(0.0, total_nav * eq_w - held_val)
            buy_budget = min(buy_budget, self.cash)

            if self.rank_by == 'profit_trend':
                trend_rank = rank_profit_trend(list(passed))
                def _pick_key(s):
                    return (trend_rank.get(id(s), float('inf')), s.code)
            else:
                def _pick_key(s):
                    return (_mo_score(s), s.code)
            picks = sorted(passed, key=_pick_key)[:budget_slots]
            ind_count = {}
            ind_value = {}
            for p in self.positions.values():
                ind = _industry_of(p.code)
                ind_count[ind] = ind_count.get(ind, 0) + 1
                ind_value[ind] = ind_value.get(ind, 0.0) + p.invested
            if picks:
                new_holds = []
                budget_left = buy_budget
                for snap_obj in picks:
                    if budget_left <= 0:
                        break
                    code = snap_obj.code
                    # Existing positions are carried forward.  Replacing the
                    # Pos object here would erase the old shares without a
                    # sale and falsely destroy portfolio value.
                    if code in self.positions:
                        continue
                    ind = _industry_of(code)
                    if SECTOR_MAX_STOCKS > 0 and ind and ind_count.get(ind, 0) >= SECTOR_MAX_STOCKS:
                        continue
                    mkt = load_market(code, qfq=False)
                    if mkt is None or D not in mkt.index:
                        continue
                    price = float(mkt.loc[D, 'close'])
                    if price <= 0:
                        continue
                    target = min(total_nav * MAX_WEIGHT, budget_left, self.cash)
                    if SECTOR_WEIGHT_CAP > 0 and ind:
                        room = SECTOR_WEIGHT_CAP * total_nav - ind_value.get(ind, 0.0)
                        if room <= 0:
                            continue
                        target = min(target, room)
                    shares = int(target / price / 100) * 100
                    if shares <= 0:
                        continue
                    amount = shares * price
                    fee = buy_fee(amount)
                    if amount + fee > self.cash:
                        shares = int((self.cash - fee) / price / 100) * 100
                        amount = shares * price
                        fee = buy_fee(amount)
                    if shares <= 0 or amount + fee > self.cash or amount > budget_left:
                        continue
                    self.cash -= amount + fee
                    uname = _name_of(code)
                    pos = Pos(code=code, name=uname, shares=shares,
                              buy_price=price, invested=amount,
                              anchor_qfq=price, buy_date=D)
                    self.positions[code] = pos
                    new_holds.append(pos)
                    ind_count[ind] = ind_count.get(ind, 0) + 1
                    ind_value[ind] = ind_value.get(ind, 0.0) + amount
                    budget_left -= amount
                    self.trades.append({'date': D, 'code': code, 'name': uname,
                                        'action': 'buy', 'shares': shares,
                                        'price': price, 'proceeds': amount, 'fee': fee,
                                        'reason': '通过防御核心条件+硬门禁, 安全边际优先/行业封顶'})
            # 记录本调仓日之后的账户状态 (用于日度净值 + 防御性报表)
            snap = list(self.positions.values())
            self._snapshots.append((D, self.cash, snap))
            indw = {k: v for k, v in ind_value.items() if v > 0}
            self.periods.append({'date': D, 'signal_date': signal_D, 'gauge': g_pct, 'eq_target': eq_w,
                                 'candidates': len(passed), 'cash': self.cash,
                                 'industry': indw})
            if verbose:
                print(f'  {D.date()} 候选{len(passed)} 目标权益{eq_w:.0%} '
                      f'现金缓冲后买入{len(new_holds) if picks else 0}只')
            i += 1

        # 5. 日度净值 (按调仓日之间的持仓/现金快照右续)
        cal = trading_days()
        first_exec = self._next_trading_day(dates[0]) if dates else None
        last_day = pd.Timestamp(end) if end is not None else pd.Timestamp(END)
        cal = cal[(cal >= (first_exec or dates[0])) & (cal <= last_day)]
        rows = {}
        hrows = {}
        si = 0
        n_snap = len(self._snapshots)
        for t0 in cal:
            while si + 1 < n_snap and self._snapshots[si + 1][0] <= t0:
                si += 1
            _, cash, poses = self._snapshots[si]
            val = cash
            held = 0.0
            for pos in poses:
                pv, cash_div = self._position_value(pos, t0)
                val += pv + cash_div
                held += pv
            rows[t0] = val
            hrows[t0] = held
        self.nav_curve = pd.Series(rows)
        self.holdings_curve = pd.Series(hrows)
        return self.nav_curve, self.trades, dates


# ─────────────────────────────────────────────────────────────
# 绩效口径
# ─────────────────────────────────────────────────────────────

def perf_metrics(nav: pd.Series, rf=0.02) -> dict:
    if len(nav) < 20:
        return {}
    rel = nav / nav.iloc[0]                       # 归一化 (nav 可为绝对金额)
    daily = rel.pct_change().dropna()
    years = max((rel.index[-1] - rel.index[0]).days / 365.25, 1 / 365.25)
    ann = rel.iloc[-1] ** (1 / years) - 1
    vol = daily.std() * np.sqrt(252)
    sharpe = (ann - rf) / vol if vol > 0 else 0.0
    dd = (rel / rel.expanding().max() - 1).min()
    win = (daily > 0).mean()
    return {'total_return': rel.iloc[-1] - 1, 'annual_return': ann,
            'annual_vol': vol, 'sharpe': sharpe, 'max_drawdown': dd,
            'win_rate': win, 'trading_days': len(daily)}


if __name__ == '__main__':
    print('module ok')
