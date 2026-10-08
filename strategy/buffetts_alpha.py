"""Buffett's Alpha A股: 大盘 + 便宜/安全/质量 三门 + 复合 z 分选股, 季度全换。

锁定口径:
  - 大盘 = 总市值>300亿 + 300亿内近60日累计成交金额前300 + 近7日均成交金额≥2亿
  - 便宜: E/P>0.02 (PE<50) 且 B/P<2 (PB>0.5), 周期/未知行业排除
  - 安全: 收缩 Beta < 1.5 (0.6×收缩贝塔 + 0.4)
  - 质量: 股息>3% 或 ROE>10%
  - 合成: composite=(value_z+safe_z+quality_z)/3, quality_z=(zROE+zTTM yoy+zD/P)/3
  - 组合: Top20, 行业 ≤5 只; 5% 等权建仓, 日度 10% 砍回 5%, 季度信号 D+1 全换
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from strategy.graham_dodd import Pos, buy_fee, sell_fee, trading_days
from strategy.graham_dodd import GrahamStrategy, load_index, load_market, load_universe, snapshot
from strategy.magic_formula import CYCLICAL_INDUSTRIES

CAP_FLOOR = 300e8
AMT7_FLOOR = 2e8
BETA_MAX = 1.5
QROE = 0.10
QDIV = 0.03
N_TOP = 20
IND_CAP = 5

_mkt_index_ret = None


def amounts(code: str, D: pd.Timestamp) -> Tuple[float, float]:
    """(近60日累计成交额, 近7日均成交额)。"""
    m = load_market(code)
    if m is None or D not in m.index:
        return np.nan, np.nan
    w = m.loc[:D].tail(60)
    a = w['close'] * w['volume']
    return a.sum(), a.tail(7).mean()


def big_pool(D: pd.Timestamp) -> pd.DataFrame:
    """大盘定义: 市值门槛 + 流动性门槛, 按近60日成交额取前300。"""
    uni = load_universe()
    rows = []
    fun = {'n': len(uni), 'notrad': 0, 'cap': 0}
    for code in uni['code']:
        s = snapshot(code, D)
        if not s.tradable or not np.isfinite(s.market_cap):
            fun['notrad'] += 1
            continue
        if s.market_cap < CAP_FLOOR:
            fun['cap'] += 1
            continue
        a60, a7 = amounts(code, D)
        rows.append((code, s.name, s.industry, s.market_cap, a60, a7))
    df = pd.DataFrame(rows, columns=['code', 'name', 'industry', 'cap', 'amt60', 'amt7']).dropna()
    df = df[df['amt7'] >= AMT7_FLOOR].sort_values('amt60', ascending=False).head(300)
    df.attrs['funnel'] = fun
    return df


def zscore(s):
    r = s.rank(method='min')
    return (r - r.mean()) / r.std()


def beta_of(code: str, D: pd.Timestamp) -> float:
    """收缩贝塔: 0.6×市场贝塔 + 0.4 (近260日, 至少120日重叠)。"""
    global _mkt_index_ret
    if _mkt_index_ret is None:
        midx = load_index('000300')
        _mkt_index_ret = np.log(midx['close'] / midx['close'].shift(1))
    m = load_market(code)
    if m is None or D not in m.index:
        return np.nan
    w = m.loc[:D].tail(260)
    r = np.log(w['close'] / w['close'].shift(1)).dropna()
    mr = _mkt_index_ret
    a = r.index.intersection(mr.index)
    if len(a) < 120:
        return np.nan
    return 0.6 * np.cov(r.loc[a], mr.loc[a])[0, 1] / mr.loc[a].var() + 0.4


def value_rank(D: pd.Timestamp, codes) -> Tuple[pd.DataFrame, dict]:
    """价值池: 三因子 (E/P, B/P, D/P) 等权 z 排名 + 硬门。"""
    rows = []
    fun = {'cyc': 0, 'neg': 0, 'cheap': 0}
    for code in codes:
        s = snapshot(code, D)
        if not s.industry or s.industry in CYCLICAL_INDUSTRIES:
            fun['cyc'] += 1
            continue
        if not (np.isfinite(s.eps_ttm) and s.eps_ttm > 0 and np.isfinite(s.bvps) and s.bvps > 0):
            fun['neg'] += 1
            continue
        ep = s.eps_ttm / s.price
        bp = s.bvps / s.price
        if not (ep > 0.02 and bp < 2):
            fun['cheap'] += 1
            continue
        dp = s.dividend_yield if np.isfinite(s.dividend_yield) else 0.0
        rows.append((code, s.name, s.industry, ep, bp, dp))
    v = pd.DataFrame(rows, columns=['code', 'name', 'industry', 'ep', 'bp', 'dp'])
    v['value_z'] = (zscore(v['ep']) + zscore(v['bp']) + zscore(v['dp'])) / 3
    return v.sort_values('value_z', ascending=False).reset_index(drop=True), fun


def gated_pool(D: pd.Timestamp, codes) -> Tuple[pd.DataFrame, dict]:
    """三门 (便宜+安全+质量) 后的候选池。"""
    rows = []
    fun = {'cyc': 0, 'neg': 0, 'cheap': 0, 'beta': 0, 'qual': 0}
    for code in codes:
        s = snapshot(code, D)
        if not s.industry or s.industry in CYCLICAL_INDUSTRIES:
            fun['cyc'] += 1
            continue
        if not (np.isfinite(s.eps_ttm) and s.eps_ttm > 0 and np.isfinite(s.bvps) and s.bvps > 0):
            fun['neg'] += 1
            continue
        if not (s.eps_ttm / s.price > 0.02 and s.bvps / s.price < 2):
            fun['cheap'] += 1
            continue
        be = beta_of(code, D)
        if not np.isfinite(be) or be >= BETA_MAX:
            fun['beta'] += 1
            continue
        roe = s.eps_ttm / s.bvps
        div = s.dividend_yield
        if not ((np.isfinite(div) and div > QDIV) or roe > QROE):
            fun['qual'] += 1
            continue
        rows.append((code, s.name, s.industry, s.eps_ttm / s.price, s.bvps / s.price,
                     div if np.isfinite(div) else 0, roe, be))
    pool = pd.DataFrame(rows, columns=['code', 'name', 'industry', 'ep', 'bp', 'dp', 'roe', 'beta'])
    return pool, fun


def add_composite(pool: pd.DataFrame, D: pd.Timestamp) -> pd.DataFrame:
    """合成 (已锁定): composite=(value_z+safe_z+quality_z)/3。"""
    pool = pool.copy()
    pool['qyoy'] = [snapshot(c, D).ttm_yoy for c in pool['code']]
    pool['value_z'] = (zscore(pool['ep']) + zscore(pool['bp']) + zscore(pool['dp'])) / 3
    pool['safe_z'] = -zscore(pool['beta'])
    pool['quality_z'] = (zscore(pool['roe']) + zscore(pool['qyoy']) + zscore(pool['dp'])) / 3
    pool['composite'] = (pool['value_z'] + pool['safe_z'] + pool['quality_z']) / 3
    return pool.sort_values('composite', ascending=False)


def select_topN(D: pd.Timestamp, n: int = N_TOP) -> pd.DataFrame:
    """大盘 + 三门 + 合成分 + 行业≤IND_CAP 的最终持仓。"""
    lc = []
    for code in load_universe()['code']:
        s = snapshot(code, D)
        if not s.tradable or not np.isfinite(s.market_cap) or s.market_cap < CAP_FLOOR:
            continue
        a60, a7 = amounts(code, D)
        if np.isfinite(a7) and a7 >= AMT7_FLOOR:
            lc.append((code, a60))
    codes = [c for c, _ in sorted(lc, key=lambda x: -x[1])[:300]]
    pool, _ = gated_pool(D, codes)
    if pool.empty:
        return pool
    p = add_composite(pool, D)
    out = []
    cnt = {}
    for _, r in p.iterrows():
        if cnt.get(r['industry'], 0) >= IND_CAP:
            continue
        out.append(r)
        cnt[r['industry']] = cnt.get(r['industry'], 0) + 1
        if len(out) >= n:
            break
    return pd.DataFrame(out)


def quarter_dates(start: pd.Timestamp, hard_end: pd.Timestamp) -> List[pd.Timestamp]:
    """从 start 起每季度首个交易日。"""
    CAL = trading_days()
    qs = []
    q = start
    while q <= hard_end:
        s = CAL[CAL >= q]
        if len(s):
            qs.append(pd.Timestamp(s[0]))
        q = (pd.Timestamp(year=q.year, month=q.month, day=1) + pd.DateOffset(months=3))
    return qs


def run_full(start: pd.Timestamp, hard_end: pd.Timestamp, capital: float = 500_000.0,
             maps: List[pd.DataFrame] = None, dates: List[pd.Timestamp] = None):
    """季度全换回测: 5% 等权建仓, 日度 10% 砍回 5%。

    maps: 预计算的各期 select_topN 结果 (可并行生成)。
    dates: 可直接指定信号日; 用于单期验证。
    返回 (nav, positions, ntrim, quarter_dates)。
    """
    CAL = trading_days()
    qs = dates if dates is not None else quarter_dates(start, hard_end)
    if maps is None:
        maps = [select_topN(D) for D in qs]

    GB = GrahamStrategy
    cash = capital
    poss = {}
    navs = {}
    ntrim = 0
    for i, D in enumerate(qs):
        nxt = pd.Timestamp(CAL[CAL > D][0])
        Q1 = qs[i + 1] if i + 1 < len(qs) else hard_end
        for code, pos in list(poss.items()):
            liq = GB._liquidation_value(pos, nxt)
            cash += liq - sell_fee(nxt, liq)
            poss.pop(code)
        tot0 = cash
        for _, r in maps[i].iterrows():
            m = load_market(r['code'])
            if m is None or nxt not in m.index:
                continue
            p = float(m.loc[nxt, 'close'])
            sh = int(tot0 * 0.05 / p / 100) * 100
            if sh <= 0:
                continue
            amt = sh * p
            cash -= amt + buy_fee(amt)
            poss[r['code']] = Pos(r['code'], r['name'], sh, p, amt, p, nxt)
        for t in CAL[(CAL >= nxt) & (CAL <= Q1)]:
            tot = cash
            mvs = {}
            for code, pos in poss.items():
                mv, dv = GB._position_value(pos, t)
                mvs[code] = mv
                tot += mv + dv
            for code, pos in list(poss.items()):
                if mvs[code] / tot > 0.10:
                    m = load_market(code)
                    prev = m.index[m.index <= t]
                    if not len(prev):
                        continue
                    p = float(m.loc[prev[-1], 'close'])
                    sh = int((mvs[code] - tot * 0.05) / p / 100) * 100
                    if sh <= 0:
                        continue
                    proc = sh * p
                    cash += proc - sell_fee(t, proc)
                    pos.shares -= sh
                    ntrim += 1
            navs[t] = tot
    nav = pd.Series(navs)
    return nav, poss, ntrim, qs
