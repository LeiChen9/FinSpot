"""目标权重组合回测引擎: 定期再平衡至目标权重, 日度净值与成交记录。

全天候/永久组合等资产配置策略共用本引擎:
  - 再平衡日按目标权重全组合调仓 (先卖后买)
  - 可选单资产权重硬顶 (超出行 > 0)
"""
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class BTResult:
    daily_nav: pd.Series = field(default_factory=pd.Series)
    daily_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly_pos: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: List[dict] = field(default_factory=list)
    total_cost: float = 0.0


def quarterly_rebalance_dates(dates, start, end) -> List[pd.Timestamp]:
    """每个季度首个交易日。"""
    days = sorted([d for d in dates if start <= d <= end])
    if not days:
        return []
    out, cur_q = [], None
    for d in days:
        q = (d.year, d.quarter)
        if q != cur_q:
            out.append(d)
            cur_q = q
    return out


def run_weight_backtest(all_data: Dict[str, pd.DataFrame], target_w: Dict[str, float],
                        rebal_dates: List[pd.Timestamp],
                        init_cap: float = 1e6,
                        comm: float = 0.0003, tax: float = 0.0005,
                        cap_code: str = None, cap: float = None) -> BTResult:
    """按目标权重回测。

    cap_code/cap: 可选的单一资产权重硬顶, 超出部分按其余正权重等比分摊。
    """
    result = BTResult()
    all_dates = sorted(set(d for df in all_data.values() for d in df.index))
    all_dates = [d for d in all_dates if rebal_dates[0] <= d <= rebal_dates[-1]]
    rebal_set = set(rebal_dates)

    nav = init_cap
    shares: Dict[str, float] = {}
    cost_acc = 0.0
    nav_recs, w_recs, pos_recs, trades = [], [], [], []

    def gc(c, d):
        df = all_data.get(c)
        if df is not None and d in df.index:
            return float(df.loc[d, 'close'])
        return None

    def calc_nav(d):
        return sum(sh * (gc(c, d) or 0) for c, sh in shares.items())

    def apply_cap(w, cap_):
        if cap_ is None or cap_code not in w:
            return dict(w)
        w = dict(w)
        if w[cap_code] <= cap_:
            return w
        over = w[cap_code] - cap_
        w[cap_code] = cap_
        others = {c: v for c, v in w.items() if c != cap_code and v > 0}
        s = sum(others.values())
        if s > 0:
            for c in others:
                w[c] += over * others[c] / s
        return w

    for day in all_dates:
        if day in rebal_set:
            nav = calc_nav(day)
            if nav <= 0:
                nav = init_cap
            tw = apply_cap(target_w, cap)
            tgt = {c: nav * v for c, v in tw.items() if v > 0}
            cur = {c: shares.get(c, 0) * (gc(c, day) or 0)
                   for c in shares if shares.get(c, 0) > 0}
            for c in list(shares):
                cv = cur.get(c, 0)
                tv = tgt.get(c, 0)
                if c not in tgt or tv < cv:
                    p = gc(c, day)
                    if p is None or p <= 0 or shares[c] <= 0:
                        continue
                    amt = cv - tv
                    if amt <= 0:
                        continue
                    sh = min(shares[c], amt / p)
                    proceeds = sh * p
                    cost = proceeds * (comm + tax)
                    cost_acc += cost
                    shares[c] -= sh
                    nav += proceeds - cost
                    trades.append({'date': day, 'code': c, 'action': 'sell',
                                   'shares': sh, 'price': p, 'amount': proceeds, 'cost': cost})
            shares = {c: s for c, s in shares.items() if s > 1e-8}
            cur2 = {c: shares.get(c, 0) * (gc(c, day) or 0)
                    for c in shares if shares.get(c, 0) > 0}
            for c, tv in tgt.items():
                cv2 = cur2.get(c, 0)
                if tv <= cv2:
                    continue
                p = gc(c, day)
                if p is None or p <= 0:
                    continue
                amt = tv - cv2
                cost = amt * comm
                cost_acc += cost
                shares[c] = shares.get(c, 0) + amt / p
                nav -= amt + cost
                trades.append({'date': day, 'code': c, 'action': 'buy',
                               'shares': amt / p, 'price': p, 'amount': amt, 'cost': cost})
            pos = {'date': day}
            for c in shares:
                p = gc(c, day)
                if p is not None:
                    pos[c] = shares[c] * p
            pos_recs.append(pos)
        tv = calc_nav(day)
        if tv > 0:
            nav_recs.append({'date': day, 'nav': tv})
            wr = {'date': day}
            for c in set(list(shares) + list(target_w)):
                p = gc(c, day)
                if p and c in shares and shares[c] > 0:
                    wr[c] = shares[c] * p / tv
                else:
                    wr[c] = 0.0
            w_recs.append(wr)

    if nav_recs:
        df = pd.DataFrame(nav_recs).set_index('date')
        result.daily_nav = df['nav'] / df['nav'].iloc[0]
    if w_recs:
        result.daily_weights = pd.DataFrame(w_recs).set_index('date')
    if pos_recs:
        result.quarterly_pos = pd.DataFrame(pos_recs).set_index('date')
    result.trades = trades
    result.total_cost = cost_acc
    return result
