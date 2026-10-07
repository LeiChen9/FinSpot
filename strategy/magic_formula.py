#!/usr/bin/env python
"""格林布拉特《股市稳赚》神奇公式 - A股 季度版回测核心逻辑 (无泄漏)

零重抓近似口径 (基于 graham_dodd 既有 sina 缓存):
    收益率(E/P) = 归母净利TTM / 总市值
    回报率(ROE) = 归母净利TTM / 归母净资产

排名: 全市场按 E/P 与 ROE 分别降序排名(第1名=1分), 综合分 = 两者之和, 越小越好;
组合: 每季度取综合分最靠前 N 只等权, 季度首交易日落仓日全换。

初选池 (防御性):
  1. 市值 ≥ 当日全A可交易市值 30% 分位 (与格雷厄姆口徑一致)
  2. 排除 金融行业 / 房地产 (高杠杆使ROE跨行业不可比 + 盈利不可持续使E/P失真)
  3. 排除 ST/*ST
  4. 上市满 1 年 (代理: 首根kline≤锚点则视为老股, 否则按首根kline起始日判定)
  5. E/P > 0 且 ROE > 0 (负值直接排除, 不入排名)

无泄漏约定与格雷厄姆策略一致:
  - 财报/分红仅使用 公告日期 ≤ 调仓日 D 的记录 (snapshot 内部保证)
  - 调仓日收盘价成交; D 无成交价 → 判不可交易
  - 成本: 佣金万3双向(单笔最低5元) + 卖出印花税(2023-08-28前0.1%/后0.05%) + 过户费0.001%双向
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy.graham_dodd import (
    SIZE_QUANTILE,
    load_universe, load_balance, load_profit, load_market,
    trading_days, rebalance_dates, industry_of, buy_fee, sell_fee, perf_metrics,
    snapshot, is_st_name, Pos,
)

N_HOLDINGS = 30                      # 组合只数 (等权)
EXCLUDE_INDUSTRIES = {'金融行业', '房地产'}   # 高杠杆/盈利不可持续 → ROE/E/P 失真的行业
# 已知强周期行业 (行业先验): 供给侧/大宗/高经营杠杆 → 盈利均值回归快而猛, 当期 ROE/E/P 顶值即危险点
CYCLICAL_INDUSTRIES = frozenset({
    '钢铁行业', '煤炭行业', '有色金属', '石油行业', '水泥行业', '化工行业',
    '农药化肥', '化纤行业', '玻璃行业', '陶瓷行业', '造纸行业', '船舶制造',
    '交通运输', '农林牧渔', '建筑建材', '纺织行业',
})
CYCLICAL_MODE_NONE = 'none'          # 基线: 周期行业照常按 ROE 降序排名
CYCLICAL_MODE_REVERSE = 'reverse'    # 反向信号: 周期行业 ROE 改为升序排名 (盈利顶值=最差)
CYCLICAL_MODE_EXCLUDE = 'exclude'    # 直接排除出股票池 (同金融/地产处理)
DEFAULT_CYCLICAL_MODE = CYCLICAL_MODE_EXCLUDE   # 主口径: 剔除周期行业 (基线 -8.6% vs exclude -1.0%)
IPO_MIN_YEARS = 1                    # 上市满 N 年 (代理口径)
MAGIC_LABELS = {'magic': '神奇公式(季度全换Top30)'}

FUNNEL_MF: Dict[pd.Timestamp, dict] = {}   # 每次全市场扫描后记录的初选池漏斗
_POOL_MF: Dict[pd.Timestamp, pd.DataFrame] = {}   # 每期全市场扫描池缓存 (key=(D), 供各模式派生)

K_ANCHOR = pd.Timestamp('2022-11-01')      # 抓取窗口起点 (老股 kline 统一锚定于此)


def listing_proxy_years(code: str, D: pd.Timestamp) -> Optional[float]:
    """上市年限代理: 无 kline 用最早报表日, 否则首根 kline (≤锚点视为老股)。"""
    frame = load_market(code, qfq=False)
    if frame is None or frame.empty:
        earliest = None
        for report in (load_balance(code), load_profit(code)):
            if report is not None and len(report):
                value = pd.to_datetime(report["报告日"], errors="coerce").dropna().min()
                if earliest is None or value < earliest:
                    earliest = value
        return None if earliest is None or pd.isna(earliest) else (D - earliest).days / 365.25
    first = frame.index.min()
    return 1.0e9 if first <= K_ANCHOR + pd.Timedelta(days=90) else (D - first).days / 365.25


is_st = is_st_name


def rank_magic_candidates(frame: pd.DataFrame, mode: str, exclude_mode: str,
                          reverse_mode: str, cyclical: frozenset) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.copy()
    if mode == exclude_mode:
        ranked = ranked[~ranked["cyclical"]].copy()
    ranked["rank_ep"] = ranked["ep"].rank(ascending=False, method="min").astype(int)
    if mode == reverse_mode:
        asc = ranked["roe"].rank(ascending=True, method="min").astype(int)
        desc = ranked["roe"].rank(ascending=False, method="min").astype(int)
        ranked["rank_roe"] = np.where(ranked["cyclical"], asc, desc)
    else:
        ranked["rank_roe"] = ranked["roe"].rank(ascending=False, method="min").astype(int)
    ranked["composite"] = ranked["rank_ep"] + ranked["rank_roe"]
    return ranked.sort_values(["composite", "rank_ep", "code"]).reset_index(drop=True)


def screen_pool(D: pd.Timestamp) -> pd.DataFrame:
    """全市场一次性扫描 → 通过防御性初选的底池 (含 cyclical 标记, 未排名)。

    与具体"周期行业处置模式"无关, 全程只扫一次, 缓存于 _POOL_MF,
    各模式 (none/reverse/exclude) 均在底池上派生排名, 避免 3× 重复扫描。
    """
    if D in _POOL_MF:
        return _POOL_MF[D]
    uni = load_universe()
    rows = []
    caps = []
    st_hit = ind_hit = age_hit = neg_hit = missing = 0
    for code in uni['code']:
        s = snapshot(code, D)
        if not s.tradable:
            missing += 1
            continue
        if not (np.isfinite(s.market_cap) and s.market_cap > 0):
            missing += 1
            continue
        # 防御性初选
        if is_st(s.name):
            st_hit += 1
            continue
        ind = industry_of(code)
        if ind in EXCLUDE_INDUSTRIES:
            ind_hit += 1
            continue
        age = listing_proxy_years(code, D)
        if age is None or age < IPO_MIN_YEARS:
            age_hit += 1
            continue
        # 指标
        eps_ttm, price, bvps = s.eps_ttm, s.price, s.bvps
        if not (np.isfinite(eps_ttm) and eps_ttm > 0 and np.isfinite(price) and price > 0):
            neg_hit += 1
            continue
        if not (np.isfinite(bvps) and bvps > 0):
            neg_hit += 1
            continue
        ep = eps_ttm / price
        roe = eps_ttm / bvps
        if ep <= 0 or roe <= 0:
            neg_hit += 1
            continue
        rows.append({'code': code, 'name': s.name, 'industry': ind,
                     'market_cap': float(s.market_cap), 'price': float(price),
                     'eps_ttm': float(eps_ttm), 'bvps': float(bvps),
                     'ep': float(ep), 'roe': float(roe)})
        caps.append(float(s.market_cap))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 市值分位下限 → 剔除底部 SIZE_QUANTILE
    floor = float(pd.Series(caps).quantile(SIZE_QUANTILE)) if caps else np.inf
    df = df[df['market_cap'] >= floor].copy()
    size_hit = int((pd.Series(caps) < floor).sum())
    df['cyclical'] = df['industry'].isin(CYCLICAL_INDUSTRIES)
    FUNNEL_MF[D] = {'tradable': len(uni) - missing, 'missing': missing, 'st': st_hit,
                    'industry': ind_hit, 'age': age_hit, 'negative': neg_hit,
                    'size': size_hit, 'cyclical_exclude': 0, 'pool': len(df)}
    _POOL_MF[D] = df
    return df


def rank_candidates(D: pd.Timestamp, cyclical_mode: str = DEFAULT_CYCLICAL_MODE) -> pd.DataFrame:
    """在缓存底池上按模式派生排名 (排名本身是向量化, 秒级)。

    cyclical_mode:
      - 'none'    基线: 周期行业照常 ROE 降序
      - 'reverse' 反向信号: 周期行业内 ROE 升序排名 (盈利顶值=最差分)
      - 'exclude' 直接从初选池剔除周期行业
    """
    df = screen_pool(D)
    if df.empty:
        return pd.DataFrame()
    df = rank_magic_candidates(
        df, cyclical_mode, CYCLICAL_MODE_EXCLUDE,
        CYCLICAL_MODE_REVERSE, CYCLICAL_INDUSTRIES,
    )
    if cyclical_mode == CYCLICAL_MODE_EXCLUDE:
        # 更新漏斗: exclude 模式下的周期剔除数
        cyc_excl = int((screen_pool(D)['cyclical']).sum())
        f = dict(FUNNEL_MF[D]); f['cyclical_exclude'] = cyc_excl; f['pool'] = len(df)
        FUNNEL_MF[D] = f
    return df


def pick_top(df: pd.DataFrame, n: int = N_HOLDINGS) -> pd.DataFrame:
    return df.head(n) if len(df) else df


def build_ranking_map(dates: List[pd.Timestamp], n: int = N_HOLDINGS,
                      cyclical_mode: str = DEFAULT_CYCLICAL_MODE,
                      verbose: bool = True) -> Dict[pd.Timestamp, pd.DataFrame]:
    """预先为每个调仓日计算 Top-N (供回测与展示复用, 避免重复全市场扫描)。"""
    mp = {}
    for d in dates:
        mp[d] = pick_top(rank_candidates(d, cyclical_mode=cyclical_mode), n=n)
        if verbose:
            print(f'  {d.date()} 候选{len(mp[d])}只 '
                  f'({"、".join(mp[d]["name"].head(5).tolist())}{"…" if len(mp[d]) > 5 else ""})')
    return mp


def cyclical_comparison(initial_capital: float = 1_000_000.0):
    """回测周期行业 none/reverse/exclude 三种处置口径并返回净值对比表。"""
    modes = {
        CYCLICAL_MODE_NONE: '基线(周期ROE降序)',
        CYCLICAL_MODE_REVERSE: '反向信号(周期ROE升序)',
        CYCLICAL_MODE_EXCLUDE: '排除周期行业',
    }
    dates = rebalance_dates()
    for date in dates:
        screen_pool(date)
    navs = {}
    for mode in modes:
        ranking_map = {
            date: pick_top(rank_candidates(date, cyclical_mode=mode), n=N_HOLDINGS)
            for date in dates
        }
        backtest = MagicBacktest(initial_capital=initial_capital, n=N_HOLDINGS)
        navs[mode], _, _ = backtest.run(dates, ranking_map=ranking_map, verbose=False)

    base = navs[CYCLICAL_MODE_NONE] / initial_capital
    hs = load_market('000300')
    hs = hs.loc[hs.index >= base.index[0], 'close']
    hs = hs / hs.iloc[0]
    zz = load_market('000906')
    zz = zz.loc[zz.index >= base.index[0], 'close']
    zz = zz / zz.iloc[0]
    comparison = pd.DataFrame({modes[mode]: navs[mode] / initial_capital for mode in modes})
    comparison['沪深300'] = hs
    comparison['中证800'] = zz
    comparison = comparison.dropna(how='all').ffill().dropna(subset=[modes[CYCLICAL_MODE_NONE]])
    return dates, navs, modes, comparison


# ─────────────────────────────────────────────────────────────
# 回测器 (独立简版: 季度全换, 无两档卖出/降仓/行业封顶)
# ─────────────────────────────────────────────────────────────

MPos = Pos


class MagicBacktest:
    def __init__(self, initial_capital: float = 1_000_000.0, n: int = N_HOLDINGS):
        self.initial_capital = initial_capital
        self.n = n
        self.cash = initial_capital
        self.positions: Dict[str, MPos] = {}
        self.trades: List[dict] = []
        self.nav_curve: pd.Series = pd.Series(dtype=float)
        self._snapshots: List[tuple] = []

    # —— 内部: 调仓日市值/卖出 ——
    def _close_at(self, code: str, pos: MPos, D: pd.Timestamp) -> Optional[float]:
        """qfq 相对锚点 → 相对收益; 缺失回退不复权。"""
        q = load_market(code, qfq=True)
        if q is not None and pos.anchor_qfq > 0:
            prev = q.index[q.index <= D]
            if len(prev):
                return float(q.loc[prev[-1], 'close']) / pos.anchor_qfq
        raw = load_market(code, qfq=False)
        if raw is not None and np.isfinite(pos.buy_price) and pos.buy_price > 0:
            prev = raw.index[raw.index <= D]
            if len(prev):
                return float(raw.loc[prev[-1], 'close']) / pos.buy_price
        return None

    def _sell_all(self, D: pd.Timestamp, reason: str = '季度全换'):
        specs = [(code, pos) for code, pos in self.positions.items()]
        for code, pos in specs:
            ratio = self._close_at(code, pos, D)
            if ratio is None:
                self.cash += pos.invested  # 无价: 按成本回收(退市近似)
                proceed = pos.invested
            else:
                proceed = pos.invested * ratio
            fee = sell_fee(D, proceed)
            self.cash += proceed - fee
            self.positions.pop(code)
            self.trades.append({'date': D, 'code': code, 'name': pos.name,
                                'action': 'sell', 'shares': pos.shares,
                                'price': np.nan, 'proceeds': proceed, 'fee': fee,
                                'reason': reason})

    def _buy(self, D: pd.Timestamp, picks: pd.DataFrame):
        budget = self.cash / max(1, len(picks))   # 等权
        for _, r in picks.iterrows():
            code = r['code']
            mkt = load_market(code, qfq=False)
            if mkt is None or D not in mkt.index:
                continue
            price = float(mkt.loc[D, 'close'])
            if price <= 0:
                continue
            budget_left = max(0.0, self.cash) - 0.0
            target = min(budget, budget_left)
            shares = int(target / price / 100) * 100
            if shares <= 0:
                continue
            amount = shares * price
            fee = buy_fee(amount)
            if amount + fee > self.cash:
                shares = int((self.cash - fee) / price / 100) * 100
                amount = shares * price
                fee = buy_fee(amount)
            if shares <= 0 or amount + fee > self.cash:
                continue
            self.cash -= amount + fee
            q = load_market(code, qfq=True)
            qd = float(q.loc[D, 'close']) if q is not None and D in q.index else price
            if not np.isfinite(qd) or qd <= 0:
                qd = price
            pos = MPos(code=code, name=r['name'], shares=shares,
                       buy_price=price, invested=amount, anchor_qfq=qd, buy_date=D)
            self.positions[code] = pos
            self.trades.append({'date': D, 'code': code, 'name': r['name'],
                                'action': 'buy', 'shares': shares,
                                'price': price, 'proceeds': amount, 'fee': fee,
                                'reason': f'综合分Top{self.n}等权买入'})

    def run(self, dates: List[pd.Timestamp],
            ranking_map: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
            verbose: bool = True) -> Tuple[pd.Series, list, list]:
        for i, D in enumerate(dates):
            if ranking_map is not None:
                picks = ranking_map.get(D, pd.DataFrame())
            else:
                picks = pick_top(rank_candidates(D), n=self.n)
            self._sell_all(D)
            if len(picks):
                self._buy(D, picks)
            snap = [(code, p.invested, float(p.anchor_qfq), float(p.buy_price))
                    for code, p in self.positions.items()]
            self._snapshots.append((D, self.cash, snap))
            if verbose:
                print(f'  {D.date()} 持仓{len(self.positions)}只 现金{self.cash:,.0f}')

        # 日度净值 (快照右续)
        cal = trading_days()
        cal = cal[(cal >= dates[0]) & (cal <= dates[-1])]
        rows = {}
        si = 0
        n_snap = len(self._snapshots)
        for t0 in cal:
            while si + 1 < n_snap and self._snapshots[si + 1][0] <= t0:
                si += 1
            _, cash, poses = self._snapshots[si]
            val = cash
            for code, invested, anchor, buy_price in poses:
                qf = load_market(code, qfq=True)
                ratio = None
                if qf is not None and anchor > 0:
                    prev = qf.index[qf.index <= t0]
                    if len(prev):
                        ratio = float(qf.loc[prev[-1], 'close']) / anchor
                if ratio is None:
                    raw = load_market(code, qfq=False)
                    if raw is not None and np.isfinite(buy_price) and buy_price > 0:
                        prev = raw.index[raw.index <= t0]
                        if len(prev):
                            ratio = float(raw.loc[prev[-1], 'close']) / buy_price
                val += invested * ratio if ratio is not None else invested
            rows[t0] = val
        self.nav_curve = pd.Series(rows)
        return self.nav_curve, self.trades, dates


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CYCLICAL_MODE
    d = rebalance_dates()[0]
    df = rank_candidates(d, cyclical_mode=mode)
    print(f'{d.date()} 候选 {len(df)} 只 (mode={mode}) → Top{N_HOLDINGS}:')
    print(pick_top(df, N_HOLDINGS)[
        ['code', 'name', 'industry', 'cyclical', 'ep', 'roe', 'rank_ep', 'rank_roe', 'composite', 'market_cap']].to_string(index=False))
