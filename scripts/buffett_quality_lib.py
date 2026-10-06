"""A conservative, point-in-time Buffett quality proxy for A shares.

It deliberately does not claim to be Buffett's full investment process.  The
local dataset has point-in-time balance sheets and income statements for most
of the market, but cash-flow data for only a handful of names.  The model thus
uses durable reported earnings, return on equity, leverage and valuation; it
does not substitute current-period E/P + ROE for EBIT/EV or owner earnings.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from graham_dodd_lib import (
    END, Pos, SIZE_QUANTILE, _industry_of, buy_fee, is_st_name, load_market,
    load_universe, sell_fee, snapshot, trading_days, GrahamBacktest,
)

N_HOLDINGS = 20
MIN_SIZE_QUANTILE = 0.50
EXCLUDED_INDUSTRIES = frozenset({
    '金融行业', '房地产', '钢铁行业', '煤炭行业', '有色金属', '石油行业',
    '水泥行业', '化工行业', '农药化肥', '化纤行业', '玻璃行业', '陶瓷行业',
    '造纸行业', '船舶制造', '交通运输', '农林牧渔', '建筑建材', '纺织行业',
})
FUNNEL_BQ: Dict[pd.Timestamp, dict] = {}


def annual_signal_dates() -> List[pd.Timestamp]:
    """First trading day of July: the prior annual report is normally public."""
    cal = trading_days()
    years = range(2023, pd.Timestamp(END).year + 1)
    out = []
    for year in years:
        d = pd.Timestamp(year=year, month=7, day=1)
        nxt = cal[cal >= d]
        if len(nxt) and nxt[0] <= pd.Timestamp(END):
            out.append(pd.Timestamp(nxt[0]))
    return out


def _five_year_quality(s) -> Optional[Tuple[float, int]]:
    eps = s.eps_annual[-5:]
    years = s.eps_years[-5:]
    if len(eps) != 5 or len(years) != 5 or min(eps) <= 0:
        return None
    if years[-1] - years[0] != 4:
        return None
    cagr = (eps[-1] / eps[0]) ** 0.25 - 1.0
    declines = sum(eps[i] < eps[i - 1] for i in range(1, 5))
    return cagr, declines


def rank_candidates(D: pd.Timestamp) -> pd.DataFrame:
    """Rank durable, profitable, reasonably priced businesses as of signal day."""
    uni = load_universe()
    rows, caps = [], []
    excluded = st = low_quality = missing = 0
    for code in uni['code']:
        s = snapshot(code, D)
        if not s.tradable or not np.isfinite(s.market_cap) or s.market_cap <= 0:
            missing += 1
            continue
        if is_st_name(s.name):
            st += 1
            continue
        # Unknown classifications do not bypass an industry safety rule.
        if not s.industry or s.industry in EXCLUDED_INDUSTRIES:
            excluded += 1
            continue
        q = _five_year_quality(s)
        pe = s.price / s.eps_ttm if np.isfinite(s.eps_ttm) and s.eps_ttm > 0 else np.nan
        roe = s.eps_ttm / s.bvps if np.isfinite(s.bvps) and s.bvps > 0 else np.nan
        if (q is None or not np.isfinite(pe) or not np.isfinite(roe)
                or pe > 30 or roe < 0.15
                or not np.isfinite(s.debt_ratio) or s.debt_ratio >= 1.0
                or not np.isfinite(s.current_ratio) or s.current_ratio < 1.0
                or q[0] < 0.05 or q[1] > 1):
            low_quality += 1
            continue
        rows.append({'code': code, 'name': s.name, 'industry': s.industry,
                     'market_cap': s.market_cap, 'pe': pe, 'roe': roe,
                     'eps_cagr_5y': q[0], 'eps_declines_5y': q[1],
                     'debt_to_equity': s.debt_ratio, 'current_ratio': s.current_ratio})
        caps.append(s.market_cap)
    if not rows:
        return pd.DataFrame()
    floor = float(pd.Series(caps).quantile(MIN_SIZE_QUANTILE))
    df = pd.DataFrame(rows)
    size_hit = int((df['market_cap'] < floor).sum())
    df = df[df['market_cap'] >= floor].copy()
    # Equal-weight ranks avoid one extreme ratio dominating the quality score.
    df['rank_pe'] = df['pe'].rank(ascending=True, method='min')
    df['rank_roe'] = df['roe'].rank(ascending=False, method='min')
    df['rank_growth'] = df['eps_cagr_5y'].rank(ascending=False, method='min')
    df['composite'] = df['rank_pe'] + df['rank_roe'] + df['rank_growth']
    df = df.sort_values(['composite', 'pe', 'code']).reset_index(drop=True)
    FUNNEL_BQ[D] = {'tradable': len(uni) - missing, 'st': st, 'industry': excluded,
                    'quality': low_quality, 'size': size_hit, 'floor': floor, 'final_pool': len(df)}
    return df


def build_ranking_map(dates: List[pd.Timestamp], n: int = N_HOLDINGS) -> Dict[pd.Timestamp, pd.DataFrame]:
    return {d: rank_candidates(d).head(n) for d in dates}


def audit_borderline(D: pd.Timestamp, top: int = 12) -> List[dict]:
    """逐项核对未入质量池的接近候选 (每一列为未达标项), 与 graham 的接近候选明细对应."""
    uni = load_universe()
    rows = []
    for code in uni['code']:
        s = snapshot(code, D)
        if not s.tradable or is_st_name(s.name):
            continue
        if not s.industry or s.industry in EXCLUDED_INDUSTRIES:
            continue
        q = _five_year_quality(s)
        pe = s.price / s.eps_ttm if np.isfinite(s.eps_ttm) and s.eps_ttm > 0 else np.nan
        roe = s.eps_ttm / s.bvps if np.isfinite(s.bvps) and s.bvps > 0 else np.nan
        fails = []
        if q is None:
            fails.append('近5年EPS不满足')
        else:
            if q[0] < 0.05:
                fails.append(f'CAGR {q[0]:.1%}<5%')
            if q[1] > 1:
                fails.append(f'下滑{q[1]}年>1')
        if not np.isfinite(pe) or pe > 30:
            fails.append(f'PE {pe:.1f}>30' if np.isfinite(pe) else 'PE缺')
        if not np.isfinite(roe) or roe < 0.15:
            fails.append(f'ROE {roe:.1%}<15%' if np.isfinite(roe) else 'ROE缺')
        if not np.isfinite(s.debt_ratio) or s.debt_ratio >= 1.0:
            fails.append(f'负债/权益 {s.debt_ratio:.2f}≥1' if np.isfinite(s.debt_ratio) else '负债/权益缺')
        if not np.isfinite(s.current_ratio) or s.current_ratio < 1.0:
            fails.append(f'流动比率 {s.current_ratio:.2f}<1' if np.isfinite(s.current_ratio) else '流动比率缺')
        if not fails:
            continue
        rows.append({'code': code, 'name': s.name, 'industry': s.industry,
                     'price': s.price, 'n_fails': len(fails), 'fails': fails})
    rows.sort(key=lambda x: (x['n_fails'], x['code']))
    return rows[:top]


@dataclass
class QualityBacktest:
    initial_capital: float = 1_000_000.0
    n: int = N_HOLDINGS

    def __post_init__(self):
        self.cash = self.initial_capital
        self.positions: Dict[str, Pos] = {}
        self.trades: List[dict] = []
        self.nav_curve = pd.Series(dtype=float)
        self.holdings_curve = pd.Series(dtype=float)
        self._snapshots = []
        self.periods: List[dict] = []   # 每调仓日: 候选/现金/行业分布 记录

    def _next_day(self, D):
        cal = trading_days()
        nxt = cal[cal > D]
        return pd.Timestamp(nxt[0]) if len(nxt) else None

    def _sell_all(self, D):
        for code, pos in list(self.positions.items()):
            proceeds = GrahamBacktest._liquidation_value(pos, D)
            fee = sell_fee(D, proceeds)
            self.cash += proceeds - fee
            self.positions.pop(code)
            self.trades.append({'date': D, 'code': code, 'name': pos.name, 'action': 'sell',
                                'shares': pos.shares, 'price': np.nan, 'proceeds': proceeds,
                                'fee': fee, 'reason': '年度质量排名更新'})

    def _buy(self, D, picks):
        budget = self.cash / len(picks) if len(picks) else 0.0
        industry_count = {}
        bought = []
        for _, row in picks.iterrows():
            if industry_count.get(row.industry, 0) >= 3:
                continue
            mkt = load_market(row.code, qfq=False)
            if mkt is None or D not in mkt.index:
                continue
            price = float(mkt.loc[D, 'close'])
            shares = int(budget / price / 100) * 100
            amount = shares * price
            fee = buy_fee(amount)
            if shares <= 0 or amount + fee > self.cash:
                continue
            self.cash -= amount + fee
            self.positions[row.code] = Pos(row.code, row.name, shares, price, amount, price, D)
            industry_count[row.industry] = industry_count.get(row.industry, 0) + 1
            bought.append(row.code)
            self.trades.append({'date': D, 'code': row.code, 'name': row.name, 'action': 'buy',
                                'shares': shares, 'price': price, 'proceeds': amount, 'fee': fee,
                                'reason': '巴菲特质量代理 Top%d' % self.n})
        return bought

    def run(self, signal_dates, ranking_map, end: Optional[pd.Timestamp] = None,
            verbose: bool = True):
        for signal_D in signal_dates:
            D = self._next_day(signal_D)
            if D is None:
                continue
            self._sell_all(D)
            bought = self._buy(D, ranking_map.get(signal_D, pd.DataFrame()).head(self.n))
            # 该期持仓的行业市值 (按买入毛投资额统计)
            indw = {}
            for code, pos in self.positions.items():
                ind = _industry_of(code)
                indw[ind] = indw.get(ind, 0.0) + pos.invested
            self.periods.append({'date': D, 'signal_date': signal_D,
                                 'candidates': len(ranking_map.get(signal_D, pd.DataFrame())),
                                 'bought': len(bought),
                                 'industry': {k: v for k, v in indw.items() if v > 0}})
            self._snapshots.append((D, self.cash, list(self.positions.values())))
            if verbose:
                print(f'  {D.date()} 候选{len(ranking_map.get(signal_D, pd.DataFrame()))} '
                      f'买入{len(bought)}只')
        if not self._snapshots:
            return self.nav_curve, self.trades, signal_dates
        last_day = pd.Timestamp(end) if end is not None else pd.Timestamp(END)
        cal = trading_days()
        cal = cal[(cal >= self._snapshots[0][0]) & (cal <= last_day)]
        rows, hrows, si = {}, {}, 0
        for t in cal:
            while si + 1 < len(self._snapshots) and self._snapshots[si + 1][0] <= t:
                si += 1
            _, cash, positions = self._snapshots[si]
            value = cash
            held = 0.0
            for pos in positions:
                market, dividends = GrahamBacktest._position_value(pos, t)
                value += market + dividends
                held += market
            rows[t] = value
            hrows[t] = held
        self.nav_curve = pd.Series(rows)
        self.holdings_curve = pd.Series(hrows)
        return self.nav_curve, self.trades, signal_dates
