#!/usr/bin/env python
"""高息价值池 × 唐奇安 20/10 主回测: 全周期 + 基准跨期对比

用法:
  python scripts/run_donchian_value_backtest.py          # 全量
  python scripts/run_donchian_value_backtest.py --demo   # 短区间快速验证
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from analysis.performance import performance_summary
from data.local import load_market_frames
from screener.dividend_value import build_value_screener
from strategy.donchian_value_strategy import DonchianValueStrategy

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
START = pd.Timestamp('2023-01-01')
BENCH = {'512890': '红利低波ETF', '510880': '红利ETF', '000300': '沪深300'}
INITIAL_CAPITAL = 1_000_000.0


def pool_union_codes(screener, data_dir=DATA_DIR, start=START) -> set:
    """按月(首个交易日)收集池并集, 用于预加载行情。

    交易日历以格力(000651)日线为代理; 池只影响预加载范围,
    策略内部仍以自身市场日历重构月度池。
    """
    cal = pd.read_csv(os.path.join(data_dir, '000651_market.csv'),
                      index_col='date', parse_dates=True).index
    month_start = {}
    for d in cal:
        if d < start:
            continue
        key = (d.year, d.month)
        month_start.setdefault(key, d)

    union = set()
    for rep in sorted(month_start.values()):
        union |= {c for c, d in screener(rep).items() if d['pass']}
    return union


def bench_nav(mkt, code, end):
    df = mkt[code]
    df = df[(df.index >= START) & (df.index <= end)]
    if df.empty:
        return None
    return df['close'] / df['close'].iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--top-n', type=int, default=300)
    parser.add_argument('--output-dir', help='可选：保存净值 CSV 和图表')
    args = parser.parse_args()

    end = pd.Timestamp('2023-06-30') if args.demo else pd.Timestamp('2026-12-31')

    screener = build_value_screener(top_n=args.top_n)
    codes = sorted(pool_union_codes(screener))
    mkt = load_market_frames(codes + list(BENCH), DATA_DIR, min_rows=1)
    mkt = {c: df for c, df in mkt.items() if (df.index.max() >= START)}
    print(f'池并集 {len(codes)}, 行情 {len(mkt)}')

    strat = DonchianValueStrategy(screener, initial_capital=INITIAL_CAPITAL)
    raw = strat.run(mkt, start_date=str(START.date()), end_date=str(end.date()))['nav']
    nav = raw[raw.index <= end]
    print(f'回测 {START.date()} ~ {end.date()}, 期末净值 {nav.iloc[-1]:,.0f}')

    navs = {'高息价值×唐奇安(含成本)': nav / nav.iloc[0]}
    for code, label in BENCH.items():
        try:
            b = bench_nav(mkt, code, end)
            if b is not None:
                navs[label] = b
            else:
                print(f'  [x] 基准 {label}: 窗口内无数据, 跳过')
        except Exception as e:
            print(f'  [x] 基准 {code}: {e}')

    rows = {}
    for label, n in navs.items():
        s = performance_summary(n.pct_change().dropna(), n.dropna())
        s['期末权益(¥)'] = f'{n.dropna().iloc[-1] * INITIAL_CAPITAL:,.0f}'
        s['交易日数'] = int(n.dropna().size)
        rows[label] = s
    summary = pd.DataFrame(rows).T
    pd.set_option('display.width', 220)
    print(summary.to_string())

    tl = strat.portfolio.trade_log
    buys = sum(t['amount'] for t in tl if t['action'] == 'buy')
    ann_turnover = buys / (len(nav) / 252) / INITIAL_CAPITAL if len(nav) else 0
    print(f'\n交易: 买入{sum(1 for t in tl if t["action"]=="buy")} 笔, '
          f'卖出{sum(1 for t in tl if t["action"]=="sell")} 笔, '
          f'年均单边换手 {ann_turnover:.2f} 倍')

    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        for label, n in navs.items():
            n.to_csv(os.path.join(out_dir, f'nav_{label}.csv'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Songti SC']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(13, 7))
        for label, n in navs.items():
            ax.plot(n.dropna(), label=label, linewidth=1.6)
        ax.set_title(f'高息价值×唐奇安 vs 基准 ({START.date()} ~ {end.date()})')
        ax.set_ylabel('NAV (初始=1)'); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'donchian_value_main.png'))
        print(f'saved results to {out_dir}')


if __name__ == '__main__':
    main()