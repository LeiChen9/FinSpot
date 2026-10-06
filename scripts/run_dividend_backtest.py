#!/usr/bin/env python
"""红利选股 × MA120 择时 主回测: 全周期 + 消融 + 基准跨期

用法:
  python scripts/run_dividend_backtest.py          # 全量
  python scripts/run_dividend_backtest.py --demo   # 短区间快速验证
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from screener.graham import GrahamScreener
from data.local import load_market_frames
from strategy.mean_reversion import (
    MeanReversionBacktest,
    rolling_ma120_vol,
)
from strategy.multifactor import perf_summary

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
START = pd.Timestamp('2023-01-01')
END = pd.Timestamp('2026-08-13')
UNIVERSE = ['000300', '000922']
BENCH = {'510880': '红利ETF', '512890': '红利低波ETF', '000922': '中证红利指数'}


def load_market(codes) -> dict:
    return load_market_frames(codes, DATA_DIR, min_rows=301)


def make_screener(mkt, uni_codes, use_lowvol=True, require_div=True):
    g = GrahamScreener(require_dividend=require_div)

    def sc(as_of):
        passed, _ = g.run(as_of, uni_codes)
        passed = [c for c in passed if c in mkt]
        if not use_lowvol:
            return passed
        vols = rolling_ma120_vol({c: mkt[c] for c in passed}, as_of)
        if vols.empty:
            return []
        return vols[vols <= vols.quantile(0.5)].index.tolist()
    return sc


def bench_nav(mkt, code, start=START, end=END):
    df = mkt[code]
    df = df[(df.index >= start) & (df.index <= end)]
    return df['close'] / df['close'].iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--output-dir', help='可选：保存净值 CSV 和图表的目录')
    args = parser.parse_args()
    end = pd.Timestamp('2023-06-30') if args.demo else END

    uni_codes = pd.read_csv(os.path.join(DATA_DIR, 'meta', 'universe.csv'),
                            dtype={'code': str})['code'].str.zfill(6).tolist()
    mkt = load_market(uni_codes + list(BENCH))
    print(f'universe {len(uni_codes)}, market {len(mkt)}')

    configs = {
        '主策略(全流程)':   dict(use_lowvol=True,  require_div=True,  use_timing=True),
        '消融-无低波过滤':   dict(use_lowvol=False, require_div=True,  use_timing=True),
        '消融-无红利约束':   dict(use_lowvol=True,  require_div=False, use_timing=True),
        '消融-无择时(等权)': dict(use_lowvol=True,  require_div=True,  use_timing=False),
    }

    navs, bts = {}, {}
    for label, cfg in configs.items():
        bt = MeanReversionBacktest(
            mkt, make_screener(mkt, uni_codes, use_lowvol=cfg['use_lowvol'],
                               require_div=cfg['require_div']),
            initial_capital=500_000.0, with_cost=True, use_timing=cfg['use_timing'],
        )
        raw = bt.run(START, end)['nav']
        navs[label] = raw / 500_000.0
        bts[label] = bt
        print(f'  [√] {label}: nav末值={raw.iloc[-1]:,.0f} '
              f'交易={len(bt.trades)} 调仓={len(bt.weights_history)}')

    # 0 成本对照
    bt0 = MeanReversionBacktest(mkt, make_screener(mkt, uni_codes), initial_capital=500_000.0, with_cost=False)
    raw0 = bt0.run(START, end)['nav']
    navs['主策略(0成本)'] = raw0 / 500_000.0
    bts['主策略(0成本)'] = bt0
    print(f'  [√] 主策略(0成本): nav末值={raw0.iloc[-1]:,.0f}')

    for code, label in BENCH.items():
        try:
            navs[label] = bench_nav(mkt, code, start=START, end=end)
        except Exception as e:
            print(f'  [x] 基准 {code}: {e}')

    d = {}
    for label, nav in navs.items():
        s = perf_summary(nav)
        s['期末权益(¥)'] = f'{nav.iloc[-1] * 500000.0:,.0f}'
        d[label] = s
    summary = pd.DataFrame(d).T
    pd.set_option('display.width', 200)
    print(summary.to_string())

    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        for label, nav in navs.items():
            nav.to_csv(os.path.join(out_dir, f'nav_{label}.csv'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Songti SC']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ax = plt.subplots(figsize=(13, 7))
        for label in ['主策略(全流程)', '主策略(0成本)'] + list(BENCH.values()):
            ax.plot(navs[label], label=label, linewidth=1.6)
        ax.set_title(f'红利选股×MA120择时 vs 基准 ({START.date()} ~ {end.date()})')
        ax.set_ylabel('NAV (初始=1)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        chart_path = os.path.join(out_dir, 'dividend_ma120_main.png')
        plt.savefig(chart_path)
        print(f'saved results to {out_dir}')


if __name__ == '__main__':
    main()
