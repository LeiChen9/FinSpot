#!/usr/bin/env python
"""高息成长池 × MA120 区间触发 主回测 (2023-01 ~ 2026-07)

用法:
  python scripts/run_pool_backtest.py
  python scripts/run_pool_backtest.py --demo     # 短区间快速验证
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from screener.pool_screener import POOL, build_pool_screener, load_qfq, load_raw
from strategy.band_backtest import BandBacktest
from strategy.multifactor import perf_summary

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
START = pd.Timestamp('2023-01-01')
END = pd.Timestamp('2026-07-31')


def make_screener(div_min=0.03, pe_max=20.0, dev_buy=-0.12):
    return build_pool_screener(div_min=div_min, pe_max=pe_max, dev_buy=dev_buy)


def bench_nav(mkt, code, start=START, end=END):
    df = mkt[code]
    df = df[(df.index >= start) & (df.index <= end)]
    return df['close'] / df['close'].iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    end = pd.Timestamp('2023-06-30') if args.demo else END

    mkt = {c: load_qfq(c) for c in POOL if load_qfq(c) is not None}
    bench = {'000300': '沪深300', '000922': '中证红利'}
    for c in bench:
        df = load_raw(c) if c not in mkt else mkt[c]
        mkt[c] = df if df is not None else load_qfq(c)
    print(f'pool {len(POOL)}, 有行情 {len([c for c in POOL if load_qfq(c) is not None])}')

    # 主策略
    bt = BandBacktest(mkt, make_screener(), initial_capital=500_000.0,
                      with_cost=True)
    raw = bt.run(START, end)['nav']
    nav = raw / 500_000.0
    print(f'[√] 主策略: nav末值={raw.iloc[-1]:,.0f} 交易={len(bt.trades)}')

    # 0 成本对照
    bt0 = BandBacktest(mkt, make_screener(), initial_capital=500_000.0,
                       with_cost=False)
    raw0 = bt0.run(START, end)['nav']
    navs = {'主策略(含成本)': nav, '主策略(0成本)': raw0 / 500_000.0}

    for code, label in bench.items():
        try:
            navs[label] = bench_nav(mkt, code, start=START, end=end)
        except Exception as e:
            print(f'  [x] 基准 {code}: {e}')

    d = {}
    for label, n in navs.items():
        s = perf_summary(n)
        s['期末权益(¥)'] = f'{n.iloc[-1] * 500000.0:,.0f}'
        d[label] = s
    summary = pd.DataFrame(d).T
    pd.set_option('display.width', 200)
    print(summary.to_string())

    # 输出 CSV
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'reports', 'generated', 'pool_ma120'))
    os.makedirs(out_dir, exist_ok=True)
    for label, n in navs.items():
        n.to_csv(os.path.join(out_dir, f'nav_{label}.csv'))
    print(f'净值已保存: {out_dir}')


if __name__ == '__main__':
    main()
