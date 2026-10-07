#!/usr/bin/env python
"""红利选股回测数据集准备

阶段:
  constituents  拉取 沪深300 ∪ 中证红利 当前成分并集 (缓存到 data/meta/universe.csv)
  market        用 akshare qfq 全历史重拉池内行情 (单一锚定点, 覆盖 2020~今)
  financial     THS 财务摘要, 缺失或旧文件补齐
  dividend      分红送配历史 (stock_dividend_cninfo)
  benchmark     510880 / 512890 / 000922 行情

幂等: 已存在且满足新鲜度条件的文件跳过, 可断点续跑。
"""
import argparse
import os
import time


import pandas as pd
from dataload.market_local import fetch_market_qfq
from dataload.universe import build_universe as _build_universe
from dataload.universe import load_universe as _load_universe
from dataload.fundamentals_fetch import fetch_financial_ths
from dataload.fundamentals_fetch import stage_dividend as _stage_dividend
from dataload.fundamentals_fetch import stage_financial as _stage_financial
from dataload.benchmark import stage_benchmark as _stage_benchmark

from common.paths import DATA_DIR
META_DIR = os.path.join(DATA_DIR, 'meta')
FIN_DIR = os.path.join(DATA_DIR, 'financial')
DIV_DIR = os.path.join(DATA_DIR, 'dividend')
MARKET_START = '2020-01-01'
MARKET_END = '2026-08-14'
UNIVERSE_IDX = ['000300', '000922']
BENCHMARK_ETFS = {'510880': '红利ETF', '512890': '红利低波ETF'}
BENCHMARK_IDX = {'000922': '中证红利指数'}


def ensure_dirs():
    for d in [META_DIR, FIN_DIR, DIV_DIR]:
        os.makedirs(d, exist_ok=True)


def load_universe() -> list:
    return _build_universe(META_DIR, UNIVERSE_IDX)


def get_universe() -> list:
    return _load_universe(META_DIR)


def stage_market(codes: list):
    todo = []
    for code in codes:
        path = os.path.join(DATA_DIR, f'{code}_market.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            if df.index.max() >= pd.Timestamp('2026-08-01'):
                continue
        todo.append(code)
    print(f'   待刷新 {len(todo)} / {len(codes)}')
    for i, code in enumerate(todo):
        path = os.path.join(DATA_DIR, f'{code}_market.csv')
        try:
            df = fetch_market_qfq(code)
            if df is None or len(df) < 200:
                print(f'   [x] {code}: 数据不足')
                continue
            df.to_csv(path)
            print(f'   [{i+1}/{len(todo)}] {code}: {len(df)} 行 {df.index[0].date()} ~ {df.index[-1].date()}')
        except Exception as e:
            print(f'   [x] {code}: {e}')
        time.sleep(1.2)
    print('   [√] 行情刷新完成')


def stage_financial(codes: list):
    return _stage_financial(codes, FIN_DIR)


def stage_dividend(codes: list):
    return _stage_dividend(codes, DIV_DIR)


def stage_benchmark():
    _stage_benchmark(DATA_DIR, BENCHMARK_ETFS, BENCHMARK_IDX)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True,
                        choices=['constituents', 'market', 'financial', 'dividend', 'benchmark', 'all'])
    parser.add_argument('--only', default='', help='只处理指定代码, 逗号分隔 (调试用)')
    args = parser.parse_args()
    ensure_dirs()

    if args.stage in ('constituents', 'all'):
        load_universe()
    if args.stage in ('market', 'financial', 'dividend', 'all'):
        codes = get_universe()
        if args.only:
            codes = [c for c in codes if c in args.only.split(',')]
        if args.stage in ('market', 'all'):
            stage_market(codes)
        if args.stage in ('financial', 'all'):
            stage_financial(codes)
        if args.stage in ('dividend', 'all'):
            stage_dividend(codes)
    if args.stage in ('benchmark', 'all'):
        stage_benchmark()


if __name__ == '__main__':
    main()
