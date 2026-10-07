"""Benchmark market-data downloads for the local preparation pipeline."""

import os
import time
import pandas as pd
from dataload.market_local import fetch_market_qfq

def stage_benchmark(data_dir: str, etfs: dict, indices: dict) -> None:
    for code, name in etfs.items():
        path = os.path.join(data_dir, f'{code}_market.csv')
        try:
            df = fetch_market_qfq(code, is_etf=True)
            if df is None or len(df) < 200:
                print(f'   [x] ETF {code}: 数据不足')
                continue
            df.to_csv(path)
            print(f'   [√] {code} {name}: {len(df)} 行')
        except Exception as exc:
            print(f'   [x] {code}: {exc}')
        time.sleep(0.5)
    for code, name in indices.items():
        path = os.path.join(data_dir, f'{code}_market.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='date', parse_dates=True)
            if df.index.max() >= pd.Timestamp('2026-08-01'):
                print(f'   [√] {code} {name}: 本地已最新')
                continue
        try:
            df = fetch_market_qfq(code)
            if df is None or len(df) < 200:
                print(f'   [x] 指数 {code}: 数据不足')
                continue
            df.to_csv(path)
            print(f'   [√] {code} {name}: {len(df)} 行')
        except Exception as exc:
            print(f'   [x] {code}: {exc}')
        time.sleep(0.5)
    print('   [√] 基准下载完成')
