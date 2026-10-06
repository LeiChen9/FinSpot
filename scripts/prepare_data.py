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
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
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
    """获取 沪深300 ∪ 中证红利 当前成分并集"""
    import akshare as ak
    codes = []
    for idx in UNIVERSE_IDX:
        for attempt in range(3):
            try:
                df = ak.index_stock_cons(symbol=idx)
                codes.extend(df['品种代码'].astype(str).str.zfill(6).tolist())
                print(f'   [√] {idx} 成分 {len(df)} 只')
                break
            except Exception as e:
                print(f'   [x] {idx} 第{attempt+1}次失败: {e}')
                time.sleep(2)
    codes = sorted(set(codes))
    os.makedirs(META_DIR, exist_ok=True)
    pd.DataFrame({'code': codes}).to_csv(os.path.join(META_DIR, 'universe.csv'), index=False)
    print(f'   并集共 {len(codes)} 只')
    return codes


def get_universe() -> list:
    path = os.path.join(META_DIR, 'universe.csv')
    if os.path.exists(path):
        codes = pd.read_csv(path, dtype={'code': str})['code'].astype(str).str.zfill(6).tolist()
        return codes
    raise RuntimeError('先运行 --stage constituents')


QQ_URL = 'https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get'
QQ_HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://gu.qq.com/'}


def _qq_fetch(ticker: str, end: str) -> list:
    """一次 QQ qfq 请求(按截止日取 800 根), 返回 [date,open,close,high,low,volume] 行列表"""
    import subprocess
    param = f'{ticker},day,2020-01-01,{end},800,qfq'
    out = subprocess.run(
        ['curl', '-s', '--max-time', '20', '-G', QQ_URL,
         '--data-urlencode', '_var=kline_dayqfq',
         '--data-urlencode', f'param={param}',
         '--data-urlencode', 'r=0.1'],
        capture_output=True, text=True,
    ).stdout
    js = json.loads(out[out.find('{'):out.rfind('}') + 1])
    d = js.get('data', {}).get(ticker, {})
    if isinstance(d, list):
        return []
    return d.get('qfqday') or d.get('day') or []


def fetch_market_qfq(code: str, is_etf: bool = False) -> pd.DataFrame | None:
    """QQ 前复权全历史 (按 end 日期 3 段 800 根拼接, 同锚点)"""
    ticker = f'sh{code}' if code.startswith(('6', '5', '9')) else f'sz{code}'
    for attempt in range(3):
        try:
            segs = [_qq_fetch(ticker, e) for e in ('2022-06-30', '2024-06-30', MARKET_END)]
            if not any(segs):
                return None
            merged = {}
            for rows in segs:
                for r in rows:
                    merged[r[0]] = [r[1], r[2], r[3], r[4], r[5]]
            dates = sorted(merged)
            if len(dates) < 200:
                return None
            df = pd.DataFrame(
                [[d] + merged[d] for d in dates],
                columns=['date', 'open', 'high', 'low', 'close', 'volume'],
            )
            df['date'] = pd.to_datetime(df['date'])
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['close'])
            df = df.set_index('date').sort_index()
            return df
        except Exception as e:
            print(f'        (第{attempt+1}次失败: {e})')
            time.sleep(2 * (attempt + 1))
    return None


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


def fetch_financial_ths(code: str) -> pd.DataFrame | None:
    import akshare as ak
    df = ak.stock_financial_abstract(symbol=code)
    if df is None or df.empty:
        return None
    return df


def stage_financial(codes: list):
    for i, code in enumerate(codes):
        path = os.path.join(FIN_DIR, f'{code}_fin.csv')
        need = True
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=1)
            need = not any(c.startswith('2025') for c in df.columns)
        if not need:
            continue
        try:
            df = fetch_financial_ths(code)
            if df is None or df.empty:
                print(f'   [x] {code}: 无摘要')
                continue
            df.to_csv(path, index=False)
            print(f'   [{i+1}/{len(codes)}] {code}: {df.shape}')
        except Exception as e:
            print(f'   [x] {code}: {e}')
        time.sleep(0.3)
    print('   [√] 财务摘要完成')


def stage_dividend(codes: list):
    import akshare as ak
    for i, code in enumerate(codes):
        path = os.path.join(DIV_DIR, f'{code}_dividend.csv')
        if os.path.exists(path):
            continue
        try:
            df = ak.stock_dividend_cninfo(symbol=code)
            if df is None or df.empty:
                print(f'   [x] {code}: 无分红记录')
                continue
            df.to_csv(path, index=False)
            print(f'   [{i+1}/{len(codes)}] {code}: {len(df)} 条')
        except Exception as e:
            print(f'   [x] {code}: {e}')
        time.sleep(0.3)
    print('   [√] 分红数据完成')


def stage_benchmark():
    for code, name in BENCHMARK_ETFS.items():
        path = os.path.join(DATA_DIR, f'{code}_market.csv')
        try:
            df = fetch_market_qfq(code, is_etf=True)
            if df is None or len(df) < 200:
                print(f'   [x] ETF {code}: 数据不足')
                continue
            df.to_csv(path)
            print(f'   [√] {code} {name}: {len(df)} 行 {df.index[0].date()} ~ {df.index[-1].date()}')
        except Exception as e:
            print(f'   [x] ETF {code}: {e}')
        time.sleep(0.5)
    for code, name in BENCHMARK_IDX.items():
        path = os.path.join(DATA_DIR, f'{code}_market.csv')
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
        except Exception as e:
            print(f'   [x] 指数 {code}: {e}')
        time.sleep(0.5)
    print('   [√] 基准下载完成')


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