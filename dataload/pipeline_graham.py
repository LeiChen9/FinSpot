#!/usr/bin/env python
"""Graham & Dodd 防御型 10 条 - A股 全市场资产池构建 (point-in-time, 断点续传)

构建离线数据库, 供 notebooks/graham_dodd_a_share.ipynb 回测使用。

缓存布局 (复用既有仓库惯例):
  data/meta/graham_universe.csv        全A股清单 (当前 + 退市), 含上市/终止日期
  data/financial/{code}_balance.csv    sina 资产负债表 (报告日, 公告日期, 关键科目)
  data/financial/{code}_profit.csv     sina 利润表     (报告日, 公告日期, EPS, 归母净利)
  data/dividend/{code}_dividend.csv    巨潮分红       (实施方案公告日期, ...)
  data/{code}_market.csv               不复权日线     (date,open,high,low,close,volume)
  data/{code}_qfq.csv                  前复权日线     (date,open,high,low,close,volume)
  data/zh_10y_treasury.csv             中债10Y国债收益率 (历史日度)
  data/meta/all_a_pe.csv               全部A股等权/中位数 PE (历史日度)
  data/000906_market.csv               中证800 基准

用法:
  python -m dataload.pipeline_graham prepare-universe
  python -m dataload.pipeline_graham fetch-all --workers 8 [--limit N]
  python -m dataload.pipeline_graham fetch-micro --workers 8
  python -m dataload.pipeline_graham check --codes 600519,601398
"""
import os
import socket

import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd


from dataload.industry import fetch_industry as _fetch_industry
from dataload.macro import fetch_macro as _fetch_macro
from dataload.market_fetch import fetch_market_history as _fetch_market_tolerated
from dataload.graham_utils import exchange_of, parse_ymd, slice_sina_report, symbol_sina

from common.paths import DATA_DIR
FIN_DIR = os.path.join(DATA_DIR, 'financial')
DIV_DIR = os.path.join(DATA_DIR, 'dividend')
META_DIR = os.path.join(DATA_DIR, 'meta')
LOG_PATH = os.path.join(META_DIR, 'graham_fetch_log.txt')

K_START = '2022-11-01'          # 首个调仓日 2023-01-03 前留足缓冲
def log(msg):
    os.makedirs(META_DIR, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n')
    print(msg)


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 清单
# ─────────────────────────────────────────────────────────────

def _get_current_list(ak):
    """当前A股清单, 带重试。首选 stock_info_a_code_name, 失败回退 spot_em。"""
    import time as _t
    last = None
    for _ in range(4):
        try:
            cur = ak.stock_info_a_code_name()
            if cur is not None and not cur.empty:
                return cur[['code', 'name']]
        except Exception as e:
            last = e
            _t.sleep(3)
    for _ in range(3):
        try:
            cur = ak.stock_zh_a_spot_em()
            cur = cur.rename(columns={'代码': 'code', '名称': 'name'})
            return cur[['code', 'name']]
        except Exception as e:
            last = e
            _t.sleep(3)
    raise RuntimeError(f'无法获取当前A股清单: {last}')


def build_universe() -> pd.DataFrame:
    import akshare as ak
    import time as _t
    rows = []

    cur_map = _get_current_list(ak)
    for r in cur_map.to_dict('records'):
        rows.append({'code': str(r['code']).zfill(6), 'name': r['name'],
                     'kind': 'current', 'ipo_date': pd.NaT, 'end_date': pd.NaT})

    for fn, cols in (('stock_info_sh_delist', None), ('stock_info_sz_delist', None)):
        try:
            df = getattr(ak, fn)()
        except Exception as e:
            log(f'[{fn}] ERR {e}')
            continue
        code_col = '公司代码' if '公司代码' in df.columns else '证券代码'
        name_col = '公司简称' if '公司简称' in df.columns else '证券简称'
        end_col = '暂停上市日期' if '暂停上市日期' in df.columns else '终止上市日期'
        ipo_col = '上市日期'
        for _, r in df.iterrows():
            rows.append({
                'code': str(r[code_col]).zfill(6),
                'name': str(r[name_col]),
                'kind': 'delisted',
                'ipo_date': pd.to_datetime(r.get(ipo_col), errors='coerce'),
                'end_date': pd.to_datetime(r.get(end_col), errors='coerce'),
            })

    uni = pd.DataFrame(rows).drop_duplicates(subset='code', keep='first').reset_index(drop=True)
    uni['mkt'] = uni['code'].map(exchange_of)
    os.makedirs(META_DIR, exist_ok=True)
    uni.to_csv(os.path.join(META_DIR, 'graham_universe.csv'), index=False)
    log(f'universe saved: {len(uni)} codes '
        f'(current {int((uni.kind=="current").sum())}, delisted {int((uni.kind=="delisted").sum())})')
    return uni


def load_universe() -> pd.DataFrame:
    p = os.path.join(META_DIR, 'graham_universe.csv')
    if not os.path.exists(p):
        return build_universe()
    return pd.read_csv(p, dtype={'code': str})


def fetch_industry() -> pd.DataFrame:
    return _fetch_industry(META_DIR, log, load_universe)


# ─────────────────────────────────────────────────────────────
# 单标的采集
# ─────────────────────────────────────────────────────────────

DS_KEEP_BS = ['资产总计', '流动资产合计', '流动负债合计', '负债合计',
              '无形资产', '商誉', '归属于母公司股东权益合计',
              '所有者权益(或股东权益)合计', '实收资本(或股本)']
DS_KEEP_PL = ['归属于母公司所有者的净利润', '基本每股收益', '稀释每股收益']


def fetch_one(code: str) -> dict:
    """返回 {code, balance, profit, dividend, market_raw, market_qfq, error}

    完成一个 code 后写入 data/meta/{code}.done 戳 (跳过依赖旧缓存/部分数据的风险),
    断点续传时已成功 code 直接跳过。
    """
    import akshare as ak
    stamp = os.path.join(META_DIR, f'{code}.done')
    if os.path.exists(stamp):
        return {'code': code, 'balance': None, 'profit': None,
                'dividend': None, 'market_raw': None, 'market_qfq': None,
                'error': '', 'stamped': True}

    res = {'code': code, 'balance': None, 'profit': None,
           'dividend': None, 'market_raw': None, 'market_qfq': None, 'error': ''}

    # 1) 资产负债表 + 利润表 (sina, 带公告日期 = 精确 point-in-time)
    for fname, keep, attr in (
            ('资产负债表', DS_KEEP_BS, 'balance'),
            ('利润表', DS_KEEP_PL, 'profit')):
        path = os.path.join(FIN_DIR, f'{code}_{attr}.csv')
        if os.path.exists(path) and os.path.getsize(path) > 0:
            res[attr] = pd.read_csv(path, dtype={'报告日': str, '公告日期': str})
            continue
        try:
            df = ak.stock_financial_report_sina(stock=symbol_sina(code), symbol=fname)
            out = slice_sina_report(df, keep + ['报告日', '公告日期'])
            if out.empty:
                res['error'] += 'empty-' + attr + ';'
                continue
            os.makedirs(FIN_DIR, exist_ok=True)
            out.to_csv(path, index=False, columns=['报告日', '公告日期'] + keep)
            res[attr] = out
        except Exception as e:
            res['error'] += f'{attr}:{type(e).__name__}:{str(e)[:60]} '

    # 2) 分红 (巨潮)
    dpath = os.path.join(DIV_DIR, f'{code}_dividend.csv')
    if os.path.exists(dpath) and os.path.getsize(dpath) > 0:
        res['dividend'] = pd.read_csv(dpath, dtype={'实施方案公告日期': str})
    else:
        try:
            df = ak.stock_dividend_cninfo(symbol=code)
            if df is not None and not df.empty:
                os.makedirs(DIV_DIR, exist_ok=True)
                df.to_csv(dpath, index=False)
                res['dividend'] = df
        except Exception as e:
            res['error'] += f'dividend:{type(e).__name__}:{str(e)[:60]} '

    # 3) 日线: 不复权(用于账面价值类条件) + 前复权(用于净值), 全部现场重取, 保证口径一致
    #    老股票(财报早于2020)理论上必在2022-11前上市 → 窗口起点校验, 防截断
    earliest_report = pd.NaT
    if res['balance'] is not None and len(res['balance']):
        earliest_report = pd.to_datetime(res['balance']['报告日'], errors='coerce').min()
    expected_min = pd.Timestamp(K_START) if (pd.notna(earliest_report)
                                             and earliest_report <= pd.Timestamp('2020-12-31')) else None

    raw = _fetch_market_tolerated(code, qfq=False, expected_min=expected_min)
    if raw is not None:
        raw.to_csv(os.path.join(DATA_DIR, f'{code}_market.csv'))
    res['market_raw'] = raw

    qfq = _fetch_market_tolerated(code, qfq=True, expected_min=expected_min)
    if raw is not None and qfq is not None:
        gap = (qfq.index.min() - raw.index.min()).days
        if gap > 60:
            qfq2 = _fetch_market_tolerated(code, qfq=True, expected_min=expected_min)
            if qfq2 is not None:  # 仍短就保留, 后续期次会被判定为无交易
                qfq = qfq2
    if qfq is not None:
        qfq.to_csv(os.path.join(DATA_DIR, f'{code}_qfq.csv'))
    res['market_qfq'] = qfq

    if not res['error'] and raw is not None and qfq is not None and len(raw):
        if expected_min is not None:
            if raw.index.min() <= expected_min + pd.Timedelta('90D'):
                open(stamp, 'w').write('ok')
            else:
                res['error'] += f'start_anchor {raw.index.min().date()}; '
        else:
            open(stamp, 'w').write('ok')
    else:
        res['error'] += f'market_missing raw={raw is not None} qfq={qfq is not None}; '
    return res


def fetch_macro():
    return _fetch_macro(DATA_DIR, META_DIR, log)


# ─────────────────────────────────────────────────────────────
# 批量编排
# ─────────────────────────────────────────────────────────────

def collect(codes, workers=8, limit=None):
    codes = [c for c in codes if c]  # drop NaN / ''
    if limit:
        codes = codes[:limit]
    # 跳过已完成戳
    todo = [c for c in codes if not os.path.exists(os.path.join(META_DIR, f'{c}.done'))]
    if len(todo) < len(codes):
        print(f'[skip] {len(codes) - len(todo)} already done')
    codes = todo
    done = 0
    bad = 0
    err_keys = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in codes}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                r = fut.result()
                done += 1
                if r.get('stamped'):
                    continue
                if r['error']:
                    bad += 1
                    err_keys.append((code, r['error'][:80]))
                    log(f'[WARN] {code}: {r["error"][:120]}')
            except Exception as e:
                bad += 1
                err_keys.append((code, f'EXC {type(e).__name__}:{str(e)[:80]}'))
                log(f'[FAIL] {code}: {type(e).__name__} {str(e)[:120]}')
    log(f'batch done: {done} processed, {bad} failed/warn. '
        f'remaining={len(codes) - done}')
    return err_keys[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prepare-universe', 'fetch-all',
                                    'fetch-macro', 'fetch-industry', 'check'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--codes', default=None)
    args = ap.parse_args()

    os.makedirs(FIN_DIR, exist_ok=True)
    os.makedirs(DIV_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    if args.cmd == 'prepare-universe':
        u = build_universe()
        print(u.head(5).to_string())
        print(u.kind.value_counts())
    elif args.cmd == 'fetch-macro':
        fetch_macro()
    elif args.cmd == 'fetch-industry':
        socket.setdefaulttimeout(30)
        os.environ['no_proxy'] = ''
        fetch_industry()
    elif args.cmd == 'check':
        codes = args.codes.split(',')
        for c in codes:
            stamped = os.path.exists(os.path.join(META_DIR, f'{c}.done'))
            print(f'== {c} stamped={stamped}')
            for tag, p in (('balance', os.path.join(FIN_DIR, f'{c}_balance.csv')),
                           ('profit', os.path.join(FIN_DIR, f'{c}_profit.csv')),
                           ('dividend', os.path.join(DIV_DIR, f'{c}_dividend.csv'))):
                if os.path.exists(p):
                    df = pd.read_csv(p)
                    print(f'  {tag}: {df.shape} cols={list(df.columns)[:6]}')
                else:
                    print(f'  {tag}: MISSING')
            for tag in ('market_raw', 'market_qfq'):
                p = os.path.join(DATA_DIR,
                                 f'{c}_market.csv' if tag == 'market_raw' else f'{c}_qfq.csv')
                if os.path.exists(p):
                    df = pd.read_csv(p, index_col='date', parse_dates=True)
                    print(f'  {tag}: {len(df)} rows {df.index.min().date()} -> {df.index.max().date()}')
                else:
                    print(f'  {tag}: MISSING')
    elif args.cmd == 'fetch-all':
        uni = load_universe()
        print(f'universe: {len(uni)} codes')
        socket.setdefaulttimeout(25)  # 兜底: 防止个别上游连接永久挂起阻塞全部 worker
        os.environ['no_proxy'] = ''   # 避免本机代理将上游拖入黑洞
        collect(uni['code'].tolist(), workers=args.workers, limit=args.limit)


if __name__ == '__main__':
    main()
