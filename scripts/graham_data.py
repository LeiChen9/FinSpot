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
  python scripts/graham_data.py prepare-universe
  python scripts/graham_data.py fetch-all --workers 8 [--limit N]
  python scripts/graham_data.py fetch-micro --workers 8
  python scripts/graham_data.py check --codes 600519,601398
"""
import os
import socket
import sys
import time
import argparse
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.sources import qq, baostock  # noqa: E402  复用本地方子: qq/baostock 行情源

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIN_DIR = os.path.join(DATA_DIR, 'financial')
DIV_DIR = os.path.join(DATA_DIR, 'dividend')
META_DIR = os.path.join(DATA_DIR, 'meta')
LOG_PATH = os.path.join(META_DIR, 'graham_fetch_log.txt')

K_START = '2022-11-01'          # 首个调仓日 2023-01-03 前留足缓冲
_UA = {"User-Agent": "Mozilla/5.0"}


def log(msg):
    os.makedirs(META_DIR, exist_ok=True)
    with open(LOG_PATH, 'a') as f:
        f.write(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n')
    print(msg)


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────

def exchange_of(code: str) -> str:
    if code[0] in ('6', '9'):
        return 'sh'
    if code[0] in ('4', '8') or code.startswith('92'):
        return 'bj'
    return 'sz'


def symbol_sina(code: str) -> str:
    return f'{exchange_of(code)}{code}'


def parse_ymd(s) -> pd.Timestamp:
    try:
        return pd.to_datetime(str(int(s)), format='%Y%m%d', errors='coerce')
    except Exception:
        return pd.NaT


def slice_sina_report(df: pd.DataFrame, keep_cols: list) -> pd.DataFrame:
    """sina 报表 → (报告日, 公告日期, keep_cols) 长表, 数值转 float; 缺失科目=NaN"""
    out = pd.DataFrame({'报告日': df['报告日'], '公告日期': df['公告日期']})
    for c in keep_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            out[c] = np.nan  # 银行等特殊行业缺科目 → NaN, 由筛选层判为不满足
    out['报告日'] = out['报告日'].map(parse_ymd)
    out['公告日期'] = out['公告日期'].map(parse_ymd)
    out = out.dropna(subset=['报告日']).sort_values('报告日')
    return out


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


# ─────────────────────────────────────────────────────────────
# 申万一级行业 (legulegu 指数成分, 全部A股一次抓取)
# ─────────────────────────────────────────────────────────────

# 申万一级 31 个行业 (2021 版, 公开标准; 行业名称与 legulegu 成分页 申万1级 列一致)
SW1_BOARDS = [
    ('801010.SI', '农林牧渔'), ('801030.SI', '基础化工'), ('801040.SI', '钢铁'),
    ('801050.SI', '有色金属'), ('801080.SI', '电子'), ('801110.SI', '家用电器'),
    ('801120.SI', '食品饮料'), ('801130.SI', '纺织服饰'), ('801140.SI', '轻工制造'),
    ('801150.SI', '医药生物'), ('801160.SI', '公用事业'), ('801170.SI', '交通运输'),
    ('801180.SI', '房地产'), ('801200.SI', '商贸零售'), ('801210.SI', '社会服务'),
    ('801230.SI', '综合'), ('801710.SI', '建筑材料'), ('801720.SI', '建筑装饰'),
    ('801730.SI', '电力设备'), ('801740.SI', '国防军工'), ('801750.SI', '计算机'),
    ('801760.SI', '传媒'), ('801770.SI', '通信'), ('801780.SI', '银行'),
    ('801790.SI', '非银金融'), ('801880.SI', '汽车'), ('801890.SI', '机械设备'),
    ('801950.SI', '煤炭'), ('801960.SI', '石油石化'), ('801970.SI', '环保'),
    ('801980.SI', '美容护理'),
]

LEGU_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _fetch_sw_first_level() -> pd.DataFrame:
    """申万一级行业指数清单 (行业代码/名称, 公开标准表, 无需联网)"""
    return pd.DataFrame(SW1_BOARDS, columns=['行业代码', '行业名称']).dropna()


def _fetch_sw_cons(symbol: str) -> pd.DataFrame:
    """抓取某申万一级行业成分 (code→申万1级列), 返回 (股票代码, 申万1级)。
    带指数退避重试, 应对 legulegu 限流。"""
    import requests as _rq
    from io import StringIO
    url = f'https://legulegu.com/stockdata/index-composition?industryCode={symbol}'
    for backoff in (3, 8, 20, 50):
        try:
            r = _rq.get(url, headers=LEGU_HEADERS, timeout=40)
            if r.status_code != 200:
                log(f'[IND] {symbol} http {r.status_code}, backoff {backoff}s')
                time.sleep(backoff)
                continue
            df = pd.read_html(StringIO(r.text))[0]
            cols = [str(c) for c in df.columns]
            ci = cols.index('股票代码') if '股票代码' in cols else 1
            l1 = cols.index('申万1级') if '申万1级' in cols else -1
            if l1 < 0:
                log(f'[IND] {symbol} 页面无 申万1级 列, backoff {backoff}s')
                time.sleep(backoff)
                continue
            out = df.iloc[:, [ci, l1]].copy()
            out.columns = ['code', 'sw1']
            out = out[out['code'].astype(str).str.match(r'^\d{6}')]
            out['code'] = out['code'].astype(str).str[:6]
            out['sw1'] = out['sw1'].astype(str).str.strip()
            out = out.dropna(subset=['sw1'])
            if len(out):
                return out
        except Exception as e:
            log(f'[IND] {symbol} EXC {type(e).__name__}:{str(e)[:60]}, backoff {backoff}s')
            time.sleep(backoff)
    return pd.DataFrame(columns=['code', 'sw1'])


def _sina_board_list() -> dict:
    """新浪行业板块清单: node 码 → (node, 中文名, 成员数,...)"""
    import requests as _rq
    import json as _json
    u = 'http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php'
    for _ in range(5):
        try:
            r = _rq.get(u, headers=_sina_headers(), timeout=30)
            raw = r.text
            i, j = raw.find('{'), raw.rfind('}')
            if i < 0 or j <= i:
                time.sleep(4)
                continue
            payload = _json.loads(raw[i:j + 1])
            if payload:
                return payload
        except Exception as e:
            log(f'[IND] 板块清单重试: {type(e).__name__}:{str(e)[:60]}')
            time.sleep(4)
    return {}


def _sina_headers() -> dict:
    return {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://finance.sina.com.cn/"}


def _sina_node_codes(node: str, max_pages: int = 30) -> list:
    """新浪板块成分 (分页抓取, sina 每页最多 100 条), 返回 code 列表"""
    import requests as _rq
    import json as _json
    out = []
    for page in range(1, max_pages + 1):
        u = (f'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/'
             f'Market_Center.getHQNodeData?page={page}&num=100&sort=symbol&asc=1'
             f'&node={node}&symbol=&_s_r_a=init')
        rows = None
        for _ in range(3):
            try:
                r = _rq.get(u, headers=_sina_headers(), timeout=30)
                txt = r.text
                i = txt.find('[')
                j = txt.rfind(']')
                if i >= 0 and j > i:
                    rows = _json.loads(txt[i:j + 1])
                break
            except Exception:
                time.sleep(2)
        if not rows:
            break
        out.extend(str(x.get('code', '')) for x in rows)
        if len(rows) < 100:
            break
        time.sleep(0.3)
    return [c for c in out if c]


def fetch_industry() -> pd.DataFrame:
    """行业分类(新浪行业板块, 覆盖几乎全部上市A股) → data/meta/industry_sina.csv,
    并写回 graham_universe.csv 的 industry 列。"""
    import requests as _rq
    import json as _json
    boards = _sina_board_list()
    print(f'新浪行业板块: {len(boards)} 个')
    ind_map = {}
    base = os.path.join(META_DIR, 'industry_sina.csv')
    if os.path.exists(base):
        for _, r in pd.read_csv(base, dtype={'code': str}).iterrows():
            ind_map.setdefault(r['code'], r['industry'])
    for node, payload in boards.items():
        parts = payload.split(',')
        name = parts[1] if len(parts) > 1 else node
        codes = _sina_node_codes(node)
        if not codes:
            log(f'[IND] {name}({node}) 成分抓取失败, 跳过')
            continue
        for c in codes:
            ind_map.setdefault(c, name)
        print(f'  {name}: {len(codes)} 只')
        time.sleep(1.0)   # 限流: 板块间隔 1s
    ind_df = pd.DataFrame(sorted(ind_map.items()), columns=['code', 'industry'])
    ind_df.to_csv(base, index=False)
    log(f'industry_sina saved: {len(ind_df)} codes')

    uni = load_universe()
    uni['industry'] = uni['code'].map(ind_df.set_index('code')['industry']).fillna('')
    uni.to_csv(os.path.join(META_DIR, 'graham_universe.csv'), index=False)
    log(f'graham_universe.csv updated with industry column ({len(uni)} rows)')
    return ind_df


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


def _market_cache(code, qfq: bool):
    name = f'{code}_qfq.csv' if qfq else f'{code}_market.csv'
    p = os.path.join(DATA_DIR, name)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    try:
        df = pd.read_csv(p, index_col='date', parse_dates=True)
        return df if len(df) else None
    except Exception:
        return None


def _fetch_market_tolerated(code, qfq: bool, expected_min):
    """反复重取直到窗口起点足够早 (expected_min 为 None 时不校验起点)

    数据源顺序: sina日线(稳定) → 东财 → qq(前复权最近640日) → baostock。
    """
    import time as _t
    for _ in range(4):
        df = _fetch_market_sina(code, qfq=qfq)
        if df is None or df.empty:
            df = _fetch_market_ak(code, qfq=qfq)
        if df is None or df.empty:
            df = _fetch_market_local(code, qfq=qfq)
        if df is None or df.empty:
            return None
        if expected_min is None:
            return df
        ok_start = df.index.min() <= expected_min + pd.Timedelta('90D')
        ok_span = (df.index.max() - df.index.min()).days >= 100
        if ok_start and ok_span:
            return df
        _t.sleep(2)
    return df


def _fetch_market_sina(code, qfq: bool):
    """sina 历史日线 (finance.sina.com.cn), 带自动重试"""
    import akshare as ak
    import time as _t
    sym = symbol_sina(code)
    for _ in range(3):
        try:
            df = ak.stock_zh_a_daily(
                symbol=sym,
                start_date=K_START.replace('-', ''),
                end_date=datetime.now().strftime('%Y%m%d'),
                adjust='qfq' if qfq else '',
            )
            if df is None or df.empty:
                return None
            df = df.reset_index()
            renamed = df.rename(columns={
                'date': 'date', 'open': 'open', 'high': 'high',
                'low': 'low', 'close': 'close', 'volume': 'volume'})
            cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c not in renamed.columns:
                    renamed[c] = np.nan
            renamed = renamed[cols]
            renamed['date'] = pd.to_datetime(renamed['date'])
            renamed = renamed.set_index('date').sort_index()
            for c in ['open', 'high', 'low', 'close', 'volume']:
                renamed[c] = pd.to_numeric(renamed[c], errors='coerce')
            return renamed
        except Exception:
            _t.sleep(2)
            continue
    return None


def _fetch_market_ak(code, qfq: bool):
    import akshare as ak
    import time as _t
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period='daily',
                start_date=K_START.replace('-', ''),
                end_date=datetime.now().strftime('%Y%m%d'),
                adjust='qfq' if qfq else '',
            )
            if df is None or df.empty:
                return None
            df = df.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                    '最低': 'low', '收盘': 'close', '成交量': 'volume'})
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            # 窗口校验: 最早日期应接近起始日 (次新股除外)
            expect = pd.Timestamp(K_START)
            if not df.empty and df.index[-1] - df.index[0] < pd.Timedelta('100D'):
                _t.sleep(2)
                continue
            return df
        except Exception:
            _t.sleep(2)
            continue
    return None


def _fetch_market_local(code, qfq: bool):
    """qq.fetch 返回前复权; baostock 返回不复权。作为降级填充另一份。"""
    start = pd.to_datetime(K_START)
    end = pd.Timestamp.now()
    if qfq:
        try:
            df = qq.fetch(code, start, end)
            if df is not None and len(df):
                return df
        except Exception:
            pass
        return None
    try:
        df = baostock.fetch(code, start, end)
        if df is not None and len(df):
            return df
        df = qq.fetch(code, start, end)
        return df
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# 宏观 + 基准
# ─────────────────────────────────────────────────────────────

def fetch_macro():
    import akshare as ak
    # 中债 10Y 国债收益率
    try:
        df = ak.bond_zh_us_rate()
        df = df.rename(columns={'日期': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', '中国国债收益率2年', '中国国债收益率5年',
                 '中国国债收益率10年', '中国国债收益率30年']].set_index('date').sort_index()
        df.to_csv(os.path.join(DATA_DIR, 'zh_10y_treasury.csv'))
        log(f'10y treasury cached: {len(df)} rows')
    except Exception as e:
        log(f'10y treasury ERR {e}')

    # 全部A股 平均/中位 PE
    try:
        df = ak.stock_a_ttm_lyr()
        df = df.rename(columns={'date': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df[['date', 'middlePETTM', 'averagePETTM', 'middlePELYR', 'averagePELYR']].set_index('date').sort_index()
        os.makedirs(META_DIR, exist_ok=True)
        df.to_csv(os.path.join(META_DIR, 'all_a_pe.csv'))
        log(f'all-a PE cached: {len(df)} rows')
    except Exception as e:
        log(f'all-a PE ERR {e}')

    # 中证800 基准 (sina 优先)
    idx = None
    try:
        idx = ak.stock_zh_index_daily(symbol="sh000906")
        if idx is not None:
            idx = idx.reset_index() if 'date' not in idx.columns else idx
    except Exception as e:
        log(f'000906 sina ERR {e}')
    if idx is None or idx.empty:
        try:
            idx = ak.index_zh_a_hist(symbol='000906', period='daily',
                                     start_date='20150101',
                                     end_date=datetime.now().strftime('%Y%m%d'))
        except Exception as e:
            log(f'000906 em ERR {e}')
    if idx is not None and not idx.empty:
        idx = idx.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                  '最低': 'low', '收盘': 'close', '成交量': 'volume'})
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c not in idx.columns:
                idx[c] = np.nan
        idx['date'] = pd.to_datetime(idx['date'])
        idx = idx[cols].set_index('date').sort_index()
        idx.to_csv(os.path.join(DATA_DIR, '000906_market.csv'))
        log(f'000906 cached: {len(idx)} rows')

    # 沪深300 基准 (sina 指数接口, 同源 000906, 覆盖长历史)
    for iw, tag in (('sh000300', '000300'), ('sh000905', '000905')):
        idf = None
        try:
            idf = ak.stock_zh_index_daily(symbol=iw)
        except Exception as e:
            log(f'{tag} sind ERR {e}')
        if idf is not None and not idf.empty:
            idf = idf.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high',
                                      '最低': 'low', '收盘': 'close', '成交量': 'volume'})
            for c in ['open', 'high', 'low', 'close', 'volume']:
                if c not in idf.columns:
                    idf[c] = np.nan
            idf['date'] = pd.to_datetime(idf['date'])
            idf = idf[['date', 'open', 'high', 'low', 'close', 'volume']].set_index('date').sort_index()
            idf.to_csv(os.path.join(DATA_DIR, f'{tag}_market.csv'))
            log(f'{tag} cached: {len(idf)} rows')


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