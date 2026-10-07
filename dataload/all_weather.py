"""A股全天候组合的数据装载: QQ/新浪行情 + 中债30年国债指数代理腿。

全部资产缓存至 data/ 本地文件, 缓存命中时完全离线可复现。
"""
import json
import os
import subprocess
import time
from typing import Dict

import pandas as pd

from common.paths import DATA_DIR

ASSET_MAP = {
    '000510': '中证A500(指数代理)',
    '510880': '上证红利ETF',
    '0030Y': '30年国债(代理)',
    '518880': '黄金ETF',
    '511880': '银华日利',
}
BENCH_MAP = {
    '000300': '沪深300',
    '000012': '中证全债',
}
ALL_CODES = {**ASSET_MAP, **BENCH_MAP, '511090': '30年国债ETF'}

QQ_ASSETS = {
    '510880': ('sh', '510880'),
    '518880': ('sh', '518880'),
    '511880': ('sh', '511880'),
    '511090': ('sh', '511090'),
}
SINA_BENCH = {
    '000300': 'sh000300',
    '000012': 'sh000012',
    '000510': 'sh000510',
}


def _std(df):
    df = df.rename(columns={'day': 'date', '日期': 'date', '开盘': 'open', '最高': 'high',
                            '最低': 'low', '收盘': 'close', '成交量': 'volume'})
    cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in cols if c in df.columns]]
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.sort_index(inplace=True)
    return df


def _http_text(url, params, headers, tries=5, timeout=20):
    """requests 带退避重试；连续失败后回退 curl（规避瞬时限流/TLS 握手被拦）。"""
    import requests
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
        except Exception:
            pass
        time.sleep(2 * (attempt + 1))
    qs = "&".join(f"{k}={v}" for k, v in (params or {}).items())
    out = subprocess.run(
        ["curl", "-s", "--max-time", str(timeout), url + ("?" + qs if qs else ""),
         "-H", "User-Agent: " + headers.get("User-Agent", "Mozilla/5.0"),
         "-H", "Referer: " + headers.get("Referer", "")],
        capture_output=True, text=True, timeout=timeout + 10)
    return out.stdout if out.returncode == 0 else ""


def _qq_rows(body, code, prefix):
    """解析 QQ kline 返回的 data 节点（dict 或 list 两种形态均兼容）。"""
    if isinstance(body, dict):
        node = body.get(prefix + code, body)
        if isinstance(node, dict):
            return node.get("qfqday") or node.get("day") or []
        if isinstance(node, list):
            return node
    elif isinstance(body, list):
        return body
    return []


def qq_page(code, prefix, end, count=800):
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    params = {"_var": "kline_dayqfq",
              "param": f"{prefix}{code},day,2000-01-01,{end},{count},qfq",
              "r": "0.8205512681390605"}
    text = _http_text(url, params, {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"})
    if not text:
        return []
    try:
        data = json.loads(text[text.find('{'):text.rfind('}') + 1])
    except Exception:
        return []
    return _qq_rows(data.get("data") if isinstance(data, dict) else None, code, prefix)


def qq_paged(code, prefix, start, end):
    """QQ 前复权，向后翻页补齐上市日至今；每页边界可能重复日期，需去重；
    本地 CSV 缓存避免重复抓取。"""
    cache = str(DATA_DIR / f'qq_{code}_qfq.csv')
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        if not df.empty and df.index[-1] >= end:
            df = df[~df.index.duplicated(keep='last')].sort_index()
            return df[(df.index >= start) & (df.index <= end)].sort_index()
    rows, cursor, guard = [], str(end), 0
    while cursor and guard < 25:
        page = qq_page(code, prefix, cursor)
        if not page:
            break
        rows = page + rows
        cursor = page[0][0]
        guard += 1
        if len(page) < 800:
            break
    if not rows:
        return pd.DataFrame(columns=['date', 'open', 'close', 'high', 'low', 'volume'])
    df = pd.DataFrame([r[:6] for r in rows],
                      columns=['date', 'open', 'close', 'high', 'low', 'volume'])
    df = _std(df)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(cache)
    return df[(df.index >= start) & (df.index <= end)].sort_index()


def sina_fetch(sym, start, end):
    """新浪日线 (无复权); 本地 CSV 缓存, 命中后离线可复现。"""
    cache = str(DATA_DIR / f'sina_{sym}.csv')
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0, parse_dates=True).sort_index()
        return df[(df.index >= start) & (df.index <= end)]
    df = None
    for datalen in (1023, 4092):
        params = {"symbol": sym, "scale": "240", "ma": "no", "datalen": str(datalen)}
        text = _http_text(url, params, {"User-Agent": "Mozilla/5.0"}, tries=4)
        if not text:
            continue
        arr = json.loads(text)
        if not arr:
            continue
        cand = _std(pd.DataFrame(arr))
        cand = cand[(cand.index >= start) & (cand.index <= end)]
        df = cand
        if cand.index.min() <= start:
            break
    if df is not None and not df.empty:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(cache)
    return df


def chinabond_30y(retries=10, timeout=40):
    """中债-30年期国债总财富指数 CBA21801（511090 的跟踪基准）。
    python requests 会被 TLS 拦截，必须走 curl；服务器间歇性超时需重试。
    时间戳为“北京时间午夜按 UTC 存”，解码需 +8h。优先读本地缓存。"""
    cache = str(DATA_DIR / 'cba21801_30y_index.json')

    def _to_series(dct):
        """原始键为“北京时间午夜按 UTC 存的 epoch ms”，解码为 naive 北京日期标签。"""
        return pd.Series({pd.Timestamp(float(k) / 1000 + 8 * 3600, unit='s'): float(v)
                          for k, v in dct.items()}).sort_index()

    if os.path.exists(cache):
        with open(cache) as f:
            dct = json.load(f)
        if dct:
            return _to_series(dct)
    url = ("https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQuery?"
           "indexid=8a8b2cef77b239980177b485d20a6379&qxlxt=00&ltcslx="
           "&zslxt=CFZS&zslxt1=CFZS&lx=1&locale=zh_CN")
    for _ in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-s", "--max-time", str(timeout), "-X", "POST", url, "--data", "",
                 "-H", "User-Agent: Mozilla/5.0",
                 "-H", "Accept: application/json",
                 "-H", "X-Requested-With: XMLHttpRequest",
                 "-H", "Referer: https://yield.chinabond.com.cn/"],
                capture_output=True, text=True, timeout=timeout + 10)
            if out.returncode != 0 or not out.stdout.strip():
                time.sleep(3)
                continue
            dct = json.loads(out.stdout).get("CFZS_00")
            if not dct:
                time.sleep(3)
                continue
            s = _to_series(dct)
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(cache, 'w') as f:
                json.dump({str(k): float(v) for k, v in dct.items()}, f)
            return s
        except Exception:
            time.sleep(3)
    raise RuntimeError("ChinaBond CBA21801 fetch failed repeatedly")


def load_all_weather_data(start: pd.Timestamp, end: pd.Timestamp) -> Dict[str, pd.DataFrame]:
    """装载全部全天候资产 (含 30 年国债代理腿拼接), 返回 {code: DataFrame}。"""
    all_data: Dict[str, pd.DataFrame] = {}

    for code, (prefix, full) in QQ_ASSETS.items():
        df = qq_paged(full, prefix, start, end)
        if df.empty:
            print(f'  FAILED(qq) {code}')
            continue
        all_data[code] = df
        print(f'OK(qq)   {code}: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}')

    for code, sym in SINA_BENCH.items():
        df = sina_fetch(sym, pd.Timestamp('2013-01-01'), end)
        df = df[(df.index >= start)] if df is not None else None
        if df is None or df.empty:
            print(f'  FAILED(sina) {code}')
            continue
        all_data[code] = df
        print(f'OK(sina) {code}: {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}')

    # 30年国债腿: 2013-07-29~2023-06-12 用 CBA21801, 2023-06-13 起切换 511090(按首日缩放，无缝衔接)
    cba = chinabond_30y()
    exc = all_data['510880'].index
    switch = pd.Timestamp('2023-06-13')
    base = cba.reindex(exc).ffill()
    assert base.notna().all(), 'CBA21801 与交易所日历对齐缺值'
    post = base.index[base.index >= switch]
    first_etf = post[0]
    ratio = float(base[first_etf]) / float(all_data['511090'].loc[first_etf, 'close'])
    leg = base.copy()
    etf_vals = all_data['511090']['close'].reindex(post)
    assert etf_vals.notna().all(), '511090 缺交易日数据'
    leg.loc[post] = (etf_vals.values * ratio)
    all_data['0030Y'] = pd.DataFrame({'close': leg.dropna()})
    print(f'OK(proxy) 0030Y: {len(leg)} rows, {leg.index[0].date()} ~ {leg.index[-1].date()}, scale={ratio:.4f}')
    print(f'\nLoaded {len(all_data)} assets')
    print('Available:', list(all_data.keys()))
    return all_data
