"""腾讯财经 API 数据源（proxy.finance.qq.com）"""
import requests
import json
import pandas as pd
from datetime import datetime


def fetch(code: str, start: datetime, end: datetime) -> pd.DataFrame | None:
    """获取 A 股/指数前复权日线行情（OHLCV）"""
    url = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://gu.qq.com/"}

    # 无法预判是 sh 还是 sz，两种都试一下
    for prefix in ("sh", "sz"):
        ticker = f"{prefix}{code}"
        params = {
            "_var": "kline_dayqfq",
            "param": f"{ticker},day,{start.strftime('%Y-%m-%d')},{end.strftime('%Y-%m-%d')},640,qfq",
            "r": "0.8205512681390605",
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            text = resp.text
            json_str = text[text.find('{'):text.rfind('}') + 1]
            data = json.loads(json_str)
            d = data.get("data", {}).get(ticker, {})
            rows = d.get("qfqday") or d.get("day")
            if not rows:
                continue
            df = pd.DataFrame(
                [r[:6] for r in rows],
                columns=["date", "open", "close", "high", "low", "volume"],
            )
            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception:
            continue
    return None
