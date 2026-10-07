"""QQ forward-adjusted market data acquisition for the dividend pipeline."""

import json
import subprocess
import time

import pandas as pd


QQ_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"
MARKET_END = "2026-08-14"


def _qq_fetch(ticker: str, end: str) -> list:
    param = f"{ticker},day,2020-01-01,{end},800,qfq"
    output = subprocess.run(
        ["curl", "-s", "--max-time", "20", "-G", QQ_URL,
         "--data-urlencode", "_var=kline_dayqfq",
         "--data-urlencode", f"param={param}",
         "--data-urlencode", "r=0.1"],
        capture_output=True, text=True,
    ).stdout
    payload = json.loads(output[output.find("{"):output.rfind("}") + 1])
    data = payload.get("data", {}).get(ticker, {})
    if isinstance(data, list):
        return []
    return data.get("qfqday") or data.get("day") or []


def fetch_market_qfq(code: str, is_etf: bool = False) -> pd.DataFrame | None:
    """Fetch and merge three QQ windows into a normalized daily frame."""
    ticker = f"sh{code}" if code.startswith(("6", "5", "9")) else f"sz{code}"
    for attempt in range(3):
        try:
            segments = [_qq_fetch(ticker, end) for end in ("2022-06-30", "2024-06-30", MARKET_END)]
            if not any(segments):
                return None
            merged = {row[0]: row[1:6] for rows in segments for row in rows}
            if len(merged) < 200:
                return None
            frame = pd.DataFrame(
                [[date] + merged[date] for date in sorted(merged)],
                columns=["date", "open", "high", "low", "close", "volume"],
            )
            frame["date"] = pd.to_datetime(frame["date"])
            for column in ("open", "high", "low", "close", "volume"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            return frame.dropna(subset=["close"]).set_index("date").sort_index()
        except Exception as error:
            print(f"        (第{attempt + 1}次失败: {error})")
            time.sleep(2 * (attempt + 1))
    return None
