"""Sina industry classification acquisition for the Graham universe."""

import json
import os
import time

import pandas as pd
import requests


def _sina_headers() -> dict:
    return {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Referer": "https://finance.sina.com.cn/",
    }


def _sina_board_list(log) -> dict:
    url = "http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"
    for _ in range(5):
        try:
            response = requests.get(url, headers=_sina_headers(), timeout=30)
            start, end = response.text.find("{"), response.text.rfind("}")
            if start >= 0 and end > start:
                payload = json.loads(response.text[start:end + 1])
                if payload:
                    return payload
        except Exception as error:
            log(f"[IND] 板块清单重试: {type(error).__name__}:{str(error)[:60]}")
        time.sleep(4)
    return {}


def _sina_node_codes(node: str, max_pages: int = 30) -> list:
    codes = []
    for page in range(1, max_pages + 1):
        url = (
            "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
            f"Market_Center.getHQNodeData?page={page}&num=100&sort=symbol&asc=1"
            f"&node={node}&symbol=&_s_r_a=init"
        )
        rows = None
        for _ in range(3):
            try:
                response = requests.get(url, headers=_sina_headers(), timeout=30)
                start, end = response.text.find("["), response.text.rfind("]")
                if start >= 0 and end > start:
                    rows = json.loads(response.text[start:end + 1])
                break
            except Exception:
                time.sleep(2)
        if not rows:
            break
        codes.extend(str(row.get("code", "")) for row in rows)
        if len(rows) < 100:
            break
        time.sleep(0.3)
    return [code for code in codes if code]


def fetch_industry(meta_dir: str, log, load_universe) -> pd.DataFrame:
    """Fetch Sina industries, update their cache and Graham universe."""
    boards = _sina_board_list(log)
    print(f"新浪行业板块: {len(boards)} 个")
    industry_by_code = {}
    cache_path = os.path.join(meta_dir, "industry_sina.csv")
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path, dtype={"code": str})
        for _, row in cached.iterrows():
            industry_by_code.setdefault(row["code"], row["industry"])

    for node, payload in boards.items():
        parts = payload.split(",")
        name = parts[1] if len(parts) > 1 else node
        codes = _sina_node_codes(node)
        if not codes:
            log(f"[IND] {name}({node}) 成分抓取失败, 跳过")
            continue
        for code in codes:
            industry_by_code.setdefault(code, name)
        print(f"  {name}: {len(codes)} 只")
        time.sleep(1.0)

    industries = pd.DataFrame(
        sorted(industry_by_code.items()), columns=["code", "industry"]
    )
    industries.to_csv(cache_path, index=False)
    log(f"industry_sina saved: {len(industries)} codes")

    universe = load_universe()
    industry_map = industries.set_index("code")["industry"]
    universe["industry"] = universe["code"].map(industry_map).fillna("")
    universe.to_csv(os.path.join(meta_dir, "graham_universe.csv"), index=False)
    log(f"graham_universe.csv updated with industry column ({len(universe)} rows)")
    return industries
