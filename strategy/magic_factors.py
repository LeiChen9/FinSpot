"""Reusable Magic Formula candidate filters and ranking."""

import numpy as np
import pandas as pd


def listing_proxy_years(code, date, load_market, load_balance, load_profit, anchor):
    frame = load_market(code, qfq=False)
    if frame is None or frame.empty:
        earliest = None
        for report in (load_balance(code), load_profit(code)):
            if report is not None and len(report):
                value = pd.to_datetime(report["报告日"], errors="coerce").dropna().min()
                if earliest is None or value < earliest:
                    earliest = value
        return None if earliest is None or pd.isna(earliest) else (date - earliest).days / 365.25
    first = frame.index.min()
    return 1.0e9 if first <= anchor + pd.Timedelta(days=90) else (date - first).days / 365.25


def rank_magic_candidates(frame: pd.DataFrame, mode: str, exclude_mode: str,
                          reverse_mode: str, cyclical: frozenset) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.copy()
    if mode == exclude_mode:
        ranked = ranked[~ranked["cyclical"]].copy()
    ranked["rank_ep"] = ranked["ep"].rank(ascending=False, method="min").astype(int)
    if mode == reverse_mode:
        asc = ranked["roe"].rank(ascending=True, method="min").astype(int)
        desc = ranked["roe"].rank(ascending=False, method="min").astype(int)
        ranked["rank_roe"] = np.where(ranked["cyclical"], asc, desc)
    else:
        ranked["rank_roe"] = ranked["roe"].rank(ascending=False, method="min").astype(int)
    ranked["composite"] = ranked["rank_ep"] + ranked["rank_roe"]
    return ranked.sort_values(["composite", "rank_ep", "code"]).reset_index(drop=True)
