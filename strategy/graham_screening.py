"""全市场 Graham 候选筛选与门禁漏斗。"""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


class GrahamScreening:
    def __init__(self, load_universe: Callable, snapshot: Callable,
                 evaluate: Callable, r10y: Callable, market_avg_pe_5y: Callable,
                 is_st_name: Callable, size_quantile: float):
        self.load_universe = load_universe
        self.snapshot = snapshot
        self.evaluate = evaluate
        self.r10y = r10y
        self.market_avg_pe_5y = market_avg_pe_5y
        self.is_st_name = is_st_name
        self.size_quantile = size_quantile
        self.funnel: Dict[pd.Timestamp, dict] = {}

    def screen(self, date: pd.Timestamp, min_pass: int = 7,
               size_quantile: float | None = None) -> List[Tuple[object, object]]:
        universe = self.load_universe()
        results, caps = [], []
        scored = 0
        rate = self.r10y(date)
        market_pe = self.market_avg_pe_5y(date)
        for code in universe["code"]:
            snap = self.snapshot(code, date)
            if not snap.tradable or self.is_st_name(snap.name) or not snap.industry:
                continue
            if not np.isfinite(snap.market_cap) or snap.market_cap <= 0:
                continue
            scored += 1
            evaluation = self.evaluate(
                snap, date, r=rate, mpe=market_pe,
                min_pass=min_pass, size_floor=None,
            )
            caps.append(float(snap.market_cap))
            if evaluation.passed:
                results.append((snap, evaluation))

        quantile = self.size_quantile if size_quantile is None else size_quantile
        floor = float(pd.Series(caps).quantile(quantile)) if caps else np.inf
        dividend_ok = earnings_ok = 0
        selected = []
        for snap, evaluation in results:
            evaluation.gates["size"] = (
                snap.market_cap >= floor, f"{snap.market_cap / 1e8:.0f}亿",
                f"≥{floor / 1e8:.0f}亿",
            )
            evaluation.gates_ok = all(
                evaluation.gates[key][0] for key in ("earnings", "dividend", "size")
            )
            dividend_ok += int(evaluation.gates["dividend"][0])
            earnings_ok += int(evaluation.gates["earnings"][0])
            if evaluation.gates_ok:
                selected.append((snap, evaluation))
        self.funnel[date] = {
            "tradable": scored, "score": len(results), "div": dividend_ok,
            "earn": earnings_ok, "final": len(selected),
        }
        return selected
