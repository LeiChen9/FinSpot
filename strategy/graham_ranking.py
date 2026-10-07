"""Candidate ranking rules for Graham-derived strategies."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def margin_of_safety_score(snapshot) -> float:
    """Combine tangible-book and normalized-earnings valuation ranks."""
    tangible_book = (
        snapshot.tbvps
        if np.isfinite(snapshot.tbvps) and snapshot.tbvps > 0
        else np.nan
    )
    earnings_multiple = (
        snapshot.price / snapshot.eps_ttm
        if np.isfinite(snapshot.eps_ttm) and snapshot.eps_ttm > 0
        else np.nan
    )
    if np.isfinite(tangible_book) and np.isfinite(earnings_multiple):
        return (snapshot.price / tangible_book) * (earnings_multiple / 20.0)
    if np.isfinite(tangible_book):
        return snapshot.price / tangible_book
    return np.inf if not np.isfinite(earnings_multiple) else earnings_multiple


def profit_trend_components(snapshot) -> Optional[Dict[str, float]]:
    """Return point-in-time earnings trend measures available for a security."""
    if not snapshot.tradable:
        return None

    components = {}
    if np.isfinite(snapshot.ttm_yoy):
        components["ttm_yoy"] = snapshot.ttm_yoy

    annual_profit = snapshot.np_annual[-6:]
    if len(annual_profit) >= 4 and all(value > 0 for value in annual_profit):
        log_profit = np.log(np.asarray(annual_profit, dtype=float))
        years = np.arange(len(log_profit), dtype=float)
        components["slope"] = float(np.polyfit(years, log_profit, 1)[0])

    roe = [value for value in snapshot.roe_annual if np.isfinite(value)]
    if roe:
        components["roe"] = roe[-1]
        if len(roe) >= 3:
            components["roe_trend"] = roe[-1] - roe[-3]

    growth_run = 0
    for index in range(len(snapshot.np_annual) - 1, 0, -1):
        if snapshot.np_annual[index] <= snapshot.np_annual[index - 1]:
            break
        growth_run += 1
    if len(snapshot.np_annual) >= 2:
        components["run"] = float(growth_run)
    return components or None


def rank_profit_trend(candidates: List["Snap"]) -> Dict[int, float]:
    """Rank candidate trend components by descending cross-sectional percentile."""
    keys = ("ttm_yoy", "slope", "roe", "roe_trend", "run")
    components = [profit_trend_components(candidate) for candidate in candidates]
    scores = {}
    for index, measures in enumerate(components):
        if measures is None:
            scores[id(candidates[index])] = float("inf")
            continue
        score = 0.0
        for key in keys:
            if key not in measures:
                continue
            values = [item[key] for item in components if item is not None and key in item]
            rank = pd.Series(values).rank(ascending=False, method="min").iloc[
                values.index(measures[key])
            ]
            score += float(rank)
        scores[id(candidates[index])] = score
    return scores
