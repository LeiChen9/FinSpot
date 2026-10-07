"""Point-in-time Graham defensive-condition and gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.graham_market import market_avg_pe_5y, r10y

if TYPE_CHECKING:
    from strategy.graham_strategy import Snap

COND_NAMES = [
    ('c1', '收益价格比 ≥ 2×AAA'),
    ('c2', 'PE ≤ 60%×市场5年均PE'),
    ('c3', '股息率 ≥ 2/3×AAA'),
    ('c4', '价格 ≤ 2/3 有形净资产'),
    ('c5', '价格 ≤ 2/3 净流动资产'),
    ('c6', '负债/净资产 < 1'),
    ('c7', '流动比率 ≥ 2'),
    ('c8', '总负债 < 2×净流动资产'),
    ('c9', '十年EPS CAGR > 7%'),
    ('c10', '十年中EPS下滑年 ≤ 2'),
]

GATE_NAMES = [
    ('earnings', '连续盈利: 近5年报 ≤1次为负 且最近一年为正'),
    ('dividend', '近5财年现金分红 ≥4 年'),
    ('size', '市值 ≥ 当日全A可交易分位下限'),
]


@dataclass
class Eval:
    passed: bool = False
    npass: int = 0
    gates_ok: bool = False
    values: Dict[str, float] = field(default_factory=dict)
    detail: Dict[str, Tuple[bool, str, str]] = field(default_factory=dict)
    gates: Dict[str, Tuple[bool, str, str]] = field(default_factory=dict)


def evaluate(s: Snap, D: pd.Timestamp, r=None, mpe=None, min_pass: int = 7,
             size_floor: Optional[float] = None) -> Eval:
    result = Eval()
    if r is None:
        r = r10y(D)
    if mpe is None:
        mpe = market_avg_pe_5y(D)
    if not s.tradable or not np.isfinite(r) or not np.isfinite(mpe):
        result.detail['c1'] = (False, 'NA', f'不可交易/缺宏观({r=:},{mpe=:})')
        return result

    pe_ttm = s.price / s.eps_ttm if np.isfinite(s.eps_ttm) and s.eps_ttm > 0 else np.nan

    def c1():
        ep = s.eps_ttm / s.price
        need = 2 * r
        return ep >= need, f'{ep:.4f}', f'≥{need:.4f}'

    def c2():
        if not np.isfinite(pe_ttm):
            return False, 'PE<0(无)', f'≤{0.6*mpe:.1f}'
        return pe_ttm <= 0.6 * mpe, f'{pe_ttm:.1f}', f'≤{0.6*mpe:.1f}'

    def c3():
        need = (2 / 3) * r
        y = s.dividend_yield
        return (np.isfinite(y) and y >= need), f'{y:.4f}' if np.isfinite(y) else '无分红', f'≥{need:.4f}'

    def c4():
        if not np.isfinite(s.tbvps):
            return False, 'NA', f'≤{2/3*0:.2f}'
        return s.price <= (2 / 3) * s.tbvps, f'{s.price:.1f} ', f'≤{2/3*s.tbvps:.1f}'

    def c5():
        if not np.isfinite(s.ncavps):
            return False, 'NA', '≤2/3×NCA'
        return s.price <= (2 / 3) * s.ncavps, f'{s.price:.1f}', f'≤{2/3*s.ncavps:.1f}'

    def c6():
        return np.isfinite(s.debt_ratio) and s.debt_ratio < 1.0, \
            f'{s.debt_ratio:.2f}' if np.isfinite(s.debt_ratio) else 'NA', '<1'

    def c7():
        return np.isfinite(s.current_ratio) and s.current_ratio >= 2.0, \
            f'{s.current_ratio:.2f}' if np.isfinite(s.current_ratio) else 'NA', '≥2'

    def c8():
        return np.isfinite(s.cond8) and s.cond8 < 1.0, \
            f'{s.cond8:.2f}' if np.isfinite(s.cond8) else 'NA', '<1'

    def c9():
        if not s.has_10y:
            return False, '不足10年', 'CAGR>7%'
        eps = s.eps_annual[-10:]
        years = s.eps_years[-10:]
        first, last = eps[0], eps[-1]
        if first <= 0 or last <= 0:
            return False, f'{last:e}', 'CAGR>7%'
        cagr = (last / first) ** (1 / (years[-1] - years[0])) - 1
        return cagr > 0.07, f'{cagr:.1%}', '>7%'

    def c10():
        if not s.has_10y:
            return False, '不足10年', '≤2'
        eps = s.eps_annual[-10:]
        declines = sum(eps[i] < eps[i - 1] for i in range(1, len(eps)))
        return declines <= 2, f'{declines}年', '≤2'

    checks = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
    for (condition_id, _label), check in zip(COND_NAMES, checks):
        passed, current, requirement = check()
        result.detail[condition_id] = (bool(passed), str(current), str(requirement))
        result.npass += int(passed)

    defensive_core = ('c1', 'c2', 'c3', 'c6', 'c7', 'c8', 'c9', 'c10')
    result.passed = result.npass >= min_pass and all(
        result.detail[key][0] for key in defensive_core
    )

    eps = s.eps_annual
    if len(eps) >= 5:
        tail = eps[-5:]
        negative = sum(not np.isfinite(value) or value < 0 for value in tail)
        earnings_ok = negative <= 1 and np.isfinite(tail[-1]) and tail[-1] > 0
        result.gates['earnings'] = (
            earnings_ok, f'{tail[-1]:.4f} (近{len(tail)}年, 负{negative}次)', '≥0 且负次数≤1'
        )
    else:
        result.gates['earnings'] = (False, f'仅{len(eps)}年报', '需≥3年报')

    fiscal_end = int(D.year) - 1
    needed_years = [fiscal_end - offset for offset in range(5)]
    paid_years = [year for year in needed_years if year in s.div_years]
    dividend_ok = len(paid_years) >= 4
    result.gates['dividend'] = (
        dividend_ok, f'{len(paid_years)}/5 财年',
        f'≥4 财年 ({needed_years[0]}~{needed_years[-1]})',
    )

    if size_floor is None or not np.isfinite(s.market_cap):
        size_ok = size_floor is None
        requirement = '未评估' if size_floor is None else f'≥{size_floor/1e8:.0f}亿'
        result.gates['size'] = (size_ok, f'{s.market_cap/1e8:.0f}亿', requirement)
    else:
        result.gates['size'] = (
            s.market_cap >= size_floor, f'{s.market_cap/1e8:.0f}亿',
            f'≥{size_floor/1e8:.0f}亿',
        )

    result.gates_ok = all(result.gates[name][0] for name in ('earnings', 'dividend', 'size'))
    return result
