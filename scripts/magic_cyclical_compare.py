#!/usr/bin/env python
"""行业先验 → 周期行业处置的三档对比回测 (神奇公式)

模式:
  none     基线: 周期行业照常 ROE 降序排名 (即原 magic_formula)
  reverse  反向信号: 周期行业内 ROE 改为升序 (盈利顶值→最差排名; 越低越靠前)
  exclude  直接剔除: 周期行业整体移出初选池

输出: 三档净值对比图 + CSV + 控制台绩效表 + 周期行业在池中占比变化
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _HOME)
sys.path.insert(0, os.path.join(_HOME, 'scripts'))

from graham_dodd_lib import (   # noqa: E402
    load_index, rebalance_dates, perf_metrics, trading_days,
)
from magic_formula_lib import (   # noqa: E402
    MagicBacktest, build_ranking_map, screen_pool, rank_candidates, pick_top,
    N_HOLDINGS,
    CYCLICAL_MODE_NONE, CYCLICAL_MODE_REVERSE, CYCLICAL_MODE_EXCLUDE,
    CYCLICAL_INDUSTRIES, FUNNEL_MF,
)

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False

MODES = {
    CYCLICAL_MODE_NONE:    '基线(周期ROE降序)',
    CYCLICAL_MODE_REVERSE: '反向信号(周期ROE升序)',
    CYCLICAL_MODE_EXCLUDE: '排除周期行业',
}
INITIAL_CAPITAL = 1_000_000.0
OUT = 'reports/generated/magic_formula'


def run_all():
    os.makedirs(OUT, exist_ok=True)
    rbs = rebalance_dates()
    navs, trade_recs = {}, {}
    # 每期只扫一次底池 (screen_pool 内部按 D 缓存), 各模式在其上派生排名 → 无重复重扫
    for D in rbs:
        screen_pool(D)
    rm_all = {}
    for mode in MODES:
        print(f'\n===== 模式: {MODES[mode]} ({mode}) =====')
        rm = {
            d: pick_top(rank_candidates(d, cyclical_mode=mode), n=N_HOLDINGS)
            for d in rbs
        }
        bt = MagicBacktest(initial_capital=INITIAL_CAPITAL, n=N_HOLDINGS)
        nav, trades, _ = bt.run(rbs, ranking_map=rm, verbose=False)
        navs[mode] = nav
        trade_recs[mode] = trades
        m = perf_metrics(nav)
        print(f'  期末净值: {nav.iloc[-1]:,.0f} 元 ({(nav.iloc[-1]/INITIAL_CAPITAL-1):+.2%})')
        print(f'  年化 {m.get("annual_return", np.nan):+.2%} | '
              f'波动 {m.get("annual_vol", np.nan):+.2%} | '
              f'夏普 {m.get("sharpe", np.nan):+.3f} | '
              f'最大回撤 {m.get("max_drawdown", np.nan):+.2%}')
        td = pd.DataFrame(trades)
        buys = td[td.action == 'buy']['proceeds'].sum()
        print(f'  交易 {len(td)} 笔 (买{len(td[td.action=="buy"])}/卖{len(td[td.action=="sell"])}), '
              f'费用 {td["fee"].sum():,.0f} 元 ({td["fee"].sum()/buys:.2%}×买入额)')
        # 周期行业在终选池中的占比
        cyc_share = []
        for d in rbs:
            picks = rm[d]
            if len(picks):
                cyc_share.append((picks['cyclical'].sum() / len(picks)))
        print(f'  周期行业占Top30比例: 均值 {np.mean(cyc_share):.0%}  '
              f'(期中位数 {np.median(cyc_share):.0%})')
    return rbs, navs, trade_recs


def compare(rbs, navs):
    base = navs[CYCLICAL_MODE_NONE] / INITIAL_CAPITAL
    hs = load_index('000300'); hs = hs.loc[hs.index >= base.index[0], 'close']; hs = hs / hs.iloc[0]
    zz = load_index('000906'); zz = zz.loc[zz.index >= base.index[0], 'close']; zz = zz / zz.iloc[0]
    comp = pd.DataFrame({MODES[m]: navs[m] / INITIAL_CAPITAL for m in MODES})
    comp['沪深300'] = hs
    comp['中证800'] = zz
    comp = comp.dropna(how='all').ffill().dropna(subset=[MODES[CYCLICAL_MODE_NONE]])
    comp.round(4).to_csv(os.path.join(OUT, 'nav_compare_cyclical.csv'))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for c in comp.columns:
        ax.plot(comp[c], label=c, linewidth=1.6)
    ax.set_title('神奇公式 · 周期行业处置三档对比 vs 基准 (Top30等权·季度全换·含成本)')
    ax.set_ylabel('净值 (起点=1)'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    img = os.path.join(OUT, 'nav_compare_cyclical.png')
    fig.savefig(img, dpi=110)
    plt.close(fig)

    print('\n===== 三档对比 (期末净值/口径) =====')
    rows = {}
    for m, nav in navs.items():
        mm = perf_metrics(nav)
        rows[MODES[m]] = {
            '期末净值': nav.iloc[-1] / INITIAL_CAPITAL - 1,
            '年化': mm.get('annual_return', np.nan),
            '年化波动': mm.get('annual_vol', np.nan),
            '夏普': mm.get('sharpe', np.nan),
            '最大回撤': mm.get('max_drawdown', np.nan),
        }
    rows['沪深300'] = {'期末净值': comp['沪深300'].iloc[-1] - 1}
    rows['中证800'] = {'期末净值': comp['中证800'].iloc[-1] - 1}
    # 与同窗口基准对齐 (回测最后一期=调仓末日后, 基准同样截到该日, 与原始notebook口径不同)
    print('注: 基准按回测窗口末(调仓末日)截取, 非指数最新交易日 → 数值高于原始notebook口径')
    pd.DataFrame(rows).T.round(4).to_string()
    print(pd.DataFrame(rows).T.round(4).to_string())
    print('\n净值对比图: ', img)


if __name__ == '__main__':
    rbs, navs, trades = run_all()
    compare(rbs, navs)
