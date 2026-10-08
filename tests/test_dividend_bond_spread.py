from datetime import datetime

import pandas as pd

from backtest.engine import affordable_shares, buy_cost, dividends_by_day, sell_cost
from strategy.dividend_bond_spread import DividendBondSpreadStrategy


def test_trading_cost_includes_min_fee():
    assert buy_cost(10_000) == 10.0
    assert sell_cost(10_000) == 15.0
    assert affordable_shares(10_000, 10.0) == 900


def test_dividends_align_to_next_trading_day():
    index = pd.bdate_range('2020-01-06', periods=3)
    divs = pd.Series([0.5], index=[pd.Timestamp('2020-01-04')])  # 周六 -> 下周一
    assert dividends_by_day(index, divs) == {index[0]: 0.5}


def test_spread_backtest_buys_on_signal_and_reinvests_dividends():
    days = pd.bdate_range('2020-01-01', periods=30)
    close = pd.Series(10.0, index=days)
    price = pd.DataFrame({'open': close, 'close': close, 'high': close, 'low': close})
    div_days = list(days[:4]) + [days[10]]
    divs = pd.Series(0.5, index=div_days)
    r10 = pd.Series(2.0, index=days)  # spread = 2.0/10*100 - 2 = 18% > 3%

    bt = DividendBondSpreadStrategy(start=datetime(2020, 1, 1), end=datetime(2020, 2, 1),
                        capital=100_000, price_df=price, divs=divs, r10=r10)
    nav = bt.run()

    assert (nav['cash'] >= 0).all()
    assert nav['shares'].iloc[-1] > 0
    assert any(t['side'] == 'buy(div)' for t in bt.trades)
    first_buy = next(t for t in bt.trades if t['side'] == 'buy')
    assert first_buy['shares'] == 900  # 1 万首买, 扣最低 5 元佣金后 900 股
