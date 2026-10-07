from backtest.portfolio import Portfolio


def test_sell_applies_commission_and_stamp_tax():
    portfolio = Portfolio(initial_cash=1_000)
    assert portfolio.buy("A", price=10, shares=100, commission=0.0)

    proceeds = portfolio.sell(
        "A", price=10, commission=0.01, stamp_tax=0.02,
    )

    assert proceeds == 970
    assert portfolio.cash == 970
    assert portfolio.trade_log[-1]["fees"] == 30
