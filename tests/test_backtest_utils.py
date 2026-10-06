from datetime import datetime

from analysis.backtest import (
    fifo_pnl,
    fifo_trade_outcomes,
    fixed_weight_history,
    first_trading_day,
    scheduled_rebalance_dates,
    trade_statistics,
    turnover_summary,
)


def test_rebalance_calendar_uses_first_available_trading_day():
    days = ["2024-01-02", "2024-04-01", "2024-07-01"]
    assert first_trading_day(days, 2024, 1) == datetime(2024, 1, 2)
    assert scheduled_rebalance_dates(days, "2024-01-01", "2024-07-02") == [
        datetime(2024, 1, 2), datetime(2024, 4, 1), datetime(2024, 7, 1)
    ]


def test_fixed_weights_and_turnover_summary():
    history = fixed_weight_history(["2024-01-02"], {"bond": 0.6, "gold": 0.4})
    assert history[0]["weights"] == {"bond": 0.6, "gold": 0.4}
    summary = turnover_summary([
        {"action": "buy", "shares": 10, "price": 5},
        {"action": "sell", "shares": 4, "price": 6},
    ], initial_capital=100, rebalance_count=2)
    assert summary["notional"] == 74
    assert summary["average_rebalance_turnover"] == 0.37


def test_fifo_pnl_includes_realized_and_unrealized_amounts():
    result = fifo_pnl([
        {"code": "A", "action": "buy", "shares": 10, "price": 10},
        {"code": "A", "action": "buy", "shares": 5, "price": 12},
        {"code": "A", "action": "sell", "shares": 12, "price": 11},
    ], {"A": 13}).iloc[0]
    # FIFO: 10 shares realize +10, then 2 shares realize -2.
    assert result["realized_pnl"] == 8
    assert result["unrealized_pnl"] == 3
    assert result["remaining_shares"] == 3


def test_fifo_trade_outcomes_aggregates_add_lots_and_fees():
    outcomes = fifo_trade_outcomes([
        {"code": "A", "action": "buy", "shares": 10, "price": 10},
        {"code": "A", "action": "buy", "shares": 5, "price": 12},
        {"code": "A", "action": "sell", "shares": 12, "price": 11},
    ], commission=0.001, stamp_tax=0.002)
    row = outcomes.iloc[0]
    assert len(outcomes) == 1
    assert row["shares"] == 12
    assert row["buy_cost"] == 10 * 10 * 1.001 + 2 * 12 * 1.001
    assert row["sell_fees"] == 12 * 11 * 0.003
    assert row["pnl"] > 0


def test_trade_statistics_exposes_expectancy_and_pnl_contributions():
    stats = trade_statistics([
        {"code": "A", "action": "buy", "shares": 100, "price": 10},
        {"code": "A", "action": "sell", "shares": 100, "price": 20},
        {"code": "B", "action": "buy", "shares": 100, "price": 10},
        {"code": "B", "action": "sell", "shares": 100, "price": 9},
    ])
    assert stats["win_count"] == 1
    assert stats["loss_count"] == 1
    assert stats["win_rate"] == 0.5
    assert stats["average_win_return"] == 1.0
    assert stats["average_loss_return"] == -0.1
    assert stats["expectancy_return"] == stats["average_return"]
    assert stats["gross_profit"] == 1000
    assert stats["gross_loss"] == 100
