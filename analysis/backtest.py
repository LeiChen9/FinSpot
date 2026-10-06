"""策略回测共用的日期、持仓历史和交易流水分析工具。"""
from collections import defaultdict, deque
from datetime import datetime
from typing import Iterable, Mapping, Sequence

import pandas as pd


def first_trading_day(trading_days: Iterable[object], year: int, month: int) -> datetime | None:
    """返回指定年月第一天及之后的首个交易日。"""
    target = pd.Timestamp(year=year, month=month, day=1)
    days = sorted(pd.Timestamp(day).normalize() for day in trading_days)
    for day in days:
        if day >= target:
            return day.to_pydatetime()
    return None


def scheduled_rebalance_dates(
    trading_days: Iterable[object],
    start: object,
    end: object,
    months: Sequence[int] = (1, 4, 7, 10),
) -> list[datetime]:
    """按指定月份生成区间内的首交易日调仓日历。"""
    start_ts, end_ts = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = sorted({pd.Timestamp(day).normalize() for day in trading_days})
    dates: list[datetime] = []
    for year in range(start_ts.year, end_ts.year + 1):
        for month in months:
            day = first_trading_day(days, year, month)
            if day is not None and start_ts <= pd.Timestamp(day) <= end_ts:
                dates.append(day)
    return sorted(set(dates))


def fixed_weight_history(
    rebalance_dates: Iterable[object], weights: Mapping[str, float]
) -> list[dict]:
    """构造可供 ``build_nav_from_weights`` 复用的固定权重调仓记录。"""
    return [
        {"date": pd.Timestamp(day).to_pydatetime(), "weights": dict(weights)}
        for day in rebalance_dates
    ]


def holdings_frame(
    weights_history: Iterable[Mapping], names: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """将调仓权重历史转换成以调仓日为索引的展示表。"""
    names = names or {}
    rows = []
    for entry in weights_history:
        row = {"date": pd.Timestamp(entry["date"])}
        row.update({names.get(code, code): weight for code, weight in entry["weights"].items()})
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def turnover_summary(
    trades: Iterable[Mapping], initial_capital: float | None = None,
    rebalance_count: int | None = None,
) -> dict:
    """统计交易笔数、累计成交额和可选的平均调仓换手率。"""
    records = list(trades)
    notional = sum(float(item["shares"]) * float(item["price"]) for item in records)
    result = {
        "trade_count": len(records),
        "buy_count": sum(item.get("action") == "buy" for item in records),
        "sell_count": sum(item.get("action") == "sell" for item in records),
        "notional": notional,
        "average_trade_notional": notional / len(records) if records else 0.0,
    }
    if initial_capital and rebalance_count:
        result["average_rebalance_turnover"] = notional / initial_capital / rebalance_count
    return result


def fifo_pnl(
    trades: Iterable[Mapping], close_prices: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """按标的以 FIFO 配对成交，并计入 ``close_prices`` 给出的未实现盈亏。"""
    lots: dict[str, deque] = defaultdict(deque)
    realized: dict[str, float] = defaultdict(float)
    codes: set[str] = set()

    for trade in trades:
        code = str(trade["code"])
        action, shares, price = trade["action"], float(trade["shares"]), float(trade["price"])
        codes.add(code)
        if action == "buy":
            lots[code].append([shares, price])
            continue
        if action != "sell":
            raise ValueError(f"unsupported trade action: {action}")
        remaining = shares
        while remaining > 1e-12 and lots[code]:
            lot_shares, lot_price = lots[code][0]
            matched = min(lot_shares, remaining)
            realized[code] += (price - lot_price) * matched
            lot_shares -= matched
            remaining -= matched
            if lot_shares <= 1e-12:
                lots[code].popleft()
            else:
                lots[code][0][0] = lot_shares
        if remaining > 1e-12:
            raise ValueError(f"sell exceeds FIFO inventory for {code}")

    close_prices = close_prices or {}
    rows = []
    for code in sorted(codes):
        remaining_shares = sum(lot[0] for lot in lots[code])
        unrealized = 0.0
        if code in close_prices:
            unrealized = sum((float(close_prices[code]) - price) * shares for shares, price in lots[code])
        rows.append({
            "code": code,
            "realized_pnl": realized[code],
            "unrealized_pnl": unrealized,
            "total_pnl": realized[code] + unrealized,
            "remaining_shares": remaining_shares,
        })
    return pd.DataFrame(rows)


def fifo_trade_outcomes(
    trades: Iterable[Mapping],
    commission: float = 0.0,
    stamp_tax: float = 0.0,
) -> pd.DataFrame:
    """按 FIFO 配对每一笔卖出，并返回真实的交易级盈亏。

    一个卖出订单可能对应多个买入批次，这里将它们合并为一个卖出订单，
    这样胜率不会因为加仓批次数而被重复计数。交易记录若包含 ``fees``，
    优先使用记录中的实际费用；否则按传入费率估算。
    """
    columns = [
        "code", "sell_date", "shares", "buy_cost", "sell_gross",
        "sell_fees", "pnl", "return_pct", "reason",
    ]
    lots: dict[str, deque] = defaultdict(deque)
    outcomes = []

    for trade in trades:
        code = str(trade["code"])
        action = trade["action"]
        shares = float(trade["shares"])
        price = float(trade["price"])
        if shares <= 0 or price < 0:
            raise ValueError("交易的 shares 必须为正数且 price 不能为负数")

        if action == "buy":
            fee = float(trade.get("fees", shares * price * commission))
            # 每股买入费用随 lot 保存，部分卖出时按比例分摊。
            lots[code].append([shares, price, fee / shares])
            continue
        if action != "sell":
            raise ValueError(f"unsupported trade action: {action}")

        remaining = shares
        buy_cost = 0.0
        matched_shares = 0.0
        while remaining > 1e-12 and lots[code]:
            lot_shares, lot_price, lot_fee_per_share = lots[code][0]
            matched = min(lot_shares, remaining)
            matched_shares += matched
            buy_cost += matched * (lot_price + lot_fee_per_share)
            lot_shares -= matched
            remaining -= matched
            if lot_shares <= 1e-12:
                lots[code].popleft()
            else:
                lots[code][0][0] = lot_shares

        if remaining > 1e-12:
            raise ValueError(f"sell exceeds FIFO inventory for {code}")

        sell_gross = matched_shares * price
        sell_fees = float(
            trade.get("fees", sell_gross * (commission + stamp_tax))
        )
        pnl = sell_gross - sell_fees - buy_cost
        outcomes.append({
            "code": code,
            "sell_date": trade.get("date"),
            "shares": matched_shares,
            "buy_cost": buy_cost,
            "sell_gross": sell_gross,
            "sell_fees": sell_fees,
            "pnl": pnl,
            "return_pct": pnl / buy_cost if buy_cost > 0 else 0.0,
            "reason": trade.get("reason", ""),
        })

    return pd.DataFrame(outcomes, columns=columns)


def trade_statistics(
    trades: Iterable[Mapping],
    commission: float = 0.0,
    stamp_tax: float = 0.0,
) -> dict:
    """汇总交易胜率、盈亏比、期望收益和按资金加权的实现收益。

    ``average_return`` 是每笔卖出订单收益率的算术平均，也就是交易期望：
    胜率乘以平均盈利，加上败率乘以平均亏损。与此同时保留按金额统计的
    ``gross_profit``/``gross_loss``，避免小额交易和大额交易被等权看待。
    """
    outcomes = fifo_trade_outcomes(trades, commission, stamp_tax)
    if outcomes.empty:
        return {
            "outcomes": outcomes,
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "flat_count": 0,
            "win_rate": 0.0,
            "average_return": 0.0,
            "expectancy_return": 0.0,
            "average_win_return": 0.0,
            "average_loss_return": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "capital_weighted_return": 0.0,
            "total_pnl": 0.0,
        }

    pnl = outcomes["pnl"]
    wins = outcomes.loc[pnl > 0]
    losses_df = outcomes.loc[pnl < 0]
    gross_profit = float(wins["pnl"].sum())
    gross_loss = float(-losses_df["pnl"].sum())
    win_count = len(wins)
    loss_count = len(losses_df)
    flat_count = len(outcomes) - win_count - loss_count
    average_return = float(outcomes["return_pct"].mean())
    return {
        "outcomes": outcomes,
        "trade_count": len(outcomes),
        "win_count": win_count,
        "loss_count": loss_count,
        "flat_count": flat_count,
        "win_rate": float(win_count / len(outcomes)),
        "average_return": average_return,
        "expectancy_return": average_return,
        "average_win_return": float(wins["return_pct"].mean()) if win_count else 0.0,
        "average_loss_return": float(losses_df["return_pct"].mean()) if loss_count else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "capital_weighted_return": float(pnl.sum() / outcomes["buy_cost"].sum()),
        "total_pnl": float(pnl.sum()),
    }
