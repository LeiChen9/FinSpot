"""Common proportional trading-cost rules used by portfolio backtests."""

COMMISSION = 0.0003
STAMP_TAX = 0.0005
SLIPPAGE = 0.0005
MIN_FEE = 5.0


def buy_cost(amount: float, with_cost: bool = True) -> float:
    if not with_cost:
        return amount
    return amount * (1 + SLIPPAGE) + max(amount * COMMISSION, MIN_FEE)


def sell_proceeds(amount: float, with_cost: bool = True) -> float:
    if not with_cost:
        return amount
    gross = amount * (1 - SLIPPAGE)
    return gross - max(gross * COMMISSION, MIN_FEE) - gross * STAMP_TAX
