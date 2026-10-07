"""Position sizing rules shared by signal-driven strategies."""


def initial_amount(nav: float, target_weight: float) -> float:
    return nav * target_weight


def capped_add_amount(nav: float, current_value: float,
                      initial_weight: float, add_fraction: float,
                      max_weight: float) -> float:
    target = initial_amount(nav, initial_weight) * add_fraction
    return max(0.0, min(target, nav * max_weight - current_value))


def whole_lot_shares(amount: float, price: float, lot: int = 100) -> int:
    return int(amount / price / lot) * lot if price > 0 else 0
