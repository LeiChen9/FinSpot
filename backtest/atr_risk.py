"""Shared ATR-based position sizing and pyramiding rules."""


def unit_size(price, atr_value, nav, risk_fraction, stop_loss_multiple,
              max_position_fraction=0.10):
    if atr_value <= 0 or price <= 0:
        return 0
    risk_amount = nav * risk_fraction
    shares = int(risk_amount / (stop_loss_multiple * atr_value) / 100) * 100
    maximum = int(nav * max_position_fraction / price / 100) * 100
    return max(min(shares, maximum), 0)


def should_add(last_entry, current_price, atr_value, threshold, add_count=0):
    if atr_value <= 0:
        return False
    return (current_price - last_entry) / atr_value >= (add_count + 1) * threshold


def trailing_stop(current_price, atr_value, multiple, previous_stop):
    return max(previous_stop, current_price - multiple * atr_value)
