"""Graham portfolio position and A-share trading-cost rules."""

from dataclasses import dataclass
import pandas as pd


@dataclass
class Position:
    code: str
    name: str
    shares: float
    buy_price: float
    invested: float
    anchor_qfq: float
    buy_date: pd.Timestamp


def stamp_tax(day: pd.Timestamp, cutoff: pd.Timestamp,
              high: float, low: float) -> float:
    return low if day >= cutoff else high


def buy_fee(amount: float, commission: float, minimum: float,
            transfer_fee: float) -> float:
    return max(amount * commission, minimum) + amount * transfer_fee


def sell_fee(day: pd.Timestamp, amount: float, commission: float,
             minimum: float, transfer_fee: float, cutoff: pd.Timestamp,
             stamp_high: float, stamp_low: float) -> float:
    tax = stamp_tax(day, cutoff, stamp_high, stamp_low)
    return max(amount * commission, minimum) + amount * transfer_fee + amount * tax
