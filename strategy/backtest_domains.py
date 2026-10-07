"""Stable registry of the repository's backtest domains."""

from strategy.graham_backtest import GrahamBacktest
from strategy.magic_backtest import MagicBacktest
from strategy.trend_execution import (
    ATRChannelStrategy, DoubleBottomStrategy, TurtleStrategy,
)
from strategy.band_backtest import BandBacktest
from strategy.mean_reversion import MeanReversionBacktest

BACKTEST_DOMAINS = {
    'graham': GrahamBacktest,
    'magic': MagicBacktest,
    'atr_channel': ATRChannelStrategy,
    'double_bottom': DoubleBottomStrategy,
    'turtle': TurtleStrategy,
    'band': BandBacktest,
    'mean_reversion': MeanReversionBacktest,
}

__all__ = ['BACKTEST_DOMAINS']
