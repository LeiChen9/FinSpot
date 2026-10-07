"""Unified backtest execution domain."""

from backtest.engine import BacktestEngine, Position, Trade
from backtest.atr_channel import ATRChannelStrategy
from backtest.band import BandBacktest
from backtest.double_bottom import DoubleBottomStrategy
from backtest.graham import GrahamBacktest, Snap, snapshot
from backtest.magic_formula import MagicBacktest
from backtest.mean_reversion import MeanReversionBacktest
from backtest.turtle import TurtleStrategy

__all__ = [
    'ATRChannelStrategy', 'BandBacktest', 'DoubleBottomStrategy',
    'GrahamBacktest', 'MagicBacktest', 'MeanReversionBacktest',
    'TurtleStrategy', 'Snap', 'snapshot',
    'BacktestEngine', 'Position', 'Trade',
]
