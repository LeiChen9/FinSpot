"""Abstract strategy and signal contracts."""

from abc import ABC, abstractmethod
import pandas as pd


class Signal(ABC):
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Return 1 for buy, 0 for hold, and -1 for sell."""


class Strategy(ABC):
    def __init__(self, signal: Signal):
        self.signal = signal
        self.name = self.__class__.__name__

    @abstractmethod
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the strategy signal calculation."""
