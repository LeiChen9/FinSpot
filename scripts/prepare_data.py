"""Compatibility entry point for preparing the local backtest dataset."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.local_pipeline import *
from data.local_pipeline import main


if __name__ == "__main__":
    main()
