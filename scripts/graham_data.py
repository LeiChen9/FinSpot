"""Compatibility entry point for the Graham data acquisition pipeline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.graham_pipeline import *
from data.graham_pipeline import main


if __name__ == "__main__":
    main()
