"""Compatibility entry point for importing local market CSV files."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.manual_import import *


if __name__ == "__main__":
    from data.manual_import import main

    main()
