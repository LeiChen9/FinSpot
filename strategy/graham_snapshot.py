"""Public point-in-time snapshot boundary for Graham-family strategies.

The implementation remains behind the legacy module during migration so old
imports keep working while consumers depend on a focused domain entry point.
"""

from strategy.graham_strategy import Snap, snapshot

__all__ = ['Snap', 'snapshot']
