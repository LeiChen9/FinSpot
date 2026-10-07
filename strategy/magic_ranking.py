"""Focused public entry point for Magic Formula universe and ranking."""

from strategy.magic_formula import (
    build_ranking_map, pick_top, rank_candidates, screen_pool,
)

__all__ = ['screen_pool', 'rank_candidates', 'pick_top', 'build_ranking_map']
