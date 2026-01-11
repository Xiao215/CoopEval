"""Backwards-compatible alias for MatchupPayoffs.

This module maintains backwards compatibility by aliasing the original
PopulationPayoffs class name to the renamed MatchupPayoffs class.

For new code, prefer importing MatchupPayoffs directly from matchup_payoffs.
"""

from src.ranking_evaluations.matchup_payoffs import MatchupPayoffs

# Backwards compatibility alias
PopulationPayoffs = MatchupPayoffs

__all__ = ["PopulationPayoffs"]
