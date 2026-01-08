# from typing import Sequence
# from src.ranking_evaluations.population_payoffs import PopulationPayoffs
# from src.mechanisms.base import Mechanism
# from tests.fakes.general_fakes import FakeAction, make_fake_move, FakeAgent

# class FakeMechanism(Mechanism):
#     """Dummy mechanism with hardcoded 3x3 bimatrix payoffs."""

#     def __init__(self, precomputed_payoffs: PopulationPayoffs):
#         """Initialize without a base game."""
#         super().__init__(base_game=FakeGame)
#         self.precomputed_payoffs = precomputed_payoffs

#     def _play_matchup(self, players, payoffs):
#         """Not used in this dummy mechanism."""
#         pass

#     def run_tournament(self, agent_cfgs: Sequence[dict]) -> PopulationPayoffs:
#         """Simply return the pre-configured payoffs."""
#         return self.precomputed_payoffs

#     def set_precomputed_payoffs(self, payoffs: PopulationPayoffs) -> None:
#         """Set the precomputed payoffs to be returned."""
#         self.precomputed_payoffs = payoffs
