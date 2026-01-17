"""Score normalization utilities for game theory experiments.

This module provides utilities for normalizing game payoffs to a [0, 1] scale
where 0 represents the Nash Equilibrium payoff and 1 represents the cooperative payoff.
"""


class NormalizeScore:
    """Normalizes game scores to [0, 1] scale based on Nash Equilibrium and cooperative payoffs.

    Precomputes normalization parameters at initialization for efficient repeated normalization.

    Args:
        game: Game name (e.g., "PrisonersDilemma", "PublicGoods")
        game_config: Game configuration dict containing payoff structure

    Attributes:
        game: Game name
        ne_payoff: Nash Equilibrium payoff (worst case)
        coop_payoff: Cooperative payoff (best case)

    Example:
        >>> normalizer = NormalizeScore("PrisonersDilemma", config)
        >>> normalized = normalizer.normalize(raw_score)
    """

    def __init__(self, game: str, game_config: dict):
        """Initialize normalizer by precomputing NE and cooperative payoffs.

        Args:
            game: Game name
            game_config: Game configuration dict
        """
        self.game = game

        if game == "PrisonersDilemma":
            payoff_matrix = game_config["kwargs"]["payoff_matrix"]
            self.ne_payoff = payoff_matrix["DD"][0]  # NE: both defect
            self.coop_payoff = payoff_matrix["CC"][1]  # Cooperative: both cooperate

        elif game == "PublicGoods":
            self.coop_payoff = game_config["kwargs"]["multiplier"]
            self.ne_payoff = 1  # NE: no one contributes

        elif game == "TravellersDilemma":
            self.ne_payoff = game_config["kwargs"]["min_claim"]
            spacing = game_config["kwargs"]["claim_spacing"]
            num_actions = game_config["kwargs"]["num_actions"]
            self.coop_payoff = self.ne_payoff + spacing * (num_actions - 1)

        elif game == "TrustGame":
            payoff_matrix = game_config["kwargs"]["payoff_matrix"]
            self.ne_payoff = payoff_matrix["KK"][0]  # NE: both keep
            self.coop_payoff = payoff_matrix["GG"][0]  # Cooperative: both give

        else:
            # For non-social-dilemma games, use identity normalization
            self.ne_payoff = 0.0
            self.coop_payoff = 1.0

    def normalize(self, score: float) -> float:
        """Normalize a score to [0, 1] scale.

        Args:
            score: Raw score from the game

        Returns:
            Normalized score where 0 = NE payoff, 1 = cooperative payoff
        """
        if self.coop_payoff == self.ne_payoff:
            # Avoid division by zero
            return 0.0

        return (score - self.ne_payoff) / (self.coop_payoff - self.ne_payoff)