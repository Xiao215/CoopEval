from typing import Callable, Sequence

from src.agents.agent_manager import Agent
from src.games.base import Action, Game


class PublicGoodsAction(Action):
    """Possible actions in the Public Goods game"""

    CONTRIBUTE = "C"
    FREE_RIDE = "F"


class PublicGoods(Game):
    """
    Public Goods environment in which N agents each decide to either
    contribute their full endowment or free-ride. After all choices are made,
    total contributions are multiplied by a factor and split equally.
    """

    def __init__(
        self,
        num_players: int,
        multiplier: float,
    ):

        assert num_players > 1, "Public Goods must have at least 2 players."
        assert 1.0 <= multiplier <= num_players, "Multiplier should be between 1 and num_players."

        self.endowment = 1  # value of endowment does not matter
        self.multiplier = multiplier
        self.num_players = num_players

        # TODO: frame as payoff
        self.prompt_template = """
        You are a player named {{player_name}} in a simultaneous contribution game with {num_players} players.
        Your objective is to maximise your wealth in the game described below.

        Game Rules
        1) You and your opponent each start with {endowment} dollars.
        2) You both decide simultaneously whether to invest all of your starting money or none of it.
        3) The total invested is multiplied by {multiplier} and split equally between you.
        4) Any dollars you do not invest remain yours.

        Actions (choose exactly one per round):
        - {PublicGoodsAction.CONTRIBUTE.to_token()} - Put all your starting money into the joint fund
        - {PublicGoodsAction.FREE_RIDE.to_token()} - Invest nothing and keep all your starting money

        {{instruction}}

        {{additional_info}}
        """

        super().__init__(
            prompt=self.prompt_template.format(
                endowment=self.endowment,
                multiplier=self.multiplier,
                num_players=num_players,
            ),
            num_players=num_players,
            num_actions=len(PublicGoodsAction),
        )

    def play(
        self,
        additional_info: list[str] | str,
        players: Sequence[Agent],
        action_map: Callable = lambda x: x,
    ) -> list[Game.Move]:
        """
        Runs the Public Goods game: collects all actions, computes payoffs,
        and returns a list of Moves with each agent's action and points earned.
        """
        assert (
            len(players) == self.num_players
        ), f"Expected {self.num_players} agents, got {len(players)}."

        if isinstance(additional_info, str):
            additional_info = [additional_info] * self.num_players

        actions = {}
        responses = {}

        for player, info in zip(players, additional_info):
            resp = self.prompt_player(player, info)
            prob_distribution = self._extract_mixed_strategy(player, resp, info)
            action_idx = self._choose_from_mix_strategy(prob_distribution)
            actions[player.name] = action_idx
            responses[player.name] = resp

        actions = action_map(actions)
        actions = {
            name: PublicGoodsAction.from_index(action)
            for name, action in actions.items()
        }

        share = self._calculate_share(actions)

        moves = []
        for name, action in actions.items():
            moves.append(
                Game.Move(
                    name=name,
                    action=action,
                    points=(
                        share
                        if action == PublicGoodsAction.CONTRIBUTE
                        else self.endowment + share
                    ),
                    response=responses[name],
                )
            )
        return moves

    def _calculate_share(self, actions: dict[str, PublicGoodsAction]) -> float:
        """
        Calculate the payoff for each agent based on their contributions.
        """

        contribution_count = sum(
            1 for v in actions.values() if v == PublicGoodsAction.CONTRIBUTE
        )

        return contribution_count * self.endowment * self.multiplier / self.num_players
