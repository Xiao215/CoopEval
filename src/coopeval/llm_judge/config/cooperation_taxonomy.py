"""Default cooperation taxonomy exposed as a Python constant."""

from __future__ import annotations

from typing import Dict

COOPERATION_TAXONOMY: Dict[str, Dict[str, str]] = {
    "categories": {
        "Individual utility maximization": "Response includes considerations of pursuing the highest possible personal payoff, optimizing for self-interest with few regard for the payoffs of other players.",
        "Strategic equilibrium focus": "Response includes considerations of appealing to game-theoretic stability, such as attempting to play a Nash equilibrium strategy. The agent bases its choice on formulating an optimal response to the anticipated, mathematically rational behavior of others.",
        "Social welfare maximization": "Response includes considerations of a utilitarian desire to maximize the combined total payoff or collective utility of all players in the game, even if it requires sacrificing some of the agent's own individual payoff.",
        "Inequity aversion": "Response includes considerations of a desire to minimize the difference in payoffs between players. The agent prioritizes symmetric outcomes, aiming to ensure no player gets significantly more or less than others.",
        "Reciprocity": "Response includes considerations of an intention to respond to the other player's actions in kind, such as rewarding perceived cooperative behavior or punishing uncooperative behavior.",
        "Strategic influence": "Response includes considerations of an attempt to shape the downstream behavior of other players or to maintain better control over the future dynamics of the game.",
        "Trust evaluation": "Response includes considerations of an assessment of whether the other player can be trusted to cooperate or act in a mutually beneficial manner.",
        "Competitiveness": "Response includes considerations of a desire to achieve a higher payoff than the other player, for example, by prioritizing relative performance and beating the other player.",
        "Uncertainty evaluation": "Response includes considerations of the need to navigate, measure, or mitigate uncertainty regarding the other player's underlying intentions or strategy.",
        "Social norm conformity": "Response includes considerations of evaluating other players' expectations or attempting to conform to a perceived norm, collective practice, or cultural appropriateness.",
        "Rule misunderstanding": "Response includes considerations of an expressed misunderstanding, uncertainty, or confusion regarding the underlying rules and mechanics of the game.",
        "Exploration-exploitation trade-off": "Response includes considerations of the need to balance exploiting known, high-performing strategies against experimenting with less-explored ones.",
        "Risk aversion": "Response includes considerations of a desire to minimize exposure to risk and unpredictable outcomes.",
        "Strategy legibility": "Response includes considerations of the intent to adopt a simple, clear strategy that is easily understood or anticipated by the other player.",
        "Multidimensional reasoning": "The agent exhibits complex reasoning that integrates various facets of the decision-making problem. The analysis goes beyond a one-dimensional approach / mathematical treatment.",
        "Others": "Response includes considerations that do not fit into any of above categories, or the reasoning is too vague to be categorized.",
    }
}

__all__ = ["COOPERATION_TAXONOMY"]
