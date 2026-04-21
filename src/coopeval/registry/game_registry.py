"""Lookup for translating config names into concrete game classes."""

from coopeval.games.matching_pennies import MatchingPennies
from coopeval.games.prisoners_dilemma import PrisonersDilemma
from coopeval.games.public_goods import PublicGoods
from coopeval.games.stag_hunt import StagHunt
from coopeval.games.travellers_dilemma import TravellersDilemma
from coopeval.games.trust_game import TrustGame

GAME_REGISTRY = {
    "PrisonersDilemma": PrisonersDilemma,
    "PublicGoods": PublicGoods,
    "TravellersDilemma": TravellersDilemma,
    "TrustGame": TrustGame,
    "StagHunt": StagHunt,
    "MatchingPennies": MatchingPennies,
}
