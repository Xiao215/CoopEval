from src.games.prisoners_dilemma import PrisonersDilemma
from src.games.public_goods import PublicGoods
from src.games.travellers_dilemma import TravellersDilemma
from src.games.trust_game import TrustGame
from src.games.stag_hunt import StagHunt
from src.games.matching_pennies import MatchingPennies

GAME_REGISTRY = {
    "PrisonersDilemma": PrisonersDilemma,
    "PublicGoods": PublicGoods,
    "TravellersDilemma": TravellersDilemma,
    "TrustGame": TrustGame,
    "StagHunt": StagHunt,
    "MatchingPennies": MatchingPennies,
}
