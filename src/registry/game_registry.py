from src.games.prisoners_dilemma import PrisonersDilemma
from src.games.prisoners_dilemma_direct import PrisonersDilemmaDirect
from src.games.public_goods import PublicGoods
from src.games.travellers_dilemma import TravellersDilemma
from src.games.trust_game import TrustGame

GAME_REGISTRY = {
    "PrisonersDilemma": PrisonersDilemma,
    "PrisonersDilemmaDirect": PrisonersDilemmaDirect,
    "PublicGoods": PublicGoods,
    "TravellersDilemma": TravellersDilemma,
    "TrustGame": TrustGame,
}
