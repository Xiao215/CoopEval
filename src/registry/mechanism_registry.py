from src.mechanisms.base import NoMechanism
from src.mechanisms.disarmament import Disarmament
from src.mechanisms.mediation import Mediation
from src.mechanisms.repetition import Repetition
from src.mechanisms.reputation import (ReputationPrisonersDilemma,
                                       ReputationPublicGoods)

MECHANISM_REGISTRY = {
    "ReputationPrisonersDilemma": ReputationPrisonersDilemma,
    "NoMechanism": NoMechanism,
    "ReputationPublicGoods": ReputationPublicGoods,
    "Repetition": Repetition,
    "Disarmament": Disarmament,
    "Mediation": Mediation,
}
