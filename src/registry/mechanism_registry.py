from src.mechanisms.no_mechanism import NoMechanism
from src.mechanisms.disarmament import Disarmament
from src.mechanisms.mediation import Mediation
from src.mechanisms.repetition import Repetition
from src.mechanisms.contracting import Contracting
# Temporarily commented out - collaborator working on reputation mechanisms
# from src.mechanisms.reputation import (ReputationPrisonersDilemma,
#                                        ReputationPublicGoods)

MECHANISM_REGISTRY = {
    # Temporarily disabled - collaborator working on reputation mechanisms
    # "ReputationPrisonersDilemma": ReputationPrisonersDilemma,
    # "ReputationPublicGoods": ReputationPublicGoods,
    "NoMechanism": NoMechanism,
    "Repetition": Repetition,
    "Disarmament": Disarmament,
    "Mediation": Mediation,
    "Contracting": Contracting,
}
