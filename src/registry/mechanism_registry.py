from src.mechanisms.no_mechanism import NoMechanism
from src.mechanisms.disarmament import Disarmament
from src.mechanisms.mediation import Mediation
from src.mechanisms.repetition import Repetition
from src.mechanisms.reputation import Reputation

MECHANISM_REGISTRY = {
    "NoMechanism": NoMechanism,
    "Reputation": Reputation,
    "Repetition": Repetition,
    "Disarmament": Disarmament,
    "Mediation": Mediation,
}
