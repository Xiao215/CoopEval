"""Registry mapping mechanism names defined in configs to implementations."""

from coopeval.mechanisms.contracting import Contracting
from coopeval.mechanisms.mediation import Mediation
from coopeval.mechanisms.no_mechanism import NoMechanism
from coopeval.mechanisms.repetition import Repetition
from coopeval.mechanisms.reputation import Reputation

MECHANISM_REGISTRY = {
    "NoMechanism": NoMechanism,
    "Reputation": Reputation,
    "ReputationFirstOrder": Reputation,
    "Repetition": Repetition,
    "Mediation": Mediation,
    "Contracting": Contracting,
}
