from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def cooperation_rate_from_histories(
    histories: Iterable[Iterable[dict]], *, cooperate_token: str = "C", contribute_token: str = "C"
) -> Dict[str, float]:
    """
    Compute per-agent cooperation rate from mechanism histories.

    Histories are lists of per-round lists of move dicts with keys: name, action, points, response.
    For PD, cooperation is action == "C"; for PG, contribution is action == "C".
    """
    counts = defaultdict(lambda: [0, 0])  # name -> [coop, total]
    for round_moves in histories:
        for move in round_moves:
            name = move.get("name")
            action = move.get("action")
            if name is None or action is None:
                continue
            counts[name][1] += 1
            if action == cooperate_token or action == contribute_token:
                counts[name][0] += 1

    rates: Dict[str, float] = {}
    for name, (c, t) in counts.items():
        rates[name] = (c / t) if t else 0.0
    return rates


def expected_payoffs_from_payoffs_json(payoffs_json: Dict[str, Any]) -> Dict[str, float]:
    """Extract expected payoff per agent from payoffs.json structure."""
    return {k: float(v) for k, v in payoffs_json.get("expected_payoff", {}).items()}


def mechanism_effectiveness(
    coop_rates_by_agent: Dict[str, float]
) -> float:
    """Aggregate cooperation as a simple average rate across agents in a run."""
    if not coop_rates_by_agent:
        return 0.0
    return sum(coop_rates_by_agent.values()) / len(coop_rates_by_agent)


def rank_agents_by_metric(metric_map: Dict[str, float]) -> List[Tuple[str, float]]:
    """Return agents sorted descending by metric value."""
    return sorted(metric_map.items(), key=lambda kv: (-kv[1], kv[0]))



