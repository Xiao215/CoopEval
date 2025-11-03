from __future__ import annotations

import json
import math
from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .models import MoveRecord, RunData


def _safe_pstdev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _compute_match_results(match_moves: List[MoveRecord]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for move in match_moves:
        result.setdefault("points", {})[move.agent] = move.points
        result.setdefault("actions", {})[move.agent] = move.action
    return result


def compute_agent_metrics(run: RunData, baseline_payoffs: Dict[str, float]) -> List[Dict[str, Any]]:
    by_agent: Dict[str, List[MoveRecord]] = defaultdict(list)
    for record in run.moves:
        by_agent[record.agent].append(record)

    # Precompute match outcomes for win/loss
    match_results = {match_id: _compute_match_results(records) for match_id, records in run.matchups.items()}

    rows: List[Dict[str, Any]] = []
    for agent in run.agents:
        records = by_agent.get(agent, [])
        points = [r.points for r in records]
        action_counts = defaultdict(int)
        coop_count = 0
        delegate_count = 0
        invalid_responses = 0
        total_responses = len(records)
        successive_same = 0
        prev_action = None
        action_transitions = 0

        wins = losses = draws = 0

        for record in records:
            action = record.action
            action_counts[action] += 1
            if action.upper().startswith("C"):
                coop_count += 1
            if action.upper().startswith("D"):
                pass  # handled via totals
            if "\"A2\"" in record.response or "A2" in record.response:
                delegate_count += 1
            if not record.response or not record.response.strip():
                invalid_responses += 1
            if prev_action is not None:
                action_transitions += 1
                if action == prev_action:
                    successive_same += 1
            prev_action = action

            match_id = record.metadata.get("match_id")
            if match_id in match_results:
                points_map = match_results[match_id]["points"]
                for opponent, opp_points in points_map.items():
                    if opponent == agent:
                        continue
                    if record.points > opp_points:
                        wins += 1
                    elif record.points < opp_points:
                        losses += 1
                    else:
                        draws += 1

        total_moves = len(points)
        coop_rate = (coop_count / total_moves) if total_moves else 0.0
        delegate_rate = (delegate_count / total_moves) if total_moves else float("nan")
        invalid_rate = (invalid_responses / total_moves) if total_moves else float("nan")
        defection_rate = 1.0 - coop_rate if total_moves else float("nan")
        repetition_rate = (successive_same / action_transitions) if action_transitions else float("nan")
        action_distribution = {k: (v / total_moves) for k, v in action_counts.items()} if total_moves else {}

        denom = wins + losses + draws
        win_rate = wins / denom if denom else float("nan")
        loss_rate = losses / denom if denom else float("nan")
        draw_rate = draws / denom if denom else float("nan")

        baseline_payoff = baseline_payoffs.get(agent)
        expected_payoff = run.expected_payoffs.get(agent, float("nan"))
        delta_payoff = (
            expected_payoff - baseline_payoff if baseline_payoff is not None and not math.isnan(expected_payoff) else float("nan")
        )

        row = {
            "run_dir": str(run.run_dir),
            "game": run.game,
            "mechanism": run.mechanism,
            "agent": agent,
            "expected_payoff": expected_payoff,
            "delta_expected_payoff": delta_payoff,
            "payoff_mean_move": float(mean(points)) if points else float("nan"),
            "payoff_std": _safe_pstdev(points),
            "payoff_min": float(min(points)) if points else float("nan"),
            "payoff_max": float(max(points)) if points else float("nan"),
            "payoff_median": float(median(points)) if points else float("nan"),
            "num_moves": total_moves,
            "coop_rate": coop_rate,
            "defection_rate": defection_rate,
            "delegate_rate": delegate_rate,
            "invalid_response_rate": invalid_rate,
            "action_repetition_rate": repetition_rate,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "draw_rate": draw_rate,
            "action_distribution": json.dumps(action_distribution),
        }
        rows.append(row)
    return rows


def compute_pairwise_metrics(run: RunData) -> List[Dict[str, Any]]:
    """Summarise ordered pair outcomes for a run."""

    rows: List[Dict[str, Any]] = []
    payoff_records: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    coop_counts: Dict[Tuple[str, str], List[float]] = defaultdict(lambda: [0.0, 0])
    stage_payoffs: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    payoffs_json = (run.mechanism_payload or {}).get("payoffs") or {}
    for profile in payoffs_json.get("profiles", []):
        players = profile.get("players") or []
        discounted = profile.get("discounted_average") or {}
        if len(players) < 2:
            # Single-agent entries (degenerate self-play) – store diagonal if present
            agent = players[0] if players else None
            if agent and agent in discounted:
                payoff_records[(agent, agent)].append(float(discounted[agent]))
            continue

        if len(players) != 2:
            # For games with more than two players we currently skip discounted mapping
            continue

        agent_a, agent_b = players
        if agent_a in discounted:
            payoff_records[(agent_a, agent_b)].append(float(discounted[agent_a]))
        if agent_b in discounted:
            payoff_records[(agent_b, agent_a)].append(float(discounted[agent_b]))

    # Collect cooperation rates and raw payoffs from recorded histories
    for round_matchups in run.rounds:
        for matchup in round_matchups:
            if not matchup:
                continue
            for record_i in matchup:
                coop_i = 1.0 if record_i.action.upper().startswith("C") else 0.0
                for record_j in matchup:
                    key = (record_i.agent, record_j.agent)
                    coop_counts[key][0] += coop_i
                    coop_counts[key][1] += 1
                    stage_payoffs[key].append(record_i.points)

    all_pairs = set(payoff_records) | set(stage_payoffs) | set(coop_counts)

    for agent_i, agent_j in sorted(all_pairs):
        payoff_vals = payoff_records.get((agent_i, agent_j))
        if payoff_vals:
            points_i = sum(payoff_vals) / len(payoff_vals)
        else:
            stage_vals = stage_payoffs.get((agent_i, agent_j))
            points_i = sum(stage_vals) / len(stage_vals) if stage_vals else float("nan")

        coop_num, coop_den = coop_counts.get((agent_i, agent_j), [0.0, 0])
        coop_i = (coop_num / coop_den) if coop_den else float("nan")

        rows.append(
            {
                "run_dir": str(run.run_dir),
                "game": run.game,
                "mechanism": run.mechanism,
                "match_id": f"{agent_i}|{agent_j}",
                "agent_i": agent_i,
                "agent_j": agent_j,
                "points_i": points_i,
                "points_j": float("nan"),
                "coop_i": coop_i,
                "coop_j": float("nan"),
                "seat_i": 0,
                "seat_j": 0,
            }
        )

    return rows


def compute_round_trajectory(run: RunData) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_idx, matchups in enumerate(run.rounds):
        flat_moves = [move for matchup in matchups for move in matchup]
        if not flat_moves:
            continue
        coop_rate = sum(1 for m in flat_moves if m.action.upper().startswith("C")) / len(flat_moves)
        rows.append(
            {
                "run_dir": str(run.run_dir),
                "game": run.game,
                "mechanism": run.mechanism,
                "round": round_idx + 1,
                "avg_pair_coop": coop_rate,
            }
        )
    return rows


def compute_conditional_cooperation(run: RunData) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not run.rounds:
        return rows

    # Works for 2-player games; for N-player we approximate using seat order
    history_by_agent: Dict[str, List[str]] = defaultdict(list)
    for matchups in run.rounds:
        if not matchups:
            continue
        matchup = matchups[0]
        for move in matchup:
            history_by_agent[move.agent].append(move.action)

    for agent, actions in history_by_agent.items():
        if len(actions) < 2:
            continue
        given_c = given_d = 0
        coop_after_c = coop_after_d = 0
        for prev, curr in zip(actions[:-1], actions[1:]):
            if prev.upper().startswith("C"):
                given_c += 1
                if curr.upper().startswith("C"):
                    coop_after_c += 1
            else:
                given_d += 1
                if curr.upper().startswith("C"):
                    coop_after_d += 1
        rows.append(
            {
                "run_dir": str(run.run_dir),
                "game": run.game,
                "mechanism": run.mechanism,
                "agent": agent,
                "p_coop_given_opp_C": (coop_after_c / given_c) if given_c else 0.0,
                "p_coop_given_opp_D": (coop_after_d / given_d) if given_d else 0.0,
                "reciprocity_index": ((coop_after_c / given_c) - (coop_after_d / given_d)) if (given_c and given_d) else 0.0,
            }
        )
    return rows


def compute_repetition_run_metrics(run: RunData) -> Dict[str, Any] | None:
    if run.mechanism != "Repetition" or not run.rounds:
        return None
    trajectory = compute_round_trajectory(run)
    if len(trajectory) < 2:
        slope = 0.0
    else:
        xs = np.arange(1, len(trajectory) + 1)
        ys = np.array([row["avg_pair_coop"] for row in trajectory])
        slope = float(np.polyfit(xs, ys, 1)[0])

    return {
        "run_dir": str(run.run_dir),
        "game": run.game,
        "mechanism": run.mechanism,
        "rounds": len(trajectory),
        "coop_trend_slope": slope,
    }


def compute_disarmament_metrics(run: RunData) -> List[Dict[str, Any]]:
    if run.mechanism != "Disarmament":
        return []
    caps_history = run.mechanism_payload.get("caps_history", {})
    rows: List[Dict[str, Any]] = []
    for agent, sessions in caps_history.items():
        total_reduction = 0.0
        reduction_steps = 0
        total_rounds = 0
        final_caps: List[float] | None = None
        for history in sessions.values():
            if not history:
                continue
            prev = [100.0 for _ in history[0]]
            for caps in history:
                if len(prev) != len(caps):
                    limit = min(len(prev), len(caps))
                    pairs = zip(prev[:limit], caps[:limit])
                else:
                    pairs = zip(prev, caps)
                reduction = sum(max(0.0, p - c) for p, c in pairs)
                if reduction > 0:
                    reduction_steps += 1
                total_reduction += reduction
                prev = caps
            total_rounds += len(history)
            final_caps = history[-1]

        if final_caps is None:
            continue
        final_sum = float(sum(final_caps)) if final_caps else float("nan")
        final_mean = final_sum / len(final_caps) if final_caps else float("nan")
        rows.append(
            {
                "run_dir": str(run.run_dir),
                "game": run.game,
                "mechanism": run.mechanism,
                "agent": agent,
                "rounds": total_rounds,
                "final_cap_sum": final_sum,
                "final_cap_mean": final_mean,
                "total_cap_reduction": total_reduction,
                "reduction_rounds": reduction_steps,
            }
        )
    return rows


def compute_reputation_metrics(run: RunData) -> List[Dict[str, Any]]:
    if not run.mechanism.startswith("Reputation") or not run.rounds:
        return []

    history_by_agent: Dict[str, List[int]] = defaultdict(list)
    action_by_round: Dict[str, List[int]] = defaultdict(list)
    for round_idx, matchups in enumerate(run.rounds):
        if not matchups:
            continue
        matchup = matchups[0]
        for move in matchup:
            coop = 1 if move.action.upper().startswith("C") else 0
            action_by_round[move.agent].append(coop)
            prev = history_by_agent[move.agent][-1] if history_by_agent[move.agent] else 0
            new_total = prev + coop
            history_by_agent[move.agent].append(new_total)

    rows: List[Dict[str, Any]] = []
    for agent, cumulative in history_by_agent.items():
        total_rounds = len(cumulative)
        if not total_rounds:
            continue
        cooperations = cumulative[-1]
        reputations = [cumulative[i] / (i + 1) for i in range(total_rounds)]
        final_rep = reputations[-1]
        avg_rep = float(mean(reputations))
        volatility = float(np.std(reputations)) if len(reputations) > 1 else 0.0

        # Lagged correlation between reputation and next-round cooperation
        if total_rounds > 1:
            x = np.array(reputations[:-1])
            y = np.array(action_by_round[agent][1:])
            if np.std(x) > 0 and np.std(y) > 0:
                corr = float(np.corrcoef(x, y)[0, 1])
            else:
                corr = 0.0
        else:
            corr = 0.0

        rows.append(
            {
                "run_dir": str(run.run_dir),
                "game": run.game,
                "mechanism": run.mechanism,
                "agent": agent,
                "rounds": total_rounds,
                "final_reputation": final_rep,
                "average_reputation": avg_rep,
                "reputation_volatility": volatility,
                "reputation_to_next_coop_corr": corr,
                "total_cooperations": cooperations,
            }
        )
    return rows


def compute_mediation_metrics(run: RunData) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if run.mechanism != "Mediation":
        return [], []

    mediator_rounds = run.mechanism_payload.get("mediator_rounds", [])
    mediator_design = _load_mediator_design(run)

    mediator_rows: Dict[str, Dict[str, Any]] = {}
    agent_rows: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "delegations": 0,
        "moves": 0,
        "delegate_points": 0.0,
        "independent_moves": 0,
        "independent_points": 0.0,
    })

    for entry in mediator_rounds:
        mediator = entry["mediator"]
        moves = entry["moves"]
        total_points = sum(m.points for m in moves)
        cooperative = all(m.action.upper().startswith("C") for m in moves)
        mediator_entry = mediator_rows.setdefault(
            mediator,
            {
                "run_dir": str(run.run_dir),
                "game": run.game,
                "mechanism": run.mechanism,
                "mediator": mediator,
                "evaluations": 0,
                "cooperative_outcomes": 0,
                "total_points": 0.0,
            },
        )
        mediator_entry["evaluations"] += 1
        mediator_entry["total_points"] += total_points / len(moves) if moves else 0.0
        if cooperative:
            mediator_entry["cooperative_outcomes"] += 1

        for move in moves:
            row = agent_rows[move.agent]
            row["moves"] += 1
            if "\"A2\"" in move.response or "A2" in move.response:
                row["delegations"] += 1
                row["delegate_points"] += move.points
            else:
                row["independent_moves"] += 1
                row["independent_points"] += move.points

    mediator_results: List[Dict[str, Any]] = []
    for mediator, info in mediator_rows.items():
        design = mediator_design.get(mediator, {}) if mediator_design else {}
        rec_one = design.get("1")
        rec_two = design.get("2")
        mediator_results.append(
            {
                **info,
                "avg_points": info["total_points"] / info["evaluations"] if info["evaluations"] else float("nan"),
                "success_rate": info["cooperative_outcomes"] / info["evaluations"] if info["evaluations"] else 0.0,
                "recommendation_one": f"A{rec_one}" if rec_one is not None else None,
                "recommendation_two": f"A{rec_two}" if rec_two is not None else None,
                "cooperative_when_all_delegate": (rec_two == 0) if rec_two is not None else False,
            }
        )

    agent_results: List[Dict[str, Any]] = []
    for agent, info in agent_rows.items():
        moves = info["moves"]
        agent_results.append(
            {
                "run_dir": str(run.run_dir),
                "game": run.game,
                "mechanism": run.mechanism,
                "agent": agent,
                "delegation_rate": (info["delegations"] / moves) if moves else 0.0,
                "avg_points_delegate": (info["delegate_points"] / info["delegations"]) if info["delegations"] else float("nan"),
                "avg_points_independent": (info["independent_points"] / info["independent_moves"]) if info["independent_moves"] else float("nan"),
            }
        )

    return mediator_results, agent_results


def _load_mediator_design(run: RunData) -> Dict[str, Any]:
    design_path = run.run.root / "mediator_design.json"
    if not design_path.exists():
        return {}
    try:
        return json.loads(design_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
