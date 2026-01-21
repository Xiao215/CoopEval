from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

from .aggregate import write_csv
from .models import RunData
from .plots import (
    plot_disarm_binding_heatmap,
    plot_disarm_cap_trajectories,
    plot_disarm_cooperation_trajectory,
    plot_disarm_negotiation_hist,
    plot_disarm_payoff_vs_caps,
    plot_disarm_reduction_bars,
)


def _parse_distribution(response: str, num_actions: int) -> Dict[int, float] | None:
    if not response:
        return None
    matches = re.findall(r"\{.*?\}", response, re.DOTALL)
    payload = json.loads(matches[-1])

    probs: Dict[int, float] = {}
    for idx in range(num_actions):
        key = f"A{idx}"
        val = payload.get(key)
        if isinstance(val, (int, float)):
            probs[idx] = float(val)
        else:
            return None
    return probs if len(probs) == num_actions else None


def _action_labels(game: str, num_actions: int) -> Tuple[List[str], List[str]]:
    short = [f"A{idx}" for idx in range(num_actions)]
    long = [f"Action {idx}" for idx in range(num_actions)]
    if game == "PrisonersDilemma" and num_actions >= 2:
        short = ["C", "D"] + short[2:]
        long = ["Cooperate", "Defect"] + long[2:]
    return short[:num_actions], long[:num_actions]


def _safe_div(num: float, den: float) -> float:
    if den in (0, 0.0):
        return float("nan")
    return num / den


def _format_action_key(prefix: str, label: str) -> str:
    clean = label.lower().replace(" ", "_")
    return f"{prefix}_{clean}"


def _clean_opponent_label(opponents: List[str]) -> str:
    return "|".join(opponents)


def generate_disarmament_report(
    *,
    runs: List[RunData],
    out_dir: Path,
    agent_metrics: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Path]:
    summary_rows: List[Dict[str, Any]] = []
    session_rows: List[Dict[str, Any]] = []
    caps_round_rows: List[Dict[str, Any]] = []
    run_rows: List[Dict[str, Any]] = []
    coop_rows: List[Dict[str, Any]] = []

    for run in runs:
        if run.mechanism != "Disarmament":
            continue

        raw_caps = run.mechanism_payload.get("caps_history", {})
        if not raw_caps:
            continue

        caps_history: Dict[str, Dict[str, List[List[float]]]] = {}
        for agent, sessions in raw_caps.items():
            caps_history[agent] = {
                session_id: list(history)
                for session_id, history in sessions.items()
            }

        session_meta: Dict[tuple[str, str], Dict[str, Any]] = {}
        run_total_reduction = 0.0
        run_first_change: int | None = None

        for agent, sessions in caps_history.items():
            for session_id, history in sessions.items():
                if not history:
                    continue
                num_actions = len(history[0])
                short_labels, long_labels = _action_labels(run.game, num_actions)
                opponents = [name for name in session_id.split("|") if name != agent]

                prev = [100.0 for _ in range(num_actions)]
                total_reduction = 0.0
                reduction_by_action = [0.0 for _ in range(num_actions)]
                reduction_rounds = 0
                first_reduction: int | None = None

                for session_round, caps in enumerate(history, start=1):
                    reduction_this_round = False
                    for action_idx, cap in enumerate(caps):
                        reduction = max(0.0, prev[action_idx] - cap)
                        if reduction > 0:
                            reduction_this_round = True
                            reduction_by_action[action_idx] += reduction
                            total_reduction += reduction
                        prev[action_idx] = cap
                        caps_round_rows.append(
                            {
                                "run_dir": str(run.run_dir),
                                "game": run.game,
                                "agent": agent,
                                "session_id": session_id,
                                "session_round": session_round,
                                "round": session_round,
                                "action": short_labels[action_idx] if action_idx < len(short_labels) else f"A{action_idx}",
                                "action_label": long_labels[action_idx] if action_idx < len(long_labels) else f"Action {action_idx}",
                                "cap": caps[action_idx],
                                "opponents": _clean_opponent_label(opponents),
                            }
                        )
                    if reduction_this_round:
                        reduction_rounds += 1
                        if first_reduction is None:
                            first_reduction = session_round

                run_total_reduction += total_reduction
                if first_reduction is not None:
                    run_first_change = (
                        first_reduction
                        if run_first_change is None
                        else min(run_first_change, first_reduction)
                    )

                session_meta[(agent, session_id)] = {
                    "run_dir": str(run.run_dir),
                    "game": run.game,
                    "agent": agent,
                    "session_id": session_id,
                    "opponents": opponents,
                    "num_actions": num_actions,
                    "short_labels": short_labels,
                    "long_labels": long_labels,
                    "history": history,
                    "negotiation_rounds": len(history),
                    "total_reduction": total_reduction,
                    "reduction_by_action": reduction_by_action,
                    "reduction_rounds": reduction_rounds,
                    "first_reduction": first_reduction,
                    "final_caps": history[-1],
                    "binding_counts": [0 for _ in range(num_actions)],
                    "slack_sums": [0.0 for _ in range(num_actions)],
                    "prob_sums": [0.0 for _ in range(num_actions)],
                    "binding_rounds": 0,
                    "coop_total": 0,
                    "coop_pre": 0,
                    "coop_post": 0,
                    "rounds_total": 0,
                    "rounds_pre": 0,
                    "rounds_post": 0,
                }

        if not session_meta:
            continue

        session_step_counter: Dict[tuple[str, str], int] = defaultdict(int)
        round_coop_totals: Dict[tuple[str, str, int], Dict[str, float]] = defaultdict(
            lambda: {"coop": 0.0, "count": 0.0}
        )

        for record in run.moves:
            match_id = record.metadata.get("match_id")
            key = (record.agent, match_id)
            meta = session_meta.get(key)

            if not meta:
                # If match_id missing, fall back to single session for the agent
                possible = [
                    info for (agent_name, _), info in session_meta.items() if agent_name == record.agent
                ]
                if len(possible) == 1:
                    meta = possible[0]
                    key = (meta["agent"], meta["session_id"])
                else:
                    continue

            num_actions = meta["num_actions"]
            session_step_counter[key] += 1
            session_round = session_step_counter[key]

            action = (record.action or "").upper()
            is_coop = 1.0 if action.startswith("C") else 0.0

            meta["rounds_total"] += 1
            meta["coop_total"] += is_coop

            first_reduction = meta.get("first_reduction")
            if first_reduction is not None and session_round >= first_reduction:
                meta["rounds_post"] += 1
                meta["coop_post"] += is_coop
            else:
                meta["rounds_pre"] += 1
                meta["coop_pre"] += is_coop

            round_coop_totals[(meta["agent"], meta["session_id"], session_round)]["coop"] += is_coop
            round_coop_totals[(meta["agent"], meta["session_id"], session_round)]["count"] += 1

            caps = record.metadata.get("new_cap")
            if not caps and session_round <= len(meta["history"]):
                caps = meta["history"][session_round - 1]
            probs = _parse_distribution(record.response, num_actions)
            if not caps or not probs:
                continue

            meta["binding_rounds"] += 1
            for idx in range(num_actions):
                cap = caps[idx]
                assigned = probs.get(idx, 0.0)
                slack = cap - assigned
                meta["slack_sums"][idx] += slack
                meta["prob_sums"][idx] += assigned
                if abs(slack) < 1e-9:
                    meta["binding_counts"][idx] += 1

        agent_sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for (agent, session_id), meta in session_meta.items():
            long_labels = meta["long_labels"]
            num_actions = meta["num_actions"]

            session_row: Dict[str, Any] = {
                "run_dir": meta["run_dir"],
                "game": meta["game"],
                "agent": agent,
                "session_id": session_id,
                "opponents": _clean_opponent_label(meta["opponents"]),
                "negotiation_rounds": meta["negotiation_rounds"],
                "first_reduction_round": meta["first_reduction"] or "",
                "reduction_rounds": meta["reduction_rounds"],
                "total_cap_reduction": meta["total_reduction"],
                "cooperation_rate": _safe_div(meta["coop_total"], meta["rounds_total"]),
                "cooperation_rate_pre": _safe_div(meta["coop_pre"], meta["rounds_pre"]),
                "cooperation_rate_post": _safe_div(meta["coop_post"], meta["rounds_post"]),
                "binding_rounds": meta["binding_rounds"],
            }

            for idx in range(num_actions):
                label = long_labels[idx]
                key_final = _format_action_key("final_cap", label)
                key_reduce = _format_action_key("reduction", label)
                key_bind = _format_action_key("binding_rate", label)
                key_slack = _format_action_key("avg_slack", label)
                key_prob = _format_action_key("avg_prob", label)

                session_row[key_final] = meta["final_caps"][idx]
                session_row[key_reduce] = meta["reduction_by_action"][idx]
                session_row[key_bind] = _safe_div(
                    meta["binding_counts"][idx], meta["binding_rounds"]
                )
                session_row[key_slack] = _safe_div(
                    meta["slack_sums"][idx], meta["binding_rounds"]
                )
                session_row[key_prob] = _safe_div(
                    meta["prob_sums"][idx], meta["binding_rounds"]
                )

            session_rows.append(session_row)
            agent_sessions[agent].append(meta)

        for (agent, session_id, session_round), data in round_coop_totals.items():
            meta = session_meta.get((agent, session_id))
            if not meta:
                continue
            coop_rows.append(
                {
                    "run_dir": meta["run_dir"],
                    "game": meta["game"],
                    "agent": agent,
                    "session_id": session_id,
                    "session_round": session_round,
                    "avg_cooperation": _safe_div(data["coop"], data["count"]),
                    "opponents": _clean_opponent_label(meta["opponents"]),
                }
            )

        for agent, metas in agent_sessions.items():
            if not metas:
                continue
            num_actions = metas[0]["num_actions"]
            long_labels = metas[0]["long_labels"]

            total_reduction = sum(m["total_reduction"] for m in metas)
            total_rounds = sum(m["rounds_total"] for m in metas)
            total_coop = sum(m["coop_total"] for m in metas)
            total_coop_pre = sum(m["coop_pre"] for m in metas)
            total_coop_post = sum(m["coop_post"] for m in metas)
            total_rounds_pre = sum(m["rounds_pre"] for m in metas)
            total_rounds_post = sum(m["rounds_post"] for m in metas)
            total_binding = sum(m["binding_rounds"] for m in metas)

            final_caps_avg = [
                _safe_div(sum(m["final_caps"][idx] for m in metas), len(metas))
                for idx in range(num_actions)
            ]
            reduction_by_action_total = [
                sum(m["reduction_by_action"][idx] for m in metas)
                for idx in range(num_actions)
            ]
            binding_counts_total = [
                sum(m["binding_counts"][idx] for m in metas)
                for idx in range(num_actions)
            ]
            slack_sums_total = [
                sum(m["slack_sums"][idx] for m in metas)
                for idx in range(num_actions)
            ]
            prob_sums_total = [
                sum(m["prob_sums"][idx] for m in metas)
                for idx in range(num_actions)
            ]

            summary_row: Dict[str, Any] = {
                "run_dir": str(run.run_dir),
                "game": run.game,
                "agent": agent,
                "action_labels": "|".join(long_labels),
                "negotiation_rounds": _safe_div(
                    sum(m["negotiation_rounds"] for m in metas), len(metas)
                ),
                "first_reduction_round": min(
                    (m["first_reduction"] for m in metas if m["first_reduction"]),
                    default="",
                ),
                "reduction_rounds": sum(m["reduction_rounds"] for m in metas),
                "total_cap_reduction": total_reduction,
                "cooperation_rate": _safe_div(total_coop, total_rounds),
                "cooperation_rate_pre": _safe_div(total_coop_pre, total_rounds_pre),
                "cooperation_rate_post": _safe_div(total_coop_post, total_rounds_post),
                "expected_payoff": run.expected_payoffs.get(agent, float("nan")),
                "session_count": len(metas),
            }

            defect_total = reduction_by_action_total[1] if num_actions > 1 else 0.0
            summary_row["defect_reduction_share"] = _safe_div(defect_total, total_reduction)

            for idx in range(num_actions):
                label = long_labels[idx]
                key_final = _format_action_key("final_cap", label)
                key_reduce = _format_action_key("reduction", label)
                key_bind = _format_action_key("binding_rate", label)
                key_slack = _format_action_key("avg_slack", label)
                key_prob = _format_action_key("avg_prob", label)

                summary_row[key_final] = final_caps_avg[idx]
                summary_row[key_reduce] = reduction_by_action_total[idx]
                summary_row[key_bind] = _safe_div(binding_counts_total[idx], total_binding)
                summary_row[key_slack] = _safe_div(slack_sums_total[idx], total_binding)
                summary_row[key_prob] = _safe_div(prob_sums_total[idx], total_binding)

            summary_rows.append(summary_row)

        agents = list(caps_history.keys())
        reduced_agents = {
            meta["agent"]
            for meta in session_meta.values()
            if meta["total_reduction"] > 0
        }
        avg_final_defect_cap = _safe_div(
            sum(
                meta["final_caps"][1] if meta["num_actions"] > 1 else meta["final_caps"][0]
                for meta in session_meta.values()
            ),
            len(session_meta),
        )

        run_rows.append(
            {
                "run_dir": str(run.run_dir),
                "game": run.game,
                "agents": "|".join(agents),
                "session_count": len(session_meta),
                "avg_negotiation_rounds": _safe_div(
                    sum(meta["negotiation_rounds"] for meta in session_meta.values()),
                    len(session_meta),
                ),
                "agents_reduced": len(reduced_agents),
                "total_cap_reduction": run_total_reduction,
                "first_change_round": run_first_change if run_first_change is not None else "",
                "avg_final_defect_cap": avg_final_defect_cap,
                "avg_payoff": mean(run.expected_payoffs.get(agent, 0.0) for agent in agents),
            }
        )

    if not summary_rows:
        return {}

    caps_path = out_dir / "disarm_caps_rounds.csv"
    sessions_path = out_dir / "disarm_session_summary.csv"
    summary_path = out_dir / "disarm_agent_summary.csv"
    runs_path = out_dir / "disarm_run_summary.csv"
    coop_path = out_dir / "disarm_round_cooperation.csv"

    write_csv(caps_round_rows, caps_path)
    write_csv(session_rows, sessions_path)
    write_csv(summary_rows, summary_path)
    write_csv(run_rows, runs_path)
    write_csv(coop_rows, coop_path)

    plot_disarm_cap_trajectories(caps_round_rows, out_dir)
    plot_disarm_reduction_bars(summary_rows, out_dir)
    plot_disarm_binding_heatmap(summary_rows, out_dir)
    plot_disarm_cooperation_trajectory(coop_rows, out_dir)
    plot_disarm_payoff_vs_caps(summary_rows, out_dir)
    plot_disarm_negotiation_hist(session_rows, out_dir)

    return {
        "disarm_caps_rounds": caps_path,
        "disarm_session_summary": sessions_path,
        "disarm_agent_summary": summary_path,
        "disarm_run_summary": runs_path,
        "disarm_round_cooperation": coop_path,
    }
