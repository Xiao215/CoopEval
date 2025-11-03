from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .io import RunPaths, find_runs, load_config


def _iter_jsonl(path: Path) -> Iterable[Any]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _is_coop(action: str) -> bool:
    # PD uses "C" for cooperate
    return action.upper().startswith("C")


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def pairwise_metrics_for_run(run: RunPaths) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cfg = load_config(run) or {}
    game = (cfg.get("game") or {}).get("type")
    mechanism = (cfg.get("mechanism") or {}).get("type")

    # Aggregate per pair
    acc: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for hp in run.history_paths:
        for entry in _iter_jsonl(hp):
            # entry is list of rounds (each round is list of move dicts)
            rounds = entry if (isinstance(entry, list) and entry and isinstance(entry[0], list)) else [entry]
            for round_moves in rounds:
                if not isinstance(round_moves, list) or len(round_moves) < 2:
                    continue
                m1, m2 = round_moves[0], round_moves[1]
                p1, p2 = m1["name"], m2["name"]
                k = _pair_key(p1, p2)
                s = acc.setdefault(k, {
                    "p1": k[0], "p2": k[1],
                    "rounds": 0,
                    "sum_pay": {k[0]: 0.0, k[1]: 0.0},
                    "sum_coop": {k[0]: 0.0, k[1]: 0.0},
                })
                # Map back to sorted positions
                name_to_move = {m1["name"]: m1, m2["name"]: m2}
                for name in k:
                    m = name_to_move[name]
                    s["sum_pay"][name] += float(m.get("points", 0.0))
                    s["sum_coop"][name] += 1.0 if _is_coop(m.get("action", "")) else 0.0
                s["rounds"] += 1

    for (a, b), s in acc.items():
        r = s["rounds"] or 1
        rows.append({
            "agent_i": a,
            "agent_j": b,
            "avg_payoff_i_vs_j": s["sum_pay"][a] / r,
            "avg_payoff_j_vs_i": s["sum_pay"][b] / r,
            "coop_rate_i_vs_j": s["sum_coop"][a] / r,
            "coop_rate_j_vs_i": s["sum_coop"][b] / r,
            "rounds": s["rounds"],
            "game": game,
            "mechanism": mechanism,
            "run_dir": str(run.root),
        })
    return rows


def conditional_cooperation_for_run(run: RunPaths) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cfg = load_config(run) or {}
    game = (cfg.get("game") or {}).get("type")
    mechanism = (cfg.get("mechanism") or {}).get("type")

    def update_pair_sequence(rounds: List[Dict[str, Any]]):
        # rounds: list of 2-move dicts per round
        history: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for rm in rounds:
            if not isinstance(rm, list) or len(rm) < 2:
                continue
            history.append((rm[0], rm[1]))
        if not history:
            return
        # establish names order by first round
        p1, p2 = history[0][0]["name"], history[0][1]["name"]
        stats = {
            p1: {"given_C": 0, "given_D": 0, "count_C": 0, "count_D": 0},
            p2: {"given_C": 0, "given_D": 0, "count_C": 0, "count_D": 0},
        }
        # iterate from second round, look at opponent previous action
        for t in range(1, len(history)):
            a_prev, b_prev = history[t - 1]
            a_cur, b_cur = history[t]
            # player a conditional on b_prev
            if _is_coop(b_prev.get("action", "")):
                stats[p1]["given_C"] += 1
                stats[p1]["count_C"] += 1 if _is_coop(a_cur.get("action", "")) else 0
            else:
                stats[p1]["given_D"] += 1
                stats[p1]["count_D"] += 1 if _is_coop(a_cur.get("action", "")) else 0
            # player b conditional on a_prev
            if _is_coop(a_prev.get("action", "")):
                stats[p2]["given_C"] += 1
                stats[p2]["count_C"] += 1 if _is_coop(b_cur.get("action", "")) else 0
            else:
                stats[p2]["given_D"] += 1
                stats[p2]["count_D"] += 1 if _is_coop(b_cur.get("action", "")) else 0
        for name, st in stats.items():
            denom_c = max(1, st["given_C"])  # avoid div/0
            denom_d = max(1, st["given_D"])
            rows.append({
                "agent": name,
                "p_coop_given_opp_C": st["count_C"] / denom_c,
                "p_coop_given_opp_D": st["count_D"] / denom_d,
                "game": game,
                "mechanism": mechanism,
                "run_dir": str(run.root),
            })

    for hp in run.history_paths:
        for entry in _iter_jsonl(hp):
            rounds = entry if (isinstance(entry, list) and entry and isinstance(entry[0], list)) else [entry]
            update_pair_sequence(rounds)

    return rows


def round_trajectory_for_run(run: RunPaths) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cfg = load_config(run) or {}
    game = (cfg.get("game") or {}).get("type")
    mechanism = (cfg.get("mechanism") or {}).get("type")

    for hp in run.history_paths:
        for entry in _iter_jsonl(hp):
            rounds = entry if (isinstance(entry, list) and entry and isinstance(entry[0], list)) else [entry]
            for idx, rm in enumerate(rounds, start=1):
                if not isinstance(rm, list) or len(rm) < 2:
                    continue
                coop = sum(1.0 if _is_coop(m.get("action", "")) else 0.0 for m in rm) / len(rm)
                rows.append({
                    "round": idx,
                    "avg_pair_coop": coop,
                    "game": game,
                    "mechanism": mechanism,
                    "run_dir": str(run.root),
                })
    return rows


def selection_trajectory_for_run(run: RunPaths, steps: int = 50, lr: float = 0.1) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cfg = load_config(run) or {}
    game = (cfg.get("game") or {}).get("type")
    mechanism = (cfg.get("mechanism") or {}).get("type")

    pay_path = (Path(run.root) / "payoffs.json")
    if not pay_path.exists():
        return rows
    with open(pay_path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    exp = rec.get("expected_payoff") or {}
    agents = list(exp.keys())
    fitness = [float(exp[a]) for a in agents]
    # Constant-fitness replicator (approximate)
    import math
    import numpy as np

    x = np.ones(len(agents)) / len(agents)
    for t in range(steps + 1):
        for i, a in enumerate(agents):
            rows.append({
                "step": t,
                "agent": a,
                "share": float(x[i]),
                "game": game,
                "mechanism": mechanism,
                "run_dir": str(run.root),
            })
        if t == steps:
            break
        f_vec = np.array(fitness)
        f_avg = float(np.dot(x, f_vec))
        w = x * np.exp(lr * (f_vec - f_avg))
        x = w / w.sum()
    return rows


def baseline_deltas(root: str | Path) -> List[Dict[str, Any]]:
    # Compute per-agent deltas between Repetition and NoMechanism using latest runs per mechanism
    runs = find_runs(root)
    # index by (game, mechanism, agent) → list of values (payoff, coop)
    payoff: Dict[Tuple[str, str, str], List[float]] = {}
    coop: Dict[Tuple[str, str, str], List[float]] = {}

    # We will derive from payoffs.json for payoffs and from histories for coop rate
    for r in runs:
        cfg = load_config(r) or {}
        game = (cfg.get("game") or {}).get("type")
        mechanism = (cfg.get("mechanism") or {}).get("type")
        pay_path = Path(r.root) / "payoffs.json"
        if pay_path.exists():
            with open(pay_path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            exp = rec.get("expected_payoff") or {}
            for a, v in exp.items():
                payoff.setdefault((game, mechanism, a), []).append(float(v))
        # Cooperation rate from histories
        # count coop per agent over all rounds
        coop_count: Dict[str, int] = {}
        total_count: Dict[str, int] = {}
        for hp in r.history_paths:
            for entry in _iter_jsonl(hp):
                rounds = entry if (isinstance(entry, list) and entry and isinstance(entry[0], list)) else [entry]
                for rm in rounds:
                    for m in rm:
                        name = m.get("name", "unknown")
                        total_count[name] = total_count.get(name, 0) + 1
                        if _is_coop(m.get("action", "")):
                            coop_count[name] = coop_count.get(name, 0) + 1
        for a, tot in total_count.items():
            rate = (coop_count.get(a, 0) / tot) if tot else 0.0
            coop.setdefault((game, mechanism, a), []).append(rate)

    # build deltas for agents where both mechanisms exist
    rows: List[Dict[str, Any]] = []
    import statistics as stats
    games = {g for (g, _, _) in payoff.keys()} | {g for (g, _, _) in coop.keys()}
    mechanisms = {m for (_, m, _) in payoff.keys()} | {m for (_, m, _) in coop.keys()}
    if not ("Repetition" in mechanisms and "NoMechanism" in mechanisms):
        return rows
    agents = {a for (_, _, a) in payoff.keys()} | {a for (_, _, a) in coop.keys()}
    for g in games:
        for a in agents:
            p_rep = payoff.get((g, "Repetition", a))
            p_base = payoff.get((g, "NoMechanism", a))
            c_rep = coop.get((g, "Repetition", a))
            c_base = coop.get((g, "NoMechanism", a))
            if not (p_rep and p_base and c_rep and c_base):
                continue
            rows.append({
                "game": g,
                "agent": a,
                "delta_expected_payoff": stats.mean(p_rep) - stats.mean(p_base),
                "delta_cooperation_rate": stats.mean(c_rep) - stats.mean(c_base),
                "rep_expected_payoff": stats.mean(p_rep),
                "base_expected_payoff": stats.mean(p_base),
                "rep_cooperation_rate": stats.mean(c_rep),
                "base_cooperation_rate": stats.mean(c_base),
            })
    return rows


