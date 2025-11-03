from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .io import RunPaths, load_config, load_payoffs
from .models import MoveRecord, RunData
from .metrics import expected_payoffs_from_payoffs_json


def _load_jsonl(path: Path) -> List[Any]:
    items: List[Any] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _make_match_id(round_index: int, matchup_key: Any) -> str:
    return f"r{round_index}:{matchup_key}"


def parse_run(run: RunPaths) -> RunData:
    config = load_config(run) or {}
    game = (config.get("game") or {}).get("type", "Unknown")
    mechanism = (config.get("mechanism") or {}).get("type", "Unknown")
    agents_cfg = config.get("agents") or []
    agents = [a.get("name", f"agent_{idx}") for idx, a in enumerate(agents_cfg)]

    payoffs_json = load_payoffs(run) or {}
    expected_payoffs = expected_payoffs_from_payoffs_json(payoffs_json)

    history_files = run.history_paths

    if not history_files:
        return RunData(
            run=run,
            config=config,
            game=game,
            mechanism=mechanism,
            agents=agents,
            moves=[],
            matchups={},
            rounds=[],
            expected_payoffs=expected_payoffs,
            mechanism_payload={},
        )

    # Currently the pipeline writes a single history jsonl per mechanism
    history_path = history_files[0]
    records = _load_jsonl(history_path)

    moves: List[MoveRecord] = []
    matchups: Dict[str, List[MoveRecord]] = defaultdict(list)
    rounds: List[List[List[MoveRecord]]] = []
    mechanism_payload: Dict[str, Any] = {"payoffs": payoffs_json}

    if mechanism in {"Repetition", "ReputationPrisonersDilemma", "ReputationPublicGoods"}:
        # records: list per round, each round list of matchups, each matchup list of moves
        for round_idx, round_data in enumerate(records):
            round_moves: List[List[MoveRecord]] = []
            for matchup_idx, matchup in enumerate(round_data):
                match_id = _make_match_id(round_idx, matchup_idx)
                match_records: List[MoveRecord] = []
                for move in matchup:
                    record = MoveRecord(
                        agent=move.get("name", "unknown"),
                        action=str(move.get("action", "")),
                        points=float(move.get("points", 0.0)),
                        response=str(move.get("response", "")),
                        round_index=round_idx,
                        matchup_index=matchup_idx,
                        metadata={"match_id": match_id},
                    )
                    moves.append(record)
                    matchups[match_id].append(record)
                    match_records.append(record)
                round_moves.append(match_records)
            rounds.append(round_moves)

    elif mechanism == "Disarmament":
        caps_history: Dict[str, Dict[str, List[List[float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for round_idx, round_entry in enumerate(records):
            # Each entry is a list with a single matchup (list of move dicts)
            round_matchups: List[List[MoveRecord]] = []
            matchup = round_entry[0] if round_entry else []
            match_records: List[MoveRecord] = []
            for move in matchup:
                agent = move.get("name", "unknown")
                match_tag = move.get("match_id") or "session_0"
                new_cap = move.get("new_cap") or []
                caps_history[agent][match_tag].append(new_cap)
                record = MoveRecord(
                    agent=agent,
                    action=str(move.get("action", "")),
                    points=float(move.get("points", 0.0)),
                    response=str(move.get("response", "")),
                    round_index=round_idx,
                    matchup_index=0,
                    metadata={"match_id": match_tag, "new_cap": new_cap},
                )
                moves.append(record)
                matchups[match_tag].append(record)
                match_records.append(record)
            round_matchups.append(match_records)
            if not round_matchups:
                round_matchups.append([])
            rounds.append(round_matchups)
        mechanism_payload["caps_history"] = {
            agent: dict(sessions) for agent, sessions in caps_history.items()
        }

    elif mechanism == "Mediation":
        mediator_rounds: List[Dict[str, Any]] = []
        for round_idx, mediator_entries in enumerate(records):
            round_moves: List[List[MoveRecord]] = []
            for entry in mediator_entries:
                mediator_name = entry.get("mediator", "unknown")
                match_id = _make_match_id(round_idx, mediator_name)
                move_records: List[MoveRecord] = []
                for move in entry.get("moves", []):
                    record = MoveRecord(
                        agent=move.get("name", "unknown"),
                        action=str(move.get("action", "")),
                        points=float(move.get("points", 0.0)),
                        response=str(move.get("response", "")),
                        round_index=round_idx,
                        matchup_index=None,
                        mediator=mediator_name,
                        metadata={"match_id": match_id, "mediator": mediator_name},
                    )
                    moves.append(record)
                    matchups[match_id].append(record)
                    move_records.append(record)
                round_moves.append(move_records)
                mediator_rounds.append({"mediator": mediator_name, "moves": move_records})
            rounds.append(round_moves)
        mechanism_payload["mediator_rounds"] = mediator_rounds

    else:
        # default parsing: treat each record as list of moves (single matchup)
        for round_idx, entry in enumerate(records):
            match_id = _make_match_id(round_idx, 0)
            round_records: List[MoveRecord] = []
            if isinstance(entry, list):
                for move in entry:
                    record = MoveRecord(
                        agent=move.get("name", "unknown"),
                        action=str(move.get("action", "")),
                        points=float(move.get("points", 0.0)),
                        response=str(move.get("response", "")),
                        round_index=round_idx,
                        matchup_index=0,
                        metadata={"match_id": match_id},
                    )
                    moves.append(record)
                    matchups[match_id].append(record)
                    round_records.append(record)
            rounds.append([round_records])

    return RunData(
        run=run,
        config=config,
        game=game,
        mechanism=mechanism,
        agents=agents,
        moves=moves,
        matchups=dict(matchups),
        rounds=rounds,
        expected_payoffs=expected_payoffs,
        mechanism_payload=mechanism_payload,
    )
