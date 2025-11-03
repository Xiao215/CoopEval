from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .io import RunPaths, find_runs, load_config, load_histories, load_payoffs
from .metrics import (
    cooperation_rate_from_histories,
    expected_payoffs_from_payoffs_json,
    mechanism_effectiveness,
)


@dataclass
class RunSummary:
    run_dir: str
    game: str | None
    mechanism: str | None
    agents: List[str]
    coop_rates: Dict[str, float]
    expected_payoffs: Dict[str, float]
    coop_average: float


def summarize_run(run: RunPaths) -> RunSummary:
    config = load_config(run) or {}
    payoffs = load_payoffs(run) or {}
    histories = load_histories(run)

    game = (config.get("game") or {}).get("type")
    mechanism = (config.get("mechanism") or {}).get("type")
    agents = [a.get("name", "unknown") for a in (config.get("agents") or [])]

    coop_rates = cooperation_rate_from_histories(histories)
    expected = expected_payoffs_from_payoffs_json(payoffs)
    coop_avg = mechanism_effectiveness(coop_rates)

    return RunSummary(
        run_dir=str(run.root),
        game=game,
        mechanism=mechanism,
        agents=agents,
        coop_rates=coop_rates,
        expected_payoffs=expected,
        coop_average=coop_avg,
    )


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def aggregate(root: str | Path, out_dir: str | Path) -> None:
    runs = find_runs(root)
    summaries = [summarize_run(r) for r in runs]

    # Save per-run summaries
    write_json([s.__dict__ for s in summaries], Path(out_dir) / "summaries.json")

    # Flatten per-agent metrics across runs for CSVs
    payoff_rows: List[Dict[str, Any]] = []
    coop_rows: List[Dict[str, Any]] = []
    mech_rows: List[Dict[str, Any]] = []
    for s in summaries:
        for agent, val in s.expected_payoffs.items():
            payoff_rows.append(
                {
                    "run_dir": s.run_dir,
                    "game": s.game,
                    "mechanism": s.mechanism,
                    "agent": agent,
                    "expected_payoff": val,
                }
            )
        for agent, rate in s.coop_rates.items():
            coop_rows.append(
                {
                    "run_dir": s.run_dir,
                    "game": s.game,
                    "mechanism": s.mechanism,
                    "agent": agent,
                    "cooperation_rate": rate,
                }
            )
        mech_rows.append(
            {
                "run_dir": s.run_dir,
                "game": s.game,
                "mechanism": s.mechanism,
                "coop_average": s.coop_average,
            }
        )

    write_csv(payoff_rows, Path(out_dir) / "agent_expected_payoffs.csv")
    write_csv(coop_rows, Path(out_dir) / "agent_cooperation_rates.csv")
    write_csv(mech_rows, Path(out_dir) / "mechanism_effectiveness.csv")



