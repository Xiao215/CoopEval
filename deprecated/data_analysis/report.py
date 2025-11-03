from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float)) and not (v != v)]
    return sum(vals) / len(vals) if vals else 0.0


def _group_mean(rows: List[Dict[str, Any]], key: str, *columns: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group_key = row.get(key, "Unknown")
        for col in columns:
            groups[group_key][col].append(_float(row.get(col)))
    return {g: {col: _mean(vals) for col, vals in cols.items()} for g, cols in groups.items()}


def _format_table(headers: List[str], rows: List[List[Any]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join([header_line, divider, body])


def build_markdown_report(out_dir: Path) -> Path:
    agent_rows = load_csv(out_dir / "agent_metrics.csv")
    mechanism_rows = load_csv(out_dir / "mechanism_summary.csv")
    repetition_rows = load_csv(out_dir / "repetition_metrics.csv")
    disarm_rows = load_csv(out_dir / "disarmament_metrics.csv")
    reputation_rows = load_csv(out_dir / "reputation_metrics.csv")
    mediator_rows = load_csv(out_dir / "mediation_mediator_metrics.csv")
    mediation_agent_rows = load_csv(out_dir / "mediation_agent_metrics.csv")

    lines: List[str] = ["# Tournament Analysis Report", ""]

    # General metrics per mechanism
    mech_summary = _group_mean(mechanism_rows, "mechanism", "avg_expected_payoff", "avg_delta_expected_payoff", "avg_cooperation_rate", "avg_delegate_rate")
    lines.append("## Mechanism Summary")
    rows = []
    for mech, stats in sorted(mech_summary.items()):
        rows.append([
            mech,
            f"{stats.get('avg_expected_payoff', 0.0):.3f}",
            f"{stats.get('avg_delta_expected_payoff', 0.0):.3f}",
            f"{stats.get('avg_cooperation_rate', 0.0):.3f}",
            f"{stats.get('avg_delegate_rate', 0.0):.3f}",
        ])
    lines.append(_format_table(["Mechanism", "Avg Payoff", "Δ Payoff vs Base", "Avg Cooperation", "Avg Delegation"], rows))
    lines.append("")

    # Agent-level aggregates
    agent_summary = _group_mean(agent_rows, "mechanism", "expected_payoff", "coop_rate", "defection_rate", "payoff_std", "win_rate")
    lines.append("## Agent Metrics by Mechanism")
    rows = []
    for mech, stats in sorted(agent_summary.items()):
        rows.append([
            mech,
            f"{stats.get('expected_payoff', 0.0):.3f}",
            f"{stats.get('coop_rate', 0.0):.3f}",
            f"{stats.get('defection_rate', 0.0):.3f}",
            f"{stats.get('payoff_std', 0.0):.3f}",
            f"{stats.get('win_rate', 0.0):.3f}",
        ])
    lines.append(_format_table(["Mechanism", "Avg Payoff", "Coop Rate", "Defection Rate", "Payoff Std", "Win Rate"], rows))
    lines.append("")

    # Repetition metrics
    if repetition_rows:
        rep_summary = _group_mean(repetition_rows, "mechanism", "coop_trend_slope")
        lines.append("## Repetition Metrics")
        rows = [[mech, f"{stats.get('coop_trend_slope', 0.0):.4f}"] for mech, stats in sorted(rep_summary.items())]
        lines.append(_format_table(["Mechanism", "Coop Trend Slope"], rows))
        lines.append("")

    # Disarmament metrics
    if disarm_rows:
        dis_summary = _group_mean(disarm_rows, "mechanism", "final_cap_mean", "total_cap_reduction")
        lines.append("## Disarmament Metrics")
        rows = [
            [
                mech,
                f"{stats.get('final_cap_mean', 0.0):.2f}",
                f"{stats.get('total_cap_reduction', 0.0):.2f}",
            ]
            for mech, stats in sorted(dis_summary.items())
        ]
        lines.append(_format_table(["Mechanism", "Final Cap Mean", "Total Cap Reduction"], rows))
        lines.append("")

    # Reputation metrics
    if reputation_rows:
        repu_summary = _group_mean(reputation_rows, "mechanism", "final_reputation", "average_reputation", "reputation_volatility")
        lines.append("## Reputation Metrics")
        rows = [
            [
                mech,
                f"{stats.get('final_reputation', 0.0):.3f}",
                f"{stats.get('average_reputation', 0.0):.3f}",
                f"{stats.get('reputation_volatility', 0.0):.3f}",
            ]
            for mech, stats in sorted(repu_summary.items())
        ]
        lines.append(_format_table(["Mechanism", "Final Reputation", "Average Reputation", "Volatility"], rows))
        lines.append("")

    # Mediation metrics
    if mediator_rows:
        med_summary = _group_mean(mediator_rows, "mechanism", "success_rate", "avg_points")
        lines.append("## Mediation Metrics")
        rows = [
            [
                mech,
                f"{stats.get('success_rate', 0.0):.3f}",
                f"{stats.get('avg_points', 0.0):.3f}",
            ]
            for mech, stats in sorted(med_summary.items())
        ]
        lines.append(_format_table(["Mechanism", "Mediator Success Rate", "Mediator Avg Points"], rows))
        lines.append("")

    if mediation_agent_rows:
        med_agent_summary = _group_mean(mediation_agent_rows, "mechanism", "delegation_rate", "avg_points_delegate", "avg_points_independent")
        rows = [
            [
                mech,
                f"{stats.get('delegation_rate', 0.0):.3f}",
                f"{stats.get('avg_points_delegate', 0.0):.3f}",
                f"{stats.get('avg_points_independent', 0.0):.3f}",
            ]
            for mech, stats in sorted(med_agent_summary.items())
        ]
        lines.append(_format_table(["Mechanism", "Delegation Rate", "Avg Points (Delegate)", "Avg Points (Independent)"], rows))
        lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

