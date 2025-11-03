Data Analysis for LLM Evolution Tournament

This package provides scripts to load experiment outputs under `results/`, compute cooperation and payoff metrics, aggregate across mechanisms and games, and generate plots and CSV/JSON summaries to address the current research questions.

Inputs expected per run directory (e.g., `results/david/`):
- `config.json`: full configuration used for the run
- `payoffs.json`: discounted profile table and expected payoffs per agent
- `*.jsonl`: mechanism-specific per-round histories (optional)

Quick use:
```bash
python -m data_analysis.cli summarize --root results/ --out outputs/analysis
python -m data_analysis.cli plot --root results/ --out outputs/analysis/figures
```

Outputs:
- CSVs for agent payoffs and cooperation
- JSON summaries per mechanism and game
- Figures for mechanism effectiveness and agent performance



