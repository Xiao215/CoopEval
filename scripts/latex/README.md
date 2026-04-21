# LaTeX Table Scripts

This directory hosts every script that emits LaTeX artifacts. The layout splits
shared logic into `tablelib/` so multiple entry points can reuse the same data
parsing and formatting code.

- `tablelib/data_loader.py` – discover experiment folders, load JSON, and validate groups.
- `tablelib/metrics.py` – normalization helpers, raw tensor construction, and shared I/O helpers.
- `tablelib/game_table.py` – per-game LaTeX table renderer and ranking utilities.
- `tablelib/aggregate.py` – aggregate-table builder with pluggable formatter strategies.
- `tablelib/colors.py` – shared LaTeX heatmap helpers used by normalized and percent outputs.
- `tablelib/formatters.py` – strategy objects for normalized vs. percent output.
- `tablelib/cli.py` – shared argument wiring and path normalization helpers.

Entry points:

- `generate_tables_normalized.py` – aggregate table on the normalized 0–1 scale.
- `generate_tables_per_game.py` – combined per-game tables on the normalized 0–1 scale.
- `generate_contract_mediation_analysis_table.py` – aggregate NS/WD table for mediation and contracting design analysis, with optional `--skip-games` and `--per-game` for one table per game instead of cross-game averaging.

The normalized and per-game entry points write to `LATEX_DIR` from `coopeval.config`.
They accept `--color` to add the shared LaTeX heatmap cell backgrounds.

Run these commands from the repository root.

## `generate_tables_normalized.py`

```bash
python scripts/latex/generate_tables_normalized.py \
  --tournament_result_dirs data/main_study
```

Add heatmap cell backgrounds with:

```bash
python scripts/latex/generate_tables_normalized.py \
  --tournament_result_dirs data/main_study \
  --color
```

## `generate_tables_per_game.py`

```bash
python scripts/latex/generate_tables_per_game.py \
  --tournament_result_dirs data/main_study
```

Add heatmap cell backgrounds with:

```bash
python scripts/latex/generate_tables_per_game.py \
  --tournament_result_dirs data/main_study \
  --color
```

## `generate_contract_mediation_analysis_table.py`

```bash
python scripts/latex/generate_contract_mediation_analysis_table.py \
  --tournament_result_dirs data/main_study
```

Generate one table per selected game instead of averaging across games with:

```bash
python scripts/latex/generate_contract_mediation_analysis_table.py \
  --tournament_result_dirs data/main_study \
  --per-game
```
