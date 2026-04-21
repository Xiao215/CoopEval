# Analysis Script Commands

Run these commands from the repository root.

Reusable analysis logic lives under `src/coopeval/analysis`; files in this
directory are command-line entry points.

## `plot_action_frequency.py`

```bash
python scripts/analysis/plot_action_frequency.py \
  --tournament_result_dirs data/main_study
```

## `plot_conditional_action_frequency.py`

```bash
python scripts/analysis/plot_conditional_action_frequency.py \
  --tournament_result_dirs data/main_study
```

## `plot_voting_adoption.py`

```bash
python scripts/analysis/plot_voting_adoption.py \
  --tournament_result_dirs data/main_study
```

## `plot_contract_design_quality.py`

```bash
python scripts/analysis/plot_contract_design_quality.py \
  --tournament_result_dirs data/main_study
```

## `plot_mediation_design_quality.py`

```bash
python scripts/analysis/plot_mediation_design_quality.py \
  --tournament_result_dirs data/main_study
```

## `plot_evo_degradation.py`

```bash
python scripts/analysis/plot_evo_degradation.py \
  --tournament_result_dirs data/main_study
```

## `plot_population_evolution.py`

```bash
python scripts/analysis/plot_population_evolution.py \
  --tournament_result_dirs data/main_study
```
