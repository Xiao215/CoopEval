# CoopEval Scripts

This folder contains command-line utilities for running experiments, analyzing
cleaned results, producing figures, generating LaTeX tables, and evaluating
justification text with an LLM judge.

Run scripts from the repository root unless a script says otherwise:

```bash
python scripts/<subfolder>/<script>.py --help
```

## Layout

- `scripts/experiments/`: batch and single-run experiment launchers.
- `scripts/analysis/`: analysis scripts for cleaned CoopEval result folders.
- `scripts/counter/`: lightweight counters for action nodes and player matches.
- `scripts/latex/`: LaTeX table generation utilities.
- `scripts/llm_judge/`: post-hoc justification judging, normalization, export,
  and reporting scripts.
- `scripts/llm_judge/plotting/`: plotting scripts for LLM judge outputs.
- `scripts/tests/`: helper scripts for testing run/resume behavior.

## Experiment Runs

Experiment runner commands are documented in
`scripts/experiments/README.md`.

## Analysis Figures

Analysis and plotting commands are documented in
`scripts/analysis/README.md`.

## Counters

Counter commands are documented in `scripts/counter/README.md`.

## LaTeX Tables

Table builders live in `scripts/latex/`. See `scripts/latex/README.md` for the
table-specific workflow.

```bash
python scripts/latex/generate_tables_normalized.py --help
python scripts/latex/generate_tables_per_game.py --help
```

## LLM Judge Workflow

The LLM judge scripts provide the post-hoc justification judging workflow:

- `scripts/llm_judge/run_justification_judge.py`: run taxonomy judging over
  CoopEval runs.
- `scripts/llm_judge/normalize_justification_labels.py`: normalize noisy labels
  into canonical categories.
- `scripts/llm_judge/export_taxonomy_dataset.py`: export normalized taxonomy
  data for downstream analysis.
- `scripts/llm_judge/build_justification_report.py`: generate markdown/CSV
  report views in a judge result directory.

Typical flow:

```bash
python scripts/llm_judge/run_justification_judge.py ...
python scripts/llm_judge/normalize_justification_labels.py <output-name>
python scripts/llm_judge/export_taxonomy_dataset.py ...
python scripts/llm_judge/build_justification_report.py outputs/judge/<output-name>
```

LLM judge plotting commands are documented in
`scripts/llm_judge/plotting/README.md`.
