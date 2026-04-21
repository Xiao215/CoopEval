# Experiment Script Commands

Run these commands from the repository root.

## `run_experiment.py`

```bash
python scripts/experiments/run_experiment.py \
  --config <config.yaml>
```

```bash
python scripts/experiments/run_experiment.py \
  --config <config.yaml> \
  --output-dir outputs/manual_runs \
  --experiment-name <experiment-name> \
  --seed 42
```

## `run_batch.sh`

```bash
bash scripts/experiments/run_batch.sh \
  --local \
  --batch-name <batch-name>
```

```bash
bash scripts/experiments/run_batch.sh \
  --slurm \
  --batch-name <batch-name>
```

```bash
bash scripts/experiments/run_batch.sh \
  --local \
  --resume outputs/<batch-dir>
```

## `run_single_experiment.sh`

```bash
bash scripts/experiments/run_single_experiment.sh \
  <experiment-index> \
  <batch-dir>
```

## `main_runner.sh`

```bash
bash scripts/experiments/main_runner.sh
```

## `slurm_array_template.sh`

This file is used by `run_batch.sh --slurm`; it is not usually run directly.
