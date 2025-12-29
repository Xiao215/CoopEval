# 🧠 LLM Evolution Tournament

**LLM Evolution Tournament** is a research framework for studying how large-language-model (LLM) agents learn to cooperate.
It supports evolutionary-style simulations in which each agent’s population share adapts over repeated game rounds according to performance.

---

## Key Objectives
- **Measure cooperation** Quantify how LLM agents behave in classic strategic-game settings.
- **Compare mechanisms** Swap in different game rules or incentive structures to see what fosters cooperation.
- **Benchmark models** Evaluate any LLM (OpenAI, Anthropic, Qwen, etc.) under identical, reproducible tasks.

---

## Quick-start

> **Requires Python 3.10**

```bash
# 1  Activate a virtual environment
python3.12 -m venv .venv312
source .venv312/bin/activate      # Windows: .venv312\Scripts\activate

# 2  Install dependencies
module load rust
pip install -r requirements.txt
```

---

## Running Experiments

1. **Choose or create a config** (YAML) in `configs/`.
2. **Launch the run script:**

```bash
python script/run_evolution.py --config your_experiment.yaml --log
```


### Sample configuration

```
evolution:
  initial_population: "uniform"  # or "random"
  steps: <RoundsOfEvolution>

game:
  type: <GameClassName>
  kwargs:
    <GameParameter>

mechanism:
  type: <MechanismClassName>
  kwargs:
    <MechanismParameter>

agents:
  - llm:
      model: <Model1Name>
      kwargs:
        <Model1Parameter>
    type: <Model1Class>
  - llm:
      model: <Model2Name>
      kwargs:
        <Model2Parameter>
    type: <Model2Class>
  # Add more agents as needed
```


### Running on a Vector server

1. Edit `run_job.sh` and `submit.sh` with your parameters.
2. Submit the job:

```bash
./submit.sh
```


---

## Repository Layout
```
llm-tournament/
├── configs/          # YAML experiment files
├── script/           # CLI entry points
├── src/
│   ├── evolution/    # Replicator-dynamics algorithms
│   ├── games/        # Game definitions
│   ├── mechanisms/   # Incentive / payoff modifiers
│   └── agents.py     # LLM-agent wrappers
├── run_job.sh
└── submit.sh
```
