# 🧠 LLM Evolution Tournament

LLM Evolution Tournament is a research playground for studying cooperation between large-language-model (LLM) agents. Agents compete (or collaborate) in repeated games, and their population share evolves according to observed payoffs—letting you explore how incentives, communication styles, and prompting strategies shape behaviour.

---

## Why this repo

- **Compare LLM behaviours** across classic strategic environments (Prisoner’s Dilemma, Traveller’s Dilemma, Public Goods, Trust Game, …).
- **Swap in mechanisms** (reputation, mediation, disarmament, contracting, etc.) to see which incentives reinforce cooperation.
- **Experiment with agent prompting** (chain-of-thought vs. direct-answer IO agents) and run them on either hosted APIs (OpenAI-compatible) or local Hugging Face checkpoints.
- **Analyse evolutionary dynamics** using discrete replicator updates driven by actual tournament outcomes.

---

## Installation

> Python 3.12

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to run local HF checkpoints, ensure your environment can access the weights (see `MODEL_WEIGHTS_DIR` in `config.py`).

---

## Quick Start

1. **Pick a configuration** from `configs/` or copy and edit one.
2. **Run the evolutionary loop**:

```bash
python script/run_evolution.py --config configs/legacy/toy_pd.yaml --log
```

3. **Inspect outputs** in `outputs/<date>/...`.

### Running on a cluster (Vector example)

```bash
./submit.sh          # wraps sbatch and uses run_job.sh under the hood
```

Both scripts are pre-filled for Vector’s SLURM setup—adjust `account`, GPUs, and walltime as needed.

---

## Configuration glossary

```yaml
evolution:
  initial_population: "uniform"   # or "random", or provide an explicit numpy vector
  steps: 200

mechanism:
  type: Repetition                 # see list below
  kwargs:
    num_rounds: 5
    discount: 0.95

game:
  type: PrisonersDilemma           # see list below
  kwargs:
    payoff_matrix:
      CC: [3, 3]
      CD: [0, 5]
      DC: [5, 0]
      DD: [1, 1]

agents:
  - llm:
      provider: HFInstance         # or OpenAI / OpenRouter / Gemini
      model: Meta-Llama-3.1-8B-Instruct
      kwargs:
        max_new_tokens: 512
    type: CoTAgent                 # or IOAgent
  - llm:
      provider: OpenAI
      model: gpt-4o-mini
      kwargs:
        max_tokens: 512
    type: IOAgent
```

### Available games (`src/games/`)

| Class                | Description |
|----------------------|-------------|
| `PrisonersDilemma`   | Two-player PD with configurable payoff matrix. |
| `PublicGoods`        | N-player public goods contribution with multiplier and optional parallel prompting. |
| `TravellersDilemma`  | Two-player traveller’s dilemma parameterised by minimum claim, spacing, bonus. |
| `TrustGame`          | Two-player simultaneous trust game (invest vs. keep). |

### Available mechanisms (`src/mechanisms/`)

| Class                         | Purpose |
|-------------------------------|---------|
| `NoMechanism`                 | Single-shot game (baseline). |
| `Repetition`                  | Repeats the base game for fixed rounds (with optional history prompt). |
| `Disarmament`                 | Negotiation of per-action probability caps before each round. |
| `Mediation`                   | Agents may delegate to a learned mediator design. |
| `Contracting`                 | Agents propose/agree to payoff-altering contracts. |
| `ReputationPrisonersDilemma`  | Tracks cooperation rates and exposes them as public info. |
| `ReputationPublicGoods`       | Tracks contributions in the public goods setting. |

### Agent wrappers (`src/agents/`)

- `IOAgent`: direct answer style (no extra reasoning instructions).
- `CoTAgent`: appends “think step by step” prompts to encourage chain-of-thought.
- Backends provided by `LLMManager`:
  - `HFInstance` (local Hugging Face checkpoints, with automatic device placement).
  - `ClientAPILLM` (OpenAI-compatible API clients: OpenAI, Gemini, OpenRouter). Configure API keys in `config.py` / environment variables via `settings`.

---

## Evolution loop (under the hood)

1. **Tournament**: The active mechanism runs all required matchups, producing a `PopulationPayoffs` object with seat-level histories and aggregates.
2. **Fitness**: The payoff table calculates expected payoffs for the current population distribution.
3. **Update**: `DiscreteReplicatorDynamics` applies an exponential-weight update (`population_update`) and normalises the distribution.

The process repeats for `evolution.steps` iterations or until convergence tolerance `tol` is reached.

Key files to inspect:

- `src/evolution/population_payoffs.py`
- `src/evolution/replicator_dynamics.py`

---

## Concurrency model

- **Games** share a `_collect_actions` helper that can prompt agents either sequentially or in parallel (`parallel_players=True`).
- **Mechanisms** use a common `run_tasks` helper (`src/utils/concurrency.py`) to fan out negotiations, mediator queries, etc.
- Seat cloning (`Agent.make_seat_clone`) produces human-friendly labels like `Gemma(CoT)#2`, preventing name collisions when identical models face off.

---

## Repository layout

```
.
├── configs/              # YAML experiment templates
├── script/run_evolution.py
├── src/
│   ├── agents/          # Agent abstractions & LLM backends
│   ├── evolution/       # Population payoffs + replicator dynamics
│   ├── games/           # Game definitions
│   ├── mechanisms/      # Incentive / enforcement layers
│   └── utils/           # Shared helpers (concurrency, etc.)
├── outputs/             # Logs and prompts generated per run
├── run_job.sh, submit.sh
└── README.md
```

---

## Contributing / extending

- **Add a new game**: create a class in `src/games/`, register it in `src/registry/game_registry.py`, and include any prompts/instructions.
- **Add a mechanism**: subclass `Mechanism` (or `RepetitiveMechanism`), implement `_play_matchup`, then register in `src/registry/mechanism_registry.py`.
- **Add agent types**: subclass `Agent`, implement `chat`, and register in `src/registry/agent_registry.py` (if available).

PRs & issues welcome—this project is evolving alongside our experiments.

---

Happy experimenting and may your agents cooperate (when you want them to)!
