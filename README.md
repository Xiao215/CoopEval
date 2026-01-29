# CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas
This repository contains the official implementation for the paper **"CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas"**.

It serves as a research framework for simulating societies of Large Language Model (LLM) agents interacting in mixed-motive games. The codebase enables the evaluation of how game-theoretic mechanisms—such as repetition, reputation, mediation, and contracts—can enforce cooperation among selfish, rational agents.

---

## 📂 Repository Overview

- **Comparisons:** Evaluate agents across 4 classic social dilemmas: **Prisoner’s Dilemma**, **Traveler’s Dilemma**, **Public Goods**, and **Trust Game**.
- **Mechanisms:** Implementations of the four cooperation mechanisms studied in the paper:
  - **Repetition:** Iterated games with history (Direct Reciprocity).
  - **Reputation:** Interactions with varying partners and observable histories (Indirect Reciprocity).
  - **Mediation:** Delegation to a third-party entity designed by agents.
  - **Contracting:** Binding agreements for outcome-conditioned payoff transfers.
- **Evolutionary Dynamics:** Tools to simulate population adaptation using **Discrete Replicator Dynamics** based on tournament payoffs.

---

## 🛠️ Installation

The codebase requires **Python 3.12**.

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to run local Hugging Face checkpoints, ensure your environment can access the weights (set `MODEL_WEIGHTS_DIR` in `config.py` if necessary).

---

## 🔑 API Configuration

This framework supports multiple LLM providers. You must provide API keys using either a `.env` file (recommended) or by exporting environment variables.

### Option 1: Using a .env file (Recommended)
Create a file named `.env` in the root directory:

```bash
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

### Option 2: Exporting variables
Alternatively, export them directly in your terminal:

```bash
export OPENAI_API_KEY='your_openai_key_here'
export GEMINI_API_KEY='your_gemini_key_here'
export OPENROUTER_API_KEY='your_openrouter_key_here'
```

---

## 🚀 Quick Start

To reproduce the main results, you can either run a single custom experiment or use the batch runner.

### Run a Single Experiment

The system requires a configuration file that combines a **Game**, **Mechanism**, and **Agent** set.

1. Create a configuration file (e.g., `configs/quick_start.yaml`):

```yaml
# configs/quick_start.yaml
game_config: "games/prisoners_dilemma.yaml"
mechanism_config: "mechanisms/repetition.yaml"
agents_config: "agents/test_agents_6.yaml"
evaluation_config: "evaluation/no_deviation_ratings.yaml"
name: "quick_start_test"
concurrency:
  max_workers: 4
  tournament_workers: 1
```

2. Run the experiment script:

```bash
export PYTHONPATH=.
python script/run_experiment.py --config configs/quick_start.yaml --output-dir outputs/
```

---

## ⚙️ Configuration Glossary

### Supported Games (`src/games/`)

| Class | Description |
| :--- | :--- |
| `PrisonersDilemma` | Two-player PD with configurable payoff matrix. |
| `PublicGoods` | N-player contribution game with multiplier $\alpha$ and redistribution. |
| `TravellersDilemma` | Two-player race-to-the-bottom parameterised by claims $\{2..k\}$. |
| `TrustGame` | Two-player sequential investment and return game. |

### Supported Mechanisms (`src/mechanisms/`)

| Class | Description |
| :--- | :--- |
| `NoMechanism` | Single-shot baseline. |
| `Repetition` | Repeated interactions with the same partner (grim trigger capable). |
| `Reputation` | Interactions with changing partners; visible history (first or higher-order). |
| `Mediation` | Agents propose and vote on a mediator delegate. |
| `Contracting` | Agents propose and sign binding payoff-transfer contracts. |

---

## 🧠 Code Structure

```text
.
├── configs/                # Experiment configurations (Games, Mechanisms, Agents)
├── script/
│   ├── run_experiment.py   # Entry point for single-tournament evaluations
│   └── run_evolution.py    # Entry point for population-level replicator dynamics
├── src/
│   ├── agents/             # LLM API wrappers & persona-driven prompting logic
│   ├── games/              # Logic for social dilemmas (PD, PGG, Trust, etc.)
│   ├── mechanisms/         # Cooperation enforcement (Reputation, Contracting, etc.)
│   ├── ranking_evolutions/ # Population fitness & Discrete Replicator Dynamics
│   └── utils/              # Async IO, logging, and metrics calculation
└── requirements.txt        # Reproducible environment specifications
```