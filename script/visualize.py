"""
Plot evolutionary dynamics results from experiment outputs.
"""
import json
import sys
from pathlib import Path

import numpy as np

# Keep everything self-contained when running directly from the repo
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualize.plot_replicator_dynamics import (plot_probability_evolution,
                                                plot_share_progression)


def load_experiment_data(output_dir: str):
    """Load population history and other data from experiment output."""
    base_path = Path(output_dir)
    
    with open(base_path / "population_history.json") as f:
        pop_history = json.load(f)
    
    with open(base_path / "config.json") as f:
        config = json.load(f)
    
    with open(base_path / "model_average_payoff.json") as f:
        avg_payoffs = json.load(f)
    
    return pop_history, config, avg_payoffs


def main():
    # Point to a concrete run so this script is copy‑and‑paste runnable; override as needed.
    output_dir = "outputs/2026/01/14/16:14/no_mechanism_public_goods"
    pop_history, config, avg_payoffs = load_experiment_data(output_dir)
    
    # Extract agent names (models)
    agent_names = list(pop_history[0].keys())
    print(f"Agents: {agent_names}")
    print(f"Average payoffs: {avg_payoffs}")
    
    # The plotting helpers expect dense numpy arrays, so collapse the list-of-dicts history.
    trajectory = []
    for timestep in pop_history:
        shares = [timestep[agent] for agent in agent_names]
        trajectory.append(np.array(shares))
    
    # Plot the share of each agent over time; stacked area chart highlights dominance shifts.
    print("\nPlotting population evolution...")
    plot_probability_evolution(
        trajectory=trajectory,
        wb=None,  # No WandB logging
        save_local=True,
        labels=agent_names,
        figsize=(12, 6)
    )
    print("Saved population_evolution.png")
    
    # A payoff-coupled trajectory plot lives in visualize.plot_replicator_dynamics; enable it
    # once PopulationPayoff snapshots are written alongside population_history.json.
    
    print("\nPlotting complete! Check figures/ directory for outputs.")


if __name__ == "__main__":
    main()
