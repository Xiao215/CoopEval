"""
Plot evolutionary dynamics results from experiment outputs.
"""
import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to import src modules
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
    # Load data
    output_dir = "outputs/2026/01/14/16:14/no_mechanism_public_goods"
    pop_history, config, avg_payoffs = load_experiment_data(output_dir)
    
    # Extract agent names (models)
    agent_names = list(pop_history[0].keys())
    print(f"Agents: {agent_names}")
    print(f"Average payoffs: {avg_payoffs}")
    
    # Convert population history to trajectory matrix
    # Each timestep is a dict, convert to list of arrays
    trajectory = []
    for timestep in pop_history:
        shares = [timestep[agent] for agent in agent_names]
        trajectory.append(np.array(shares))
    
    # Plot 1: Population evolution stacked area chart
    print("\nPlotting population evolution...")
    plot_probability_evolution(
        trajectory=trajectory,
        wb=None,  # No WandB logging
        save_local=True,
        labels=agent_names,
        figsize=(12, 6)
    )
    print("Saved population_evolution.png")
    
    # Plot 2: Share progression with payoffs (if you have payoff trajectory)
    # Note: This requires a PopulationPayoff object and full dynamics tuple
    # For now, we'll just show the population evolution plot
    
    print("\nPlotting complete! Check figures/ directory for outputs.")


if __name__ == "__main__":
    main()
