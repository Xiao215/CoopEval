#!/usr/bin/env python3
"""
Demo script to test the improved plotting functions.
Creates sample data and generates example plots to showcase the visual improvements.
"""

import csv
import json
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt

from plots import (
    plot_mechanism_effectiveness,
    plot_agent_performance,
    plot_pairwise_and_trajectories
)

def create_sample_data():
    """Create sample CSV data for testing plots"""
    temp_dir = Path(tempfile.mkdtemp())

    # Sample mechanism effectiveness data
    mech_data = [
        {"game": "PrisonersDilemma", "mechanism": "Repetition", "coop_average": 0.75},
        {"game": "PrisonersDilemma", "mechanism": "NoMechanism", "coop_average": 0.45},
        {"game": "PrisonersDilemma", "mechanism": "TitForTat", "coop_average": 0.82},
        {"game": "PrisonersDilemma", "mechanism": "Punishment", "coop_average": 0.68},
        {"game": "StagHunt", "mechanism": "Repetition", "coop_average": 0.85},
        {"game": "StagHunt", "mechanism": "NoMechanism", "coop_average": 0.55},
    ]

    mech_csv = temp_dir / "mechanism_effectiveness.csv"
    with open(mech_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game", "mechanism", "coop_average"])
        writer.writeheader()
        writer.writerows(mech_data)

    # Sample agent payoff data
    payoff_data = [
        {"run_dir": "run1", "game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "CooperativeAgent", "expected_payoff": 3.2},
        {"run_dir": "run1", "game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "DefectiveAgent", "expected_payoff": 2.1},
        {"run_dir": "run1", "game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "TitForTatAgent", "expected_payoff": 3.8},
        {"run_dir": "run1", "game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "RandomAgent", "expected_payoff": 2.5},
    ]

    payoff_csv = temp_dir / "agent_payoffs.csv"
    with open(payoff_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_dir", "game", "mechanism", "agent", "expected_payoff"])
        writer.writeheader()
        writer.writerows(payoff_data)

    # Sample cooperation data
    coop_data = [
        {"run_dir": "run1", "game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "CooperativeAgent", "cooperation_rate": 0.9},
        {"run_dir": "run1", "game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "DefectiveAgent", "cooperation_rate": 0.1},
        {"run_dir": "run1", "game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "TitForTatAgent", "cooperation_rate": 0.8},
        {"run_dir": "run1", "game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "RandomAgent", "cooperation_rate": 0.5},
    ]

    coop_csv = temp_dir / "agent_cooperation.csv"
    with open(coop_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_dir", "game", "mechanism", "agent", "cooperation_rate"])
        writer.writeheader()
        writer.writerows(coop_data)

    # Sample pairwise data
    pairwise_data = [
        {"game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent_i": "CooperativeAgent", "agent_j": "TitForTatAgent",
         "avg_payoff_i_vs_j": 3.1, "avg_payoff_j_vs_i": 3.2,
         "coop_rate_i_vs_j": 0.85, "coop_rate_j_vs_i": 0.8, "rounds": 100},
        {"game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent_i": "DefectiveAgent", "agent_j": "RandomAgent",
         "avg_payoff_i_vs_j": 2.5, "avg_payoff_j_vs_i": 2.2,
         "coop_rate_i_vs_j": 0.1, "coop_rate_j_vs_i": 0.45, "rounds": 100},
    ]

    pairwise_csv = temp_dir / "pairwise.csv"
    with open(pairwise_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game", "mechanism", "agent_i", "agent_j",
            "avg_payoff_i_vs_j", "avg_payoff_j_vs_i",
            "coop_rate_i_vs_j", "coop_rate_j_vs_i", "rounds"
        ])
        writer.writeheader()
        writer.writerows(pairwise_data)

    # Sample conditional cooperation data
    conditional_data = [
        {"game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "TitForTatAgent", "p_coop_given_opp_C": 0.95, "p_coop_given_opp_D": 0.05},
        {"game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "CooperativeAgent", "p_coop_given_opp_C": 0.9, "p_coop_given_opp_D": 0.8},
        {"game": "PrisonersDilemma", "mechanism": "Repetition",
         "agent": "DefectiveAgent", "p_coop_given_opp_C": 0.1, "p_coop_given_opp_D": 0.05},
    ]

    conditional_csv = temp_dir / "conditional.csv"
    with open(conditional_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "game", "mechanism", "agent", "p_coop_given_opp_C", "p_coop_given_opp_D"
        ])
        writer.writeheader()
        writer.writerows(conditional_data)

    # Sample trajectory data
    trajectory_data = []
    import random
    random.seed(42)
    base_coop = 0.6
    for round_num in range(1, 21):
        # Add some realistic trajectory with slight decline over time
        coop_rate = base_coop + 0.3 * (1 - round_num/20) + random.uniform(-0.1, 0.1)
        coop_rate = max(0.1, min(0.9, coop_rate))  # Keep within bounds
        trajectory_data.append({
            "game": "PrisonersDilemma", "mechanism": "Repetition",
            "round": round_num, "avg_pair_coop": coop_rate
        })

    trajectory_csv = temp_dir / "trajectory.csv"
    with open(trajectory_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game", "mechanism", "round", "avg_pair_coop"])
        writer.writeheader()
        writer.writerows(trajectory_data)

    return {
        "temp_dir": temp_dir,
        "mechanism_csv": mech_csv,
        "payoff_csv": payoff_csv,
        "cooperation_csv": coop_csv,
        "pairwise_csv": pairwise_csv,
        "conditional_csv": conditional_csv,
        "trajectory_csv": trajectory_csv
    }

def main():
    """Generate sample plots to demonstrate visual improvements"""
    print("Creating sample data...")
    data_paths = create_sample_data()

    print("Generating improved plots...")

    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Test mechanism effectiveness plot
        print("  - Mechanism effectiveness plot")
        plot_mechanism_effectiveness(data_paths["mechanism_csv"], output_dir)

        # Test agent performance plots
        print("  - Agent performance plots")
        plot_agent_performance(data_paths["payoff_csv"], data_paths["cooperation_csv"], output_dir)

        # Test pairwise and trajectory plots
        print("  - Pairwise and trajectory plots")
        plot_pairwise_and_trajectories(
            data_paths["pairwise_csv"],
            data_paths["conditional_csv"],
            data_paths["trajectory_csv"],
            output_dir
        )

        print(f"\nDemo plots generated successfully in: {output_dir.absolute()}/figures/")
        print("Visual improvements include:")
        print("  ✓ Modern color palettes and gradients")
        print("  ✓ Enhanced typography and spacing")
        print("  ✓ Better annotations and legends")
        print("  ✓ Higher resolution output (300 DPI)")
        print("  ✓ Improved bar charts and scatter plots")
        print("  ✓ Trend lines and statistical annotations")

    except Exception as e:
        print(f"Error generating plots: {e}")
        return 1

    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(data_paths["temp_dir"], ignore_errors=True)

    return 0

if __name__ == "__main__":
    exit(main())