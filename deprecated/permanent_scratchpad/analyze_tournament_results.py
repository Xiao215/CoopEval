#!/usr/bin/env python3
"""
Analyze tournament results and create scatterplots showing cooperation vs payoff
for each mechanism, with one point per model pair.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_tournament_data(base_dir):
    """Load all tournament data from outputs directory."""

    # Model names for reference
    model_names = ['gpt-oss-20b', 'deepseek-chat-v3.1', 'qwen3-next-80b-a3b-instruct', 'gpt-5-nano']

    # Create all possible pairs (10 unique combinations including self-play)
    all_pairs = []
    for i in range(len(model_names)):
        for j in range(i, len(model_names)):
            all_pairs.append((model_names[i], model_names[j]))

    # Store results by mechanism
    results = defaultdict(lambda: defaultdict(lambda: {'payoff': None, 'coop_rate': None, 'rounds': 0}))

    # Process each run directory
    for root, dirs, files in os.walk(base_dir):
        if 'payoffs.json' not in files:
            continue

        # Extract mechanism from directory name
        dir_name = os.path.basename(root)
        mechanism = dir_name.split('_')[1] if '_' in dir_name else 'unknown'

        # Load payoff data
        with open(os.path.join(root, 'payoffs.json'), 'r') as f:
            payoff_data = json.load(f)

        # Find JSONL file for cooperation data
        jsonl_file = None
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_file = os.path.join(root, file)
                break

        # Process each player pairing
        for profile in payoff_data['profiles']:
            # Get simplified player names
            players = []
            for p in profile['players']:
                if '/' in p:
                    name = p.split('/')[1].split('(')[0]
                else:
                    name = p
                players.append(name)

            pair = tuple(sorted(players))

            # Calculate average payoff for the pair
            payoffs = list(profile['discounted_average'].values())
            avg_payoff = sum(payoffs) / len(payoffs) if payoffs else 0
            results[mechanism][pair]['payoff'] = avg_payoff

            # Count rounds
            if 'rounds' in profile:
                results[mechanism][pair]['rounds'] = len(profile['rounds'])

        # Process JSONL for cooperation data
        if jsonl_file:
            pair_actions = defaultdict(lambda: defaultdict(lambda: {'C': 0, 'D': 0}))

            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)

                        # Extract moves from different formats
                        moves = []
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, list):
                                    moves.extend(item)
                                elif isinstance(item, dict):
                                    if 'moves' in item:
                                        moves.extend(item['moves'])
                                    else:
                                        moves.append(item)

                        # Track moves by pairs in this round
                        round_moves = []
                        for move in moves:
                            if isinstance(move, dict) and 'name' in move and 'action' in move:
                                player = move['name'].split('/')[1].split('(')[0]
                                action = move['action']
                                if action in ['C', 'D']:
                                    round_moves.append((player, action))

                        # Process pairs of moves
                        if len(round_moves) >= 2:
                            # Take pairs of moves (for games with multiple interactions)
                            for i in range(0, len(round_moves) - 1, 2):
                                if i + 1 < len(round_moves):
                                    p1, a1 = round_moves[i]
                                    p2, a2 = round_moves[i + 1]
                                    pair = tuple(sorted([p1, p2]))

                                    pair_actions[pair][p1][a1] += 1
                                    pair_actions[pair][p2][a2] += 1
                    except:
                        continue

            # Calculate cooperation rates for each pair
            for pair, player_actions in pair_actions.items():
                total_c = sum(player_actions[p]['C'] for p in player_actions)
                total_d = sum(player_actions[p]['D'] for p in player_actions)
                total_actions = total_c + total_d

                if total_actions > 0:
                    coop_rate = (total_c / total_actions) * 100
                    if pair in results[mechanism]:
                        results[mechanism][pair]['coop_rate'] = coop_rate

    return results, all_pairs


def create_mechanism_scatterplot(mechanism_name, data, all_pairs, ax):
    """Create a scatterplot for a single mechanism."""

    # Prepare data points
    x_values = []  # Cooperation rates
    y_values = []  # Average payoffs
    labels = []
    colors = []

    # Color map for different pair types
    color_map = {
        'self': 'red',      # Self-play
        'mixed': 'blue',    # Different models
    }

    for pair in all_pairs:
        if pair in data and data[pair]['payoff'] is not None:
            # Use 0% cooperation if no cooperation data available
            coop_rate = data[pair]['coop_rate'] if data[pair]['coop_rate'] is not None else 0

            x_values.append(coop_rate)
            y_values.append(data[pair]['payoff'])

            # Create label
            p1_short = pair[0].replace('qwen3-next-80b-a3b-instruct', 'qwen3')
            p1_short = p1_short.replace('deepseek-chat-v3.1', 'deepseek')
            p1_short = p1_short.replace('gpt-oss-20b', 'gpt-oss')
            p1_short = p1_short.replace('gpt-5-nano', 'gpt-nano')

            p2_short = pair[1].replace('qwen3-next-80b-a3b-instruct', 'qwen3')
            p2_short = p2_short.replace('deepseek-chat-v3.1', 'deepseek')
            p2_short = p2_short.replace('gpt-oss-20b', 'gpt-oss')
            p2_short = p2_short.replace('gpt-5-nano', 'gpt-nano')

            if pair[0] == pair[1]:
                labels.append(f"{p1_short} (self)")
                colors.append(color_map['self'])
            else:
                labels.append(f"{p1_short} vs {p2_short}")
                colors.append(color_map['mixed'])

    # Create scatter plot
    if x_values:
        scatter = ax.scatter(x_values, y_values, c=colors, s=100, alpha=0.6, edgecolors='black', linewidth=1)

        # Add labels for each point
        for i, label in enumerate(labels):
            ax.annotate(label, (x_values[i], y_values[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)

        # Add trend line if there are enough points with variance
        if len(x_values) > 2 and len(set(x_values)) > 1:
            try:
                z = np.polyfit(x_values, y_values, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_values), max(x_values), 100)
                ax.plot(x_trend, p(x_trend), "g--", alpha=0.5, label=f'Trend: y={z[0]:.4f}x+{z[1]:.3f}')
                ax.legend(fontsize=8)
            except:
                pass  # Skip trend line if fitting fails

    # Formatting
    ax.set_xlabel('Cooperation Rate (%)', fontsize=10)
    ax.set_ylabel('Average Payoff', fontsize=10)
    ax.set_title(f'{mechanism_name.upper()}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Set axis limits with some padding
    if x_values:
        x_padding = (max(x_values) - min(x_values)) * 0.1 if max(x_values) > min(x_values) else 5
        y_padding = (max(y_values) - min(y_values)) * 0.1 if max(y_values) > min(y_values) else 0.1

        ax.set_xlim(min(x_values) - x_padding, max(x_values) + x_padding)
        ax.set_ylim(min(y_values) - y_padding, max(y_values) + y_padding)


def print_summary_statistics(results, all_pairs):
    """Print summary statistics for each mechanism."""

    print("=" * 80)
    print("TOURNAMENT RESULTS SUMMARY")
    print("=" * 80)

    mechanisms = ['disarmament', 'mediation', 'repetition', 'reputationprisonersdilemma', 'nomechanism']

    for mechanism in mechanisms:
        if mechanism not in results:
            continue

        print(f"\n{mechanism.upper()}")
        print("-" * 70)

        # Calculate statistics
        payoffs = []
        coop_rates = []

        for pair in all_pairs:
            if pair in results[mechanism]:
                if results[mechanism][pair]['payoff'] is not None:
                    payoffs.append(results[mechanism][pair]['payoff'])
                if results[mechanism][pair]['coop_rate'] is not None:
                    coop_rates.append(results[mechanism][pair]['coop_rate'])

        if payoffs:
            print(f"  Average Payoff: {np.mean(payoffs):.3f} (std: {np.std(payoffs):.3f})")
            print(f"  Payoff Range: {min(payoffs):.3f} - {max(payoffs):.3f}")

        if coop_rates:
            print(f"  Average Cooperation: {np.mean(coop_rates):.1f}% (std: {np.std(coop_rates):.1f}%)")
            print(f"  Cooperation Range: {min(coop_rates):.1f}% - {max(coop_rates):.1f}%")

        # Show individual pair results
        print("\n  Pair Results:")
        for pair in all_pairs:
            if pair in results[mechanism] and results[mechanism][pair]['payoff'] is not None:
                payoff = results[mechanism][pair]['payoff']
                coop = results[mechanism][pair]['coop_rate'] if results[mechanism][pair]['coop_rate'] is not None else 0

                p1 = pair[0].replace('qwen3-next-80b-a3b-instruct', 'qwen3')
                p1 = p1.replace('deepseek-chat-v3.1', 'deepseek')
                p2 = pair[1].replace('qwen3-next-80b-a3b-instruct', 'qwen3')
                p2 = p2.replace('deepseek-chat-v3.1', 'deepseek')

                pair_str = f"{p1:15s} vs {p2:15s}" if p1 != p2 else f"{p1:15s} (self-play)"
                print(f"    {pair_str}: Payoff={payoff:.3f}, Coop={coop:.1f}%")


def main():
    """Main function to run the analysis."""

    # Set the base directory
    base_dir = '/Users/davidguzman/Documents/GitHub/agent-tournament/outputs/2025'

    print("Loading tournament data...")
    results, all_pairs = load_tournament_data(base_dir)

    # Print summary statistics
    print_summary_statistics(results, all_pairs)

    # Create figure with subplots for each mechanism
    mechanisms = ['disarmament', 'mediation', 'repetition', 'reputationprisonersdilemma', 'nomechanism']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, mechanism in enumerate(mechanisms):
        if mechanism in results:
            create_mechanism_scatterplot(mechanism, results[mechanism], all_pairs, axes[i])
        else:
            axes[i].text(0.5, 0.5, f'No data for {mechanism}',
                        ha='center', va='center', fontsize=12)
            axes[i].set_title(mechanism.upper())

    # Hide the extra subplot if we have fewer than 6 mechanisms
    if len(mechanisms) < 6:
        axes[-1].axis('off')

    # Add main title
    fig.suptitle('Tournament Results: Cooperation Rate vs Average Payoff by Mechanism\n(10 Model Pairs per Mechanism)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save the figure
    output_path = '/Users/davidguzman/Documents/GitHub/agent-tournament/permanent_scratchpad/tournament_scatterplots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nScatterplots saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()