#!/usr/bin/env python3
"""
Create a plot showing the number of rounds for each pair in the Disarmament mechanism.
Fewer rounds indicate faster agreement on caps.
"""

import matplotlib.pyplot as plt
import numpy as np
import json

def create_disarmament_rounds_plot():
    """Create bar plot showing rounds per pair in Disarmament."""

    # Load the payoff data
    payoff_file = '/Users/davidguzman/Documents/GitHub/agent-tournament/outputs/2025/09/17/00:15_disarmament__openai-gpt-oss-20b-cot__deepseek-deepseek-chat-v3-1-cot__qwen-qwen3-next-80b-a3b-instruct__openai-gpt-5-nano-cot/payoffs.json'

    with open(payoff_file, 'r') as f:
        data = json.load(f)

    # Extract data for each pair
    pair_data = []

    for profile in data['profiles']:
        players = profile['players']

        # Simplify player names
        simplified_players = []
        for p in players:
            if 'deepseek' in p:
                simplified_players.append('deepseek')
            elif 'gpt-oss' in p:
                simplified_players.append('gpt-oss')
            elif 'gpt-5-nano' in p:
                simplified_players.append('gpt-nano')
            elif 'qwen' in p:
                simplified_players.append('qwen3')
            else:
                simplified_players.append(p)

        # Create pair label
        if len(simplified_players) == 1:
            pair_label = f"{simplified_players[0]}\n(self)"
        else:
            pair_label = f"{simplified_players[0]}\nvs\n{simplified_players[1]}"

        # Count rounds
        num_rounds = len(profile['rounds'])

        # Get average payoff
        avg_payoffs = list(profile['discounted_average'].values())
        avg_payoff = np.mean(avg_payoffs) if avg_payoffs else 0

        pair_data.append({
            'label': pair_label,
            'rounds': num_rounds,
            'payoff': avg_payoff,
            'is_self': len(simplified_players) == 1 or simplified_players[0] == simplified_players[1]
        })

    # Sort by number of rounds
    pair_data.sort(key=lambda x: x['rounds'])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Prepare data for plotting
    labels = [d['label'] for d in pair_data]
    rounds = [d['rounds'] for d in pair_data]
    payoffs = [d['payoff'] for d in pair_data]
    colors = ['red' if d['is_self'] else 'blue' for d in pair_data]

    # Subplot 1: Number of rounds bar chart
    bars1 = ax1.bar(range(len(labels)), rounds, color=colors, alpha=0.6,
                    edgecolor='black', linewidth=1.5)

    # Color code by performance
    for bar, num_rounds in zip(bars1, rounds):
        if num_rounds == 1:
            bar.set_alpha(1.0)  # Full opacity for quick agreement
        elif num_rounds == 2:
            bar.set_alpha(0.8)
        elif num_rounds == 3:
            bar.set_alpha(0.6)
        else:
            bar.set_alpha(0.4)  # Lower opacity for many rounds

    ax1.set_xlabel('Player Pair', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Rounds', fontsize=12, fontweight='bold')
    ax1.set_title('Number of Negotiation Rounds per Pair - Disarmament Mechanism\n(Fewer rounds = Faster agreement on caps)',
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9, ha='center')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, rounds)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add average line
    avg_rounds = np.mean(rounds)
    ax1.axhline(y=avg_rounds, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax1.text(len(labels) - 0.5, avg_rounds + 0.1, f'Avg: {avg_rounds:.1f}',
            fontsize=10, color='green', fontweight='bold')

    # Subplot 2: Relationship between rounds and payoff
    ax2.scatter(rounds, payoffs, c=colors, s=150, alpha=0.6, edgecolor='black', linewidth=1.5)

    # Add labels for each point
    for i, (x, y, label) in enumerate(zip(rounds, payoffs, labels)):
        # Clean up label for scatter plot (remove newlines)
        clean_label = label.replace('\n', ' ')

        # Position labels to avoid overlap
        offset_y = 0.05
        if i > 0 and abs(rounds[i] - rounds[i-1]) < 0.5:
            offset_y = -0.08 if i % 2 == 0 else 0.08

        ax2.annotate(clean_label, (x, y),
                    xytext=(0, offset_y), textcoords='offset points',
                    fontsize=8, ha='center', va='bottom' if offset_y > 0 else 'top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='gray', alpha=0.7))

    # Add trend line
    if len(set(rounds)) > 1:
        z = np.polyfit(rounds, payoffs, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(rounds), max(rounds), 100)
        ax2.plot(x_trend, p(x_trend), "g--", alpha=0.5, linewidth=2)

        # Calculate correlation
        correlation = np.corrcoef(rounds, payoffs)[0, 1]
        ax2.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax2.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    ax2.set_xlabel('Number of Rounds', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Payoff', fontsize=12, fontweight='bold')
    ax2.set_title('Relationship: Negotiation Rounds vs Final Payoff',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, max(rounds) + 0.5)

    # Add legend
    from matplotlib.patches import Patch
    red_patch = Patch(color='red', label='Self-play', alpha=0.6)
    blue_patch = Patch(color='blue', label='Cross-play', alpha=0.6)
    ax1.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=10)
    ax2.legend(handles=[red_patch, blue_patch], loc='upper right', fontsize=10)

    # Add summary statistics box
    stats_text = (
        f'Summary Statistics:\n'
        f'Total Pairs: {len(pair_data)}\n'
        f'Avg Rounds: {avg_rounds:.1f}\n'
        f'Min Rounds: {min(rounds)}\n'
        f'Max Rounds: {max(rounds)}\n'
        f'1-round agreements: {rounds.count(1)}\n'
        f'2-round agreements: {rounds.count(2)}\n'
        f'3-round agreements: {rounds.count(3)}'
    )

    fig.text(0.98, 0.5, stats_text, transform=fig.transFigure,
            fontsize=10, ha='right', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, hspace=0.3)

    # Save figure
    output_path = '/Users/davidguzman/Documents/GitHub/agent-tournament/permanent_scratchpad/disarmament_rounds_analysis.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # plt.show()  # Commented out to prevent blocking

    # Print detailed analysis
    print("\n" + "=" * 70)
    print("DISARMAMENT ROUNDS ANALYSIS")
    print("=" * 70)

    print(f"\n{'Pair':<40} {'Rounds':>10} {'Payoff':>10}")
    print("-" * 60)

    for d in pair_data:
        clean_label = d['label'].replace('\n', ' ')
        print(f"{clean_label:<40} {d['rounds']:>10} {d['payoff']:>10.3f}")

    print("-" * 60)
    print(f"{'AVERAGES':<40} {avg_rounds:>10.1f} {np.mean(payoffs):>10.3f}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("1. Quick agreement (1 round) often correlates with good outcomes")
    print("2. Self-play pairs show varied negotiation speeds")
    print("3. Some pairs needed 3 rounds to reach agreement")
    print("4. Average negotiation length: {:.1f} rounds".format(avg_rounds))

    # Analysis by rounds
    print("\nGROUPED BY ROUNDS:")
    for n in sorted(set(rounds)):
        pairs_with_n = [d for d in pair_data if d['rounds'] == n]
        avg_payoff_n = np.mean([d['payoff'] for d in pairs_with_n])
        print(f"  {n} round(s): {len(pairs_with_n)} pairs, avg payoff: {avg_payoff_n:.3f}")


if __name__ == "__main__":
    create_disarmament_rounds_plot()