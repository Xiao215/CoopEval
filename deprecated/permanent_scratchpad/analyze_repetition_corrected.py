#!/usr/bin/env python3
"""
Corrected analysis for Repetition mechanism using the actual pairwise data.
"""

import matplotlib.pyplot as plt
import numpy as np

def create_corrected_repetition_plot():
    """Create scatterplot with correct Repetition data from the heatmap."""

    # Correct pairwise data from the heatmap
    # Format: (agent1, agent2, payoff, cooperation_rate)
    pairwise_data = [
        # deepseek-chat-v3.1 row
        ('deepseek-chat-v3.1', 'deepseek-chat-v3.1', 1.31, 0.30),
        ('deepseek-chat-v3.1', 'gpt-5-nano', 0.89, 0.10),
        ('deepseek-chat-v3.1', 'gpt-oss-20b', 0.97, 0.00),
        ('deepseek-chat-v3.1', 'qwen3-next-80b-a3b-instruct', 0.97, 0.00),

        # gpt-5-nano row (excluding duplicate with deepseek)
        ('gpt-5-nano', 'gpt-5-nano', 1.13, 0.00),
        ('gpt-5-nano', 'gpt-oss-20b', 0.97, 0.00),
        ('gpt-5-nano', 'qwen3-next-80b-a3b-instruct', 0.97, 0.00),
        ('gpt-5-nano', 'qwen3-next-80b-a3b-instruct', 1.20, 0.00),  # Second entry in matrix

        # gpt-oss-20b row (excluding duplicates)
        ('gpt-oss-20b', 'gpt-oss-20b', 0.97, 0.00),
        ('gpt-oss-20b', 'qwen3-next-80b-a3b-instruct', 0.97, 0.00),

        # qwen3-next-80b-a3b-instruct row (excluding duplicates)
        ('qwen3-next-80b-a3b-instruct', 'gpt-5-nano', 0.85, 0.10),
        ('qwen3-next-80b-a3b-instruct', 'qwen3-next-80b-a3b-instruct', 1.03, 0.08),
    ]

    # Remove duplicates and average where needed
    unique_pairs = {}
    for agent1, agent2, payoff, coop_rate in pairwise_data:
        key = tuple(sorted([agent1, agent2]))
        if key not in unique_pairs:
            unique_pairs[key] = {'payoffs': [], 'coop_rates': []}
        unique_pairs[key]['payoffs'].append(payoff)
        unique_pairs[key]['coop_rates'].append(coop_rate)

    # Average duplicates
    final_data = []
    for pair, values in unique_pairs.items():
        avg_payoff = np.mean(values['payoffs'])
        avg_coop = np.mean(values['coop_rates'])
        final_data.append((pair[0], pair[1], avg_payoff, avg_coop))

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for plotting
    x_values = []  # Cooperation rates
    y_values = []  # Payoffs
    labels = []
    colors = []
    sizes = []

    for agent1, agent2, payoff, coop_rate in final_data:
        x_values.append(coop_rate * 100)  # Convert to percentage
        y_values.append(payoff)

        # Shorten names for labels
        a1_short = agent1.replace('deepseek-chat-v3.1', 'deepseek')
        a1_short = a1_short.replace('qwen3-next-80b-a3b-instruct', 'qwen3')
        a1_short = a1_short.replace('gpt-oss-20b', 'gpt-oss')
        a1_short = a1_short.replace('gpt-5-nano', 'gpt-nano')

        a2_short = agent2.replace('deepseek-chat-v3.1', 'deepseek')
        a2_short = a2_short.replace('qwen3-next-80b-a3b-instruct', 'qwen3')
        a2_short = a2_short.replace('gpt-oss-20b', 'gpt-oss')
        a2_short = a2_short.replace('gpt-5-nano', 'gpt-nano')

        if agent1 == agent2:
            labels.append(f"{a1_short}\n(self)")
            colors.append('red')
            sizes.append(200)
        else:
            labels.append(f"{a1_short}\nvs\n{a2_short}")
            colors.append('blue')
            sizes.append(150)

    # Create scatter plot
    scatter = ax.scatter(x_values, y_values, c=colors, s=sizes, alpha=0.6,
                        edgecolors='black', linewidth=2)

    # Add labels with smart positioning to avoid overlap
    for i, label in enumerate(labels):
        offset_x = 5
        offset_y = 5

        # Special positioning for clustered points at 0% cooperation
        if x_values[i] == 0:
            # Stack labels vertically for points at x=0
            if y_values[i] < 0.98:
                offset_y = -20
            elif y_values[i] < 1.0:
                offset_y = 10
            elif y_values[i] < 1.15:
                offset_y = 20
            else:
                offset_y = -10

            # Alternate left/right for better spacing
            if i % 2 == 0:
                offset_x = -80
                ha = 'right'
            else:
                offset_x = 10
                ha = 'left'
        else:
            ha = 'center'

        ax.annotate(label, (x_values[i], y_values[i]),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=9, ha=ha, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.8))

    # Add trend line if there's variation
    if len(set(x_values)) > 1:
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, max(x_values) + 5, 100)
        ax.plot(x_line, p(x_line), "g--", alpha=0.5, linewidth=2,
               label=f'Trend: payoff = {z[0]:.4f}×coop + {z[1]:.3f}')

        # Calculate correlation
        correlation = np.corrcoef(x_values, y_values)[0, 1]
        # Place correlation text outside plot area
        ax.text(1.02, 0.3, f'Correlation: {correlation:.3f}',
               transform=ax.transAxes, fontsize=11,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # Formatting
    ax.set_xlabel('Cooperation Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Payoff', fontsize=14, fontweight='bold')
    ax.set_title('REPETITION MECHANISM - Corrected Pairwise Results\nCooperation vs Payoff',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set axis limits
    ax.set_xlim(-2, 35)  # Most data is between 0-30%
    ax.set_ylim(0.8, 1.4)

    # Calculate statistics
    avg_coop = np.mean(x_values)
    avg_payoff = np.mean(y_values)
    std_coop = np.std(x_values)
    std_payoff = np.std(y_values)

    # Add legend outside the plot area
    from matplotlib.patches import Patch
    red_patch = Patch(color='red', label='Self-play', alpha=0.6)
    blue_patch = Patch(color='blue', label='Cross-play', alpha=0.6)
    legend = ax.legend(handles=[red_patch, blue_patch], loc='upper left',
                      bbox_to_anchor=(1.02, 1), fontsize=11, title='Pair Type')

    # Add statistics box outside the plot area
    stats_text = (f'Statistics:\n'
                 f'n = {len(final_data)} pairs\n'
                 f'Avg Coop: {avg_coop:.1f}%\n'
                 f'(±{std_coop:.1f}%)\n'
                 f'Avg Payoff: {avg_payoff:.3f}\n'
                 f'(±{std_payoff:.3f})')

    # Place stats box to the right of the plot
    ax.text(1.02, 0.6, stats_text, transform=ax.transAxes,
           fontsize=10, ha='left', va='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Adjust layout to prevent overlap with external elements
    plt.subplots_adjust(right=0.82)

    # Save the figure
    output_path = '/Users/davidguzman/Documents/GitHub/agent-tournament/permanent_scratchpad/repetition_corrected_scatterplot.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Corrected plot saved to: {output_path}")

    plt.show()

    # Print the data table
    print("\nCORRECTED REPETITION DATA:")
    print("=" * 70)
    print(f"{'Pair':<45} {'Payoff':>10} {'Coop %':>10}")
    print("-" * 70)

    for agent1, agent2, payoff, coop_rate in sorted(final_data, key=lambda x: -x[2]):
        if agent1 == agent2:
            pair_str = f"{agent1} (self)"
        else:
            pair_str = f"{agent1} vs {agent2}"
        print(f"{pair_str:<45} {payoff:>10.3f} {coop_rate*100:>10.1f}%")

    print("-" * 70)
    print(f"{'AVERAGES':<45} {avg_payoff:>10.3f} {avg_coop:>10.1f}%")

    # Analysis summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    print("1. Very low cooperation overall - most pairs show 0% cooperation")
    print("2. deepseek-chat-v3.1 shows highest cooperation (30%) in self-play")
    print("3. deepseek is the only model showing any meaningful cooperation")
    print("4. Slight positive correlation between cooperation and payoff")
    print("5. Most games converge to mutual defection (Nash equilibrium)")
    print("6. Payoff range is narrow (0.85 - 1.31), indicating limited benefit from cooperation")


if __name__ == "__main__":
    create_corrected_repetition_plot()