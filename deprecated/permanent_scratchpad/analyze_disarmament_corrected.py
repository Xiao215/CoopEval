#!/usr/bin/env python3
"""
Corrected analysis for Disarmament mechanism using the actual pairwise data.
"""

import matplotlib.pyplot as plt
import numpy as np

def create_corrected_disarmament_plot():
    """Create scatterplot with correct Disarmament data from the heatmap."""

    # Correct pairwise data from the heatmap
    # Format: (agent1, agent2, payoff, cooperation_rate)
    pairwise_data = [
        # deepseek-chat-v3.1 row
        ('deepseek-chat-v3.1', 'deepseek-chat-v3.1', 1.91, 1.00),
        ('deepseek-chat-v3.1', 'gpt-5-nano', 2.36, 1.00),
        ('deepseek-chat-v3.1', 'gpt-oss-20b', 0.00, 1.00),  # Payoff 0 but coop 1.0
        ('deepseek-chat-v3.1', 'qwen3-next-80b-a3b-instruct', 1.91, 1.00),

        # gpt-5-nano row (excluding duplicate with deepseek)
        ('gpt-5-nano', 'gpt-5-nano', 1.00, 1.00),
        ('gpt-5-nano', 'gpt-oss-20b', 0.95, 0.14),
        ('gpt-5-nano', 'qwen3-next-80b-a3b-instruct', 0.95, 0.00),

        # gpt-oss-20b row (excluding duplicates)
        ('gpt-oss-20b', 'gpt-oss-20b', 0.95, 0.00),
        ('gpt-oss-20b', 'qwen3-next-80b-a3b-instruct', 0.95, 0.14),

        # qwen3-next-80b-a3b-instruct self-play
        ('qwen3-next-80b-a3b-instruct', 'qwen3-next-80b-a3b-instruct', 0.95, 0.43),
    ]

    # Note: Using upper triangle of symmetric matrix to avoid duplicates
    # The actual unique pairs are 10 (4 self-play + 6 cross-play)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for plotting
    x_values = []  # Cooperation rates
    y_values = []  # Payoffs
    labels = []
    colors = []
    sizes = []

    for agent1, agent2, payoff, coop_rate in pairwise_data:
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

    # Add labels with smart positioning
    for i, label in enumerate(labels):
        # Adjust positions to avoid overlap
        offset_x = 5
        offset_y = 5

        # Special positioning for overlapping points
        if x_values[i] == 100 and y_values[i] == 1.91:  # deepseek pairs at (100, 1.91)
            if 'deepseek\n(self)' in label:
                offset_x = -80
                offset_y = 0
            else:
                offset_x = 10
                offset_y = 10

        elif x_values[i] == 0 and abs(y_values[i] - 0.95) < 0.01:  # Points at (0, 0.95)
            if 'gpt-oss\n(self)' in label:
                offset_y = -20
            else:
                offset_y = 20

        elif abs(x_values[i] - 14) < 1 and abs(y_values[i] - 0.95) < 0.01:  # Points at (14, 0.95)
            if 'gpt-nano' in label:
                offset_y = 20
            else:
                offset_y = -20

        ax.annotate(label, (x_values[i], y_values[i]),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=9, ha='center', alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.8))

    # Add trend line
    if len(set(x_values)) > 1:
        # Remove outlier (deepseek vs gpt-oss with 0 payoff) for trend calculation
        x_trend = [x for x, y in zip(x_values, y_values) if y > 0.1]
        y_trend = [y for x, y in zip(x_values, y_values) if y > 0.1]

        if len(x_trend) > 2:
            z = np.polyfit(x_trend, y_trend, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, 100, 100)
            ax.plot(x_line, p(x_line), "g--", alpha=0.5, linewidth=2,
                   label=f'Trend: payoff = {z[0]:.4f}×coop + {z[1]:.3f}')

            # Calculate correlation
            correlation = np.corrcoef(x_trend, y_trend)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # Formatting
    ax.set_xlabel('Cooperation Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Payoff', fontsize=14, fontweight='bold')
    ax.set_title('DISARMAMENT MECHANISM - Corrected Pairwise Results\nCooperation vs Payoff',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set axis limits
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.2, 2.5)

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
    stats_text = (f'Statistics (n=10 pairs):\n'
                 f'Avg Cooperation: {avg_coop:.1f}%\n'
                 f'(±{std_coop:.1f}%)\n'
                 f'Avg Payoff: {avg_payoff:.3f}\n'
                 f'(±{std_payoff:.3f})')

    # Place stats box to the right of the plot
    ax.text(1.02, 0.5, stats_text, transform=ax.transAxes,
           fontsize=10, ha='left', va='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Adjust layout to prevent overlap with external elements
    plt.subplots_adjust(right=0.82)

    # Save the figure
    output_path = '/Users/davidguzman/Documents/GitHub/agent-tournament/permanent_scratchpad/disarmament_corrected_scatterplot.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Corrected plot saved to: {output_path}")

    plt.show()

    # Print the data table
    print("\nCORRECTED DISARMAMENT DATA:")
    print("=" * 70)
    print(f"{'Pair':<45} {'Payoff':>10} {'Coop %':>10}")
    print("-" * 70)

    for agent1, agent2, payoff, coop_rate in pairwise_data:
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
    print("1. deepseek-chat-v3.1 shows consistent 100% cooperation with most partners")
    print("2. Strong positive correlation between cooperation and payoff (excluding anomaly)")
    print("3. Self-play results vary widely: deepseek (1.91), gpt-nano (1.00), others (0.95)")
    print("4. ANOMALY: deepseek vs gpt-oss shows 100% cooperation but 0 payoff!")
    print("5. qwen3 shows moderate cooperation (43%) in self-play")
    print("6. gpt-oss shows minimal cooperation across all pairings")


if __name__ == "__main__":
    create_corrected_disarmament_plot()