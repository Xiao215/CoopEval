#!/usr/bin/env python3
"""
Create individual scatterplots for each mechanism showing cooperation vs payoff.
"""

import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_and_process_data(base_dir):
    """Load tournament data and calculate cooperation rates and payoffs per pair."""

    model_names = ['gpt-oss-20b', 'deepseek-chat-v3.1', 'qwen3-next-80b-a3b-instruct', 'gpt-5-nano']

    # Store results by mechanism
    results = defaultdict(lambda: defaultdict(lambda: {'payoff': None, 'coop_rate': None}))

    for root, dirs, files in os.walk(base_dir):
        if 'payoffs.json' not in files:
            continue

        dir_name = os.path.basename(root)
        mechanism = dir_name.split('_')[1] if '_' in dir_name else 'unknown'

        # Load payoff data
        with open(os.path.join(root, 'payoffs.json'), 'r') as f:
            payoff_data = json.load(f)

        # Find JSONL file
        jsonl_file = None
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_file = os.path.join(root, file)
                break

        # Process payoffs
        for profile in payoff_data['profiles']:
            players = []
            for p in profile['players']:
                name = p.split('/')[1].split('(')[0] if '/' in p else p
                players.append(name)

            pair = tuple(sorted(players))
            payoffs = list(profile['discounted_average'].values())
            avg_payoff = sum(payoffs) / len(payoffs) if payoffs else 0
            results[mechanism][pair]['payoff'] = avg_payoff

        # Process cooperation data
        if jsonl_file:
            pair_actions = defaultdict(lambda: defaultdict(lambda: {'C': 0, 'D': 0}))

            with open(jsonl_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
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

                        round_moves = []
                        for move in moves:
                            if isinstance(move, dict) and 'name' in move and 'action' in move:
                                player = move['name'].split('/')[1].split('(')[0]
                                action = move['action']
                                if action in ['C', 'D']:
                                    round_moves.append((player, action))

                        # Process pairs
                        if len(round_moves) >= 2:
                            for i in range(0, len(round_moves) - 1, 2):
                                if i + 1 < len(round_moves):
                                    p1, a1 = round_moves[i]
                                    p2, a2 = round_moves[i + 1]
                                    pair = tuple(sorted([p1, p2]))

                                    pair_actions[pair][p1][a1] += 1
                                    pair_actions[pair][p2][a2] += 1
                    except:
                        continue

            # Calculate cooperation rates
            for pair, player_actions in pair_actions.items():
                total_c = sum(player_actions[p]['C'] for p in player_actions)
                total_d = sum(player_actions[p]['D'] for p in player_actions)
                total_actions = total_c + total_d

                if total_actions > 0:
                    coop_rate = (total_c / total_actions) * 100
                    if pair in results[mechanism]:
                        results[mechanism][pair]['coop_rate'] = coop_rate

    return results


def create_mechanism_plot(mechanism_name, data, output_dir):
    """Create individual plot for a mechanism."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Prepare data points
    points = []
    for pair, values in data.items():
        if values['payoff'] is not None:
            coop = values['coop_rate'] if values['coop_rate'] is not None else 0
            points.append({
                'pair': pair,
                'coop': coop,
                'payoff': values['payoff'],
                'is_self': pair[0] == pair[1]
            })

    if not points:
        ax.text(0.5, 0.5, f'No data for {mechanism_name}', ha='center', va='center', fontsize=14)
        ax.set_title(mechanism_name.upper())
        plt.savefig(f'{output_dir}/{mechanism_name}_scatterplot.png', dpi=200, bbox_inches='tight')
        plt.close()
        return

    # Sort for consistent ordering
    points = sorted(points, key=lambda x: (x['coop'], x['payoff']))

    # Colors
    colors = ['red' if p['is_self'] else 'blue' for p in points]

    # Create scatter plot
    x_values = [p['coop'] for p in points]
    y_values = [p['payoff'] for p in points]

    scatter = ax.scatter(x_values, y_values, c=colors, s=150, alpha=0.7,
                        edgecolors='black', linewidth=1.5)

    # Add labels
    for i, point in enumerate(points):
        # Shorten names
        p1 = point['pair'][0].replace('qwen3-next-80b-a3b-instruct', 'qwen3')
        p1 = p1.replace('deepseek-chat-v3.1', 'deepseek')
        p1 = p1.replace('gpt-oss-20b', 'gpt-oss')
        p1 = p1.replace('gpt-5-nano', 'gpt-nano')

        p2 = point['pair'][1].replace('qwen3-next-80b-a3b-instruct', 'qwen3')
        p2 = p2.replace('deepseek-chat-v3.1', 'deepseek')
        p2 = p2.replace('gpt-oss-20b', 'gpt-oss')
        p2 = p2.replace('gpt-5-nano', 'gpt-nano')

        if point['is_self']:
            label = f"{p1}\n(self)"
        else:
            label = f"{p1}\nvs\n{p2}"

        # Adjust label position to avoid overlap
        offset_x = 5
        offset_y = 5
        if i > 0 and abs(x_values[i] - x_values[i-1]) < 5:
            offset_y = -15 if i % 2 == 0 else 15

        ax.annotate(label, (x_values[i], y_values[i]),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=9, ha='left', alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Add trend line if there's variation
    if len(set(x_values)) > 1:
        try:
            z = np.polyfit(x_values, y_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(x_values), max(x_values), 100)
            ax.plot(x_trend, p(x_trend), "g--", alpha=0.5, linewidth=2,
                   label=f'Trend: payoff = {z[0]:.4f}×coop + {z[1]:.3f}')

            # Calculate correlation
            correlation = np.corrcoef(x_values, y_values)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        except:
            pass

    # Formatting
    ax.set_xlabel('Cooperation Rate (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Payoff', fontsize=14, fontweight='bold')
    ax.set_title(f'{mechanism_name.upper()} - Cooperation vs Payoff', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set axis limits with padding
    x_padding = 5
    y_padding = 0.1

    ax.set_xlim(-x_padding, max(100, max(x_values) + x_padding))
    ax.set_ylim(min(y_values) - y_padding, max(y_values) + y_padding)

    # Add legend
    red_patch = plt.scatter([], [], c='red', s=100, label='Self-play', alpha=0.7)
    blue_patch = plt.scatter([], [], c='blue', s=100, label='Cross-play', alpha=0.7)
    ax.legend(handles=[red_patch, blue_patch], loc='lower right', fontsize=11)

    # Statistics box
    avg_coop = np.mean(x_values)
    avg_payoff = np.mean(y_values)
    stats_text = f'Avg Cooperation: {avg_coop:.1f}%\nAvg Payoff: {avg_payoff:.3f}'
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
           fontsize=10, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    output_path = f'{output_dir}/{mechanism_name}_scatterplot.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"Created plot: {output_path}")


def main():
    """Main function."""

    base_dir = '/Users/davidguzman/Documents/GitHub/agent-tournament/outputs/2025'
    output_dir = '/Users/davidguzman/Documents/GitHub/agent-tournament/permanent_scratchpad'

    print("Loading tournament data...")
    results = load_and_process_data(base_dir)

    mechanisms = ['disarmament', 'mediation', 'repetition', 'reputationprisonersdilemma', 'nomechanism']

    print("\nCreating individual mechanism plots...")
    for mechanism in mechanisms:
        if mechanism in results:
            create_mechanism_plot(mechanism, results[mechanism], output_dir)

    print("\nAll plots created successfully!")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)

    for mechanism in mechanisms:
        if mechanism in results:
            data = results[mechanism]
            payoffs = [v['payoff'] for v in data.values() if v['payoff'] is not None]
            coop_rates = [v['coop_rate'] if v['coop_rate'] is not None else 0
                         for v in data.values() if v['payoff'] is not None]

            if payoffs:
                print(f"\n{mechanism.upper()}:")
                print(f"  Pairs analyzed: {len(payoffs)}")
                print(f"  Avg Payoff: {np.mean(payoffs):.3f} (±{np.std(payoffs):.3f})")
                print(f"  Avg Cooperation: {np.mean(coop_rates):.1f}% (±{np.std(coop_rates):.1f}%)")

                # Find best performing pair
                best_pair = max(data.items(), key=lambda x: x[1]['payoff'] if x[1]['payoff'] else 0)
                print(f"  Best pair: {best_pair[0]} with payoff {best_pair[1]['payoff']:.3f}")


if __name__ == "__main__":
    main()