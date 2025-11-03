#!/usr/bin/env python3
"""
Create a round trajectory plot for deepseek-chat-v3.1 self-play in Repetition mechanism.
Shows the evolution of cooperation/defection over rounds.
"""

import matplotlib.pyplot as plt
import numpy as np
import json

def create_deepseek_trajectory_plot():
    """Create trajectory plot showing deepseek's behavior over rounds."""

    # Load the payoff data
    payoff_file = '/Users/davidguzman/Documents/GitHub/agent-tournament/outputs/2025/09/16/17:07_repetition__openai-gpt-oss-20b-cot__deepseek-deepseek-chat-v3-1-cot__qwen-qwen3-next-80b-a3b-instruct__openai-gpt-5-nano-cot/payoffs.json'

    with open(payoff_file, 'r') as f:
        data = json.load(f)

    # Find deepseek self-play profile
    deepseek_profile = None
    for profile in data['profiles']:
        players = profile['players']
        if (len(players) == 2 and
            'deepseek/deepseek-chat-v3.1(CoT)' in players[0] and
            'deepseek/deepseek-chat-v3.1(CoT)' in players[1]):
            deepseek_profile = profile
            break

    if not deepseek_profile:
        print("Could not find deepseek self-play data")
        return

    # Extract round payoffs
    rounds = deepseek_profile['rounds']
    round_numbers = list(range(1, len(rounds) + 1))
    payoffs = [r[0] for r in rounds]  # Single value per round for self-play

    # Determine actions from payoffs
    # Payoff matrix: CC=2, CD=0, DC=3, DD=1
    # In self-play: CC=2, DD=1
    actions = []
    cooperation_rate = []
    cumulative_coop = 0

    for i, payoff in enumerate(payoffs):
        if payoff == 2.0:
            actions.append('C')  # Both cooperated
            cumulative_coop += 1
        elif payoff == 1.0:
            actions.append('D')  # Both defected
        else:
            actions.append('?')  # Unexpected payoff

        cooperation_rate.append(cumulative_coop / (i + 1) * 100)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Color scheme
    color_cooperate = '#2E7D32'  # Green
    color_defect = '#C62828'  # Red
    color_mixed = '#FF9800'  # Orange

    # Subplot 1: Round-by-round payoffs
    ax1 = axes[0]
    bars = ax1.bar(round_numbers, payoffs, width=0.6, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Color bars based on action
    for i, (bar, action) in enumerate(zip(bars, actions)):
        if action == 'C':
            bar.set_facecolor(color_cooperate)
        elif action == 'D':
            bar.set_facecolor(color_defect)
        else:
            bar.set_facecolor(color_mixed)

    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Payoff', fontsize=12)
    ax1.set_title('Round-by-Round Payoffs', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 2.5)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(round_numbers)

    # Add horizontal lines for reference
    ax1.axhline(y=2, color=color_cooperate, linestyle='--', alpha=0.5, label='Mutual Cooperation (2)')
    ax1.axhline(y=1, color=color_defect, linestyle='--', alpha=0.5, label='Mutual Defection (1)')
    ax1.legend(loc='upper right')

    # Add action labels
    for i, (x, y, action) in enumerate(zip(round_numbers, payoffs, actions)):
        ax1.text(x, y + 0.1, f'{action}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Subplot 2: Cumulative average payoff
    ax2 = axes[1]
    cumulative_avg = np.cumsum(payoffs) / np.arange(1, len(payoffs) + 1)
    ax2.plot(round_numbers, cumulative_avg, marker='o', markersize=8, linewidth=2,
             color='#1565C0', label='Cumulative Average')
    ax2.fill_between(round_numbers, cumulative_avg, alpha=0.3, color='#1565C0')

    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Cumulative Avg Payoff', fontsize=12)
    ax2.set_title('Cumulative Average Payoff Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(round_numbers)
    ax2.set_ylim(0.8, 2.2)

    # Add final average
    final_avg = cumulative_avg[-1]
    ax2.axhline(y=final_avg, color='orange', linestyle=':', alpha=0.7)
    ax2.text(len(rounds) + 0.1, final_avg, f'Final: {final_avg:.3f}',
             fontsize=10, va='center', fontweight='bold')

    # Subplot 3: Cooperation rate over time
    ax3 = axes[2]
    ax3.plot(round_numbers, cooperation_rate, marker='s', markersize=8, linewidth=2,
             color=color_cooperate, label='Cooperation Rate')
    ax3.fill_between(round_numbers, cooperation_rate, alpha=0.3, color=color_cooperate)

    ax3.set_xlabel('Round', fontsize=12)
    ax3.set_ylabel('Cooperation Rate (%)', fontsize=12)
    ax3.set_title('Cooperation Rate Evolution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(round_numbers)
    ax3.set_ylim(0, 105)

    # Add 50% reference line
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')

    # Add final cooperation rate
    final_coop = cooperation_rate[-1]
    ax3.text(len(rounds) + 0.1, final_coop, f'Final: {final_coop:.1f}%',
             fontsize=10, va='center', fontweight='bold', color=color_cooperate)

    # Overall title
    fig.suptitle('Deepseek-Chat-v3.1 Self-Play Trajectory - Repetition Mechanism\n10 Rounds of Prisoner\'s Dilemma',
                fontsize=16, fontweight='bold', y=1.02)

    # Add text box with summary statistics
    summary_text = (
        f'Summary Statistics:\n'
        f'Total Rounds: {len(rounds)}\n'
        f'Cooperations: {actions.count("C")}\n'
        f'Defections: {actions.count("D")}\n'
        f'Cooperation Rate: {final_coop:.1f}%\n'
        f'Avg Payoff: {final_avg:.3f}\n'
        f'Strategy Shift: Round 6'
    )

    # Place summary box
    fig.text(0.98, 0.5, summary_text, transform=fig.transFigure,
            fontsize=10, ha='right', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, hspace=0.3)

    # Save figure
    output_path = '/Users/davidguzman/Documents/GitHub/agent-tournament/permanent_scratchpad/deepseek_trajectory_repetition.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Trajectory plot saved to: {output_path}")

    plt.show()

    # Print analysis
    print("\n" + "=" * 70)
    print("DEEPSEEK SELF-PLAY ANALYSIS - REPETITION MECHANISM")
    print("=" * 70)

    print(f"\nRound-by-round breakdown:")
    print(f"{'Round':<8} {'Payoff':<10} {'Action':<10} {'Cumul Avg':<12} {'Coop Rate':<10}")
    print("-" * 60)

    for i, (rnd, pay, act, avg, coop) in enumerate(zip(round_numbers, payoffs, actions, cumulative_avg, cooperation_rate)):
        print(f"{rnd:<8} {pay:<10.1f} {act:<10} {avg:<12.3f} {coop:<10.1f}%")

    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS:")
    print("=" * 70)
    print("1. Clear strategy shift at round 6: from defection to cooperation")
    print("2. First 5 rounds: mutual defection (DD)")
    print("3. Last 5 rounds: mutual cooperation (CC)")
    print("4. This pattern suggests a tit-for-tat or trigger strategy")
    print("5. Final cooperation rate: 50% (5 C, 5 D)")
    print("6. Payoff improved from 1.0 (defection) to 2.0 (cooperation)")
    print("7. Demonstrates learning or strategic adaptation mid-game")


if __name__ == "__main__":
    create_deepseek_trajectory_plot()