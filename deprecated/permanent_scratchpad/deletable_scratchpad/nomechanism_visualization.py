import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Data from the NoMechanism experiment
models = [
    'GPT-OSS-20B',
    'DeepSeek-Chat-v3.1',
    'Qwen3-Next-80B',
    'GPT-5-Nano'
]

payoffs = [0.4375, 0.4375, 0.4375, 0.4375]
cooperation_rates = [0.0, 0.0, 0.0, 0.0]

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Prisoners\' Dilemma Performance - No Mechanism', fontsize=16, fontweight='bold')

# 1. Payoffs bar chart
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars1 = ax1.bar(models, payoffs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax1.set_title('Expected Payoffs by Model', fontweight='bold')
ax1.set_ylabel('Expected Payoff')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)
for bar, payoff in zip(bars1, payoffs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{payoff:.4f}', ha='center', va='bottom', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# 2. Cooperation rates (all zero)
bars2 = ax2.bar(models, cooperation_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax2.set_title('Cooperation Rates by Model', fontweight='bold')
ax2.set_ylabel('Cooperation Rate')
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, 0.05,
             '0%', ha='center', va='bottom', fontweight='bold', color='red')
ax2.tick_params(axis='x', rotation=45)

# 3. Payoff matrix visualization
payoff_matrix = np.array([[2, 0], [3, 1]])
im = ax3.imshow(payoff_matrix, cmap='RdYlBu_r', aspect='equal')
ax3.set_title('Prisoners\' Dilemma Payoff Matrix', fontweight='bold')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Cooperate', 'Defect'])
ax3.set_yticklabels(['Cooperate', 'Defect'])
ax3.set_xlabel('Opponent Action')
ax3.set_ylabel('Player Action')

# Add text annotations to the heatmap
for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, payoff_matrix[i, j], ha="center", va="center",
                       color="white", fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label('Payoff', rotation=270, labelpad=15)

# 4. Game theory explanation
ax4.axis('off')
explanation_text = """
Game Theory Analysis:

• All models achieved Nash Equilibrium
• Mutual defection (DD) = 1.0 payoff each
• Expected payoff = 0.4375 (weighted average)
• Zero cooperation demonstrates rational
  but suboptimal behavior

• Pareto optimal: Mutual cooperation (CC)
  would yield 2.0 payoff each
• Improvement potential: +1.5625 per agent
• This baseline shows need for cooperation
  mechanisms
"""

ax4.text(0.05, 0.95, explanation_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/davidguzman/Documents/GitHub/agent-tournament/deletable_scratchpad/nomechanism_results.png',
            dpi=300, bbox_inches='tight')
plt.show()

# Create a summary table
summary_df = pd.DataFrame({
    'Model': models,
    'Expected Payoff': payoffs,
    'Cooperation Rate': cooperation_rates,
    'Strategy': ['Defect'] * 4,
    'Outcome': ['Nash Equilibrium'] * 4
})

print("\n" + "="*60)
print("PRISONERS' DILEMMA - NO MECHANISM RESULTS")
print("="*60)
print(summary_df.to_string(index=False))
print("="*60)
print(f"Average Expected Payoff: {np.mean(payoffs):.4f}")
print(f"Potential with Cooperation: 2.0000 (+{2.0 - np.mean(payoffs):.4f})")
print("="*60)