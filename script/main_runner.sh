#!/bin/bash
# Main runner script to execute run_experiment.py for all combinations of games and mechanisms
# Usage: ./main_runner.sh

# Exit on error
set -e

# =============================================================================
# CONFIGURATION - Edit these variables to customize your experiments
# =============================================================================

# Agents configuration (relative to configs/)
AGENTS_CONFIG="agents/default_agents.yaml"

# Evaluation configuration (relative to configs/)
EVALUATION_CONFIG="evaluation/default_evaluation.yaml"

# Concurrency setting (number of parallel workers)
CONCURRENCY=1

# List of game config paths (relative to configs/)
# Based on games in src/games/
GAME_CONFIGS=(
    "games/matching_pennies.yaml"
    "games/prisoners_dilemma.yaml"
    "games/public_goods.yaml"
    "games/stag_hunt.yaml"
    "games/travellers_dilemma.yaml"
    "games/trust_game.yaml"
)

# List of mechanism config paths (relative to configs/)
# Based on mechanisms in src/mechanisms/
MECHANISM_CONFIGS=(
    "mechanisms/no_mechanism.yaml"
    "mechanisms/contracting.yaml"
    "mechanisms/disarmament.yaml"
    "mechanisms/mediation.yaml"
    "mechanisms/repetition.yaml"
    "mechanisms/reputation.yaml"
)

# =============================================================================
# END CONFIGURATION
# =============================================================================

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Activate conda environment if needed (uncomment and modify as needed)
# source ~/anaconda3/bin/activate llmcoop

# Counter for tracking experiments
total_experiments=$((${#GAME_CONFIGS[@]} * ${#MECHANISM_CONFIGS[@]}))
current=0

echo "=================================================="
echo "Running $total_experiments experiments"
echo "Games: ${#GAME_CONFIGS[@]}, Mechanisms: ${#MECHANISM_CONFIGS[@]}"
echo "=================================================="
echo ""

# Loop through all combinations
for game in "${GAME_CONFIGS[@]}"; do
    for mechanism in "${MECHANISM_CONFIGS[@]}"; do
        current=$((current + 1))
        
        echo "[$current/$total_experiments] Running: Game=$game, Mechanism=$mechanism"
        echo "--------------------------------------------------"
        
        # Create a temporary main config file
        TEMP_CONFIG=$(mktemp "${PROJECT_ROOT}/configs/main/.temp_config_XXXXXX.yaml")
        
        # Generate the main config content
        cat > "$TEMP_CONFIG" << EOF
# Auto-generated config for game-mechanism combination
game_config: $game
mechanism_config: $mechanism
agents_config: $AGENTS_CONFIG
evaluation_config: $EVALUATION_CONFIG
name: $(basename "$game" .yaml)_$(basename "$mechanism" .yaml)
concurrency: $CONCURRENCY
EOF
        
        # Run the experiment
        python script/run_experiment.py --config "$(basename "$TEMP_CONFIG")"
        
        # Check if successful
        if [ $? -eq 0 ]; then
            echo "✓ Completed successfully"
        else
            echo "✗ Failed"
        fi
        
        # Clean up temp config
        rm -f "$TEMP_CONFIG"
        
        echo ""
    done
done

echo "=================================================="
echo "All experiments completed!"
echo "=================================================="
