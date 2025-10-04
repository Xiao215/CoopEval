#!/bin/bash
set -euo pipefail

# Activate venv
source .venv312/bin/activate
export PYTHONPATH=.

# Resolve directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Manually specified YAMLs
YAMLS=(
    # None
)

# Config folders (relative to script dir)
YAML_DIRS=(
  "test/repetition"
)

# Expand dirs relative to SCRIPT_DIR
for dir in "${YAML_DIRS[@]}"; do
  full_dir="$SCRIPT_DIR/configs/$dir"
  if [[ -d "$full_dir" ]]; then
    while IFS= read -r -d '' file; do
      YAMLS+=("$file")
    done < <(find "$full_dir" -type f -name "*.yaml" -print0 | sort -z)
  fi
done

# Run through all configs
for config in "${YAMLS[@]}"; do
  echo "🚀 Running with config: $config"

  if ! python3 "$SCRIPT_DIR/script/run_evolution.py" --config "$config"; then
    echo "❌ ERROR: Execution failed for $config"
    # Uncomment this to stop immediately
    # exit 1
  else
    echo "✅ Success: $config"
  fi
done