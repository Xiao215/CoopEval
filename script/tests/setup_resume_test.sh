#!/bin/bash
# Setup script to prepare an existing successful batch for resume testing
# Usage: ./setup_resume_test.sh <batch_dir>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <batch_dir>"
    echo "Example: $0 outputs/test_resume_20260120_024307"
    exit 1
fi

BATCH_DIR=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "============================================================"
echo "SETUP RESUME TEST FROM EXISTING BATCH"
echo "============================================================"
echo ""

# Validate batch directory
if [ ! -d "$BATCH_DIR" ]; then
    echo -e "${RED}ERROR: Batch directory not found: $BATCH_DIR${NC}"
    exit 1
fi

if [ ! -f "$BATCH_DIR/batch_summary.json" ]; then
    echo -e "${RED}ERROR: batch_summary.json not found in $BATCH_DIR${NC}"
    exit 1
fi

echo -e "${BLUE}Modifying batch: $BATCH_DIR${NC}"
echo ""

# Create backup
BACKUP_FILE="$BATCH_DIR/batch_summary.json.backup"
if [ ! -f "$BACKUP_FILE" ]; then
    cp "$BATCH_DIR/batch_summary.json" "$BACKUP_FILE"
    echo -e "${GREEN}✓ Created backup: batch_summary.json.backup${NC}"
else
    echo -e "${YELLOW}⚠ Backup already exists, skipping${NC}"
fi

echo ""
echo "Modifying experiment states..."
echo ""

# Use Python to modify batch_summary.json
python3 << PYEOF
import json

batch_dir = '${BATCH_DIR}'
summary_file = f"{batch_dir}/batch_summary.json"

# Load current summary
with open(summary_file, 'r') as f:
    summary = json.load(f)

experiments = summary['experiments']

# 1. Keep no_mechanism_prisoners_dilemma as SUCCESS (no changes)
print("✓ no_mechanism_prisoners_dilemma: SUCCESS (unchanged)")

# 2. Change no_mechanism_public_goods to FAILED
if 'no_mechanism_public_goods' in experiments:
    experiments['no_mechanism_public_goods']['status'] = 'failed'
    experiments['no_mechanism_public_goods']['exit_code'] = 1
    print("✗ no_mechanism_public_goods: Changed to FAILED")

# 3. Change no_mechanism_trust_game to IN_PROGRESS (crashed)
if 'no_mechanism_trust_game' in experiments:
    # Remove end_time and duration to simulate crash
    experiments['no_mechanism_trust_game']['status'] = 'in_progress'
    if 'end_time' in experiments['no_mechanism_trust_game']:
        del experiments['no_mechanism_trust_game']['end_time']
    if 'duration_seconds' in experiments['no_mechanism_trust_game']:
        del experiments['no_mechanism_trust_game']['duration_seconds']
    if 'exit_code' in experiments['no_mechanism_trust_game']:
        del experiments['no_mechanism_trust_game']['exit_code']
    print("⋯ no_mechanism_trust_game: Changed to IN_PROGRESS (simulated crash)")

# 4. Remove no_mechanism_travellers_dilemma from summary (simulate never started)
if 'no_mechanism_travellers_dilemma' in experiments:
    del experiments['no_mechanism_travellers_dilemma']
    print("○ no_mechanism_travellers_dilemma: Removed from summary (never started)")

# Update completed count
summary['completed_experiments'] = len([
    e for e in experiments.values()
    if e.get('status') in ['success', 'failed']
])

# Write modified summary
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=4)

print(f"\n✓ Updated batch_summary.json")
print(f"  Completed experiments: {summary['completed_experiments']}/{summary['total_experiments']}")
PYEOF

echo ""

# Delete the travellers_dilemma directory to simulate "never started"
if [ -d "$BATCH_DIR/no_mechanism_travellers_dilemma" ]; then
    rm -rf "$BATCH_DIR/no_mechanism_travellers_dilemma"
    echo -e "${BLUE}✓ Deleted no_mechanism_travellers_dilemma directory${NC}"
fi

echo ""
echo "============================================================"
echo "FINAL BATCH STATE"
echo "============================================================"

python3 << PYEOF
import json

with open('${BATCH_DIR}/batch_summary.json', 'r') as f:
    summary = json.load(f)

print(f"Total experiments: {summary['total_experiments']}")
print(f"Completed: {summary['completed_experiments']}")
print("")
print("Experiment statuses:")
for name in sorted(summary['experiments'].keys()):
    exp = summary['experiments'][name]
    status = exp.get('status', 'unknown')
    status_emoji = {'success': '✓', 'failed': '✗', 'in_progress': '⋯'}.get(status, '?')
    print(f"  {status_emoji} {name}: {status}")

# Check for travellers_dilemma
if 'no_mechanism_travellers_dilemma' not in summary['experiments']:
    print(f"  ○ no_mechanism_travellers_dilemma: NOT IN SUMMARY (never started)")
PYEOF

echo ""
echo "============================================================"
echo "EXPECTED RESUME BEHAVIOR"
echo "============================================================"
echo -e "${GREEN}✓ no_mechanism_prisoners_dilemma: SKIP (already successful)${NC}"
echo -e "${YELLOW}⟳ no_mechanism_public_goods: RETRY (failed, retry enabled)${NC}"
echo -e "${YELLOW}⟳ no_mechanism_trust_game: RETRY (in_progress = crashed)${NC}"
echo -e "${YELLOW}⟳ no_mechanism_travellers_dilemma: RUN (never started)${NC}"
echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "To test resume:"
echo "  ./script/run_batch.sh --resume $BATCH_DIR --local"
echo ""
echo "To restore original:"
echo "  mv $BATCH_DIR/batch_summary.json.backup $BATCH_DIR/batch_summary.json"
echo ""