#!/usr/bin/env bash
# Run the strategy judge across the 10 deepseek self-play ablation runs
# (probabilistic and direct-action variants) and concatenate the results.
#
# Usage:
#     bash post_processing/judge/run_strategy_judge.sh
#
# Optional environment variables:
#   - JUDGE_MODEL (default: deepseek/deepseek-chat-v3.1)
#   - OUTPUT_JSON  (default: outputs/2025/11/03/strategy_judgements_deepseek.json)

set -euo pipefail

MODEL="${JUDGE_MODEL:-deepseek/deepseek-chat-v3.1}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/2025/11/03/strategy_judgements_deepseek.json}"

RUNS=(
  "outputs/2025/11/03/18:06_probabilistic_deepseek/Repetition_PrisonersDilemma_readable.json"
  "outputs/2025/11/03/18:16_probabilistic_deepseek/Repetition_PrisonersDilemma_readable.json"
  "outputs/2025/11/03/18:25_probabilistic_deepseek/Repetition_PrisonersDilemma_readable.json"
  "outputs/2025/11/03/18:32_probabilistic_deepseek/Repetition_PrisonersDilemma_readable.json"
  "outputs/2025/11/03/18:39_probabilistic_deepseek/Repetition_PrisonersDilemma_readable.json"
  "outputs/2025/11/03/18:44_direct_deepseek/Repetition_PrisonersDilemmaDirect_readable.json"
  "outputs/2025/11/03/18:53_direct_deepseek/Repetition_PrisonersDilemmaDirect_readable.json"
  "outputs/2025/11/03/19:03_direct_deepseek/Repetition_PrisonersDilemmaDirect_readable.json"
  "outputs/2025/11/03/19:09_direct_deepseek/Repetition_PrisonersDilemmaDirect_readable.json"
  "outputs/2025/11/03/19:20_direct_deepseek/Repetition_PrisonersDilemmaDirect_readable.json"
)

declare -a TMP_FILES=()

for run_path in "${RUNS[@]}"; do
  run_dir="$(dirname "$run_path")"
  run_name="$(basename "$run_dir")"
  tmp_file="${run_dir}/strategy_judgements_${run_name}.json"

  echo "Judging ${run_name} with model ${MODEL}..."
  python post_processing/judge/strategy_judge.py \
    --input "$run_path" \
    --model "$MODEL" \
    --output "$tmp_file"

  TMP_FILES+=("$tmp_file")
done

echo "Aggregating results into ${OUTPUT_JSON}..."
python - <<'PY' "${OUTPUT_JSON}" "${TMP_FILES[@]}"
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
input_paths = [Path(p) for p in sys.argv[2:]]

aggregate = []
for path in input_paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    aggregate.append(
        {
            "source": str(path),
            "entries": data,
        }
    )

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
print(f"Wrote aggregated judgements to {output_path}")
PY

echo "Done."
