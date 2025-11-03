# Post-processing Utilities

## `jsonl_to_readable_json.py`

- **Purpose**: convert tournament logs stored as JSONL (one match per line) into a readable JSON structure that enumerates rounds, players, realized actions, points, inferred `{A0, A1}` probability distributions, and the original natural-language responses.
- **Common use case**: inspect a single matchup after a run or archive a cleaner version alongside other analysis artifacts.

### Usage

```bash
python post_processing/jsonl_to_readable_json.py \
  --input outputs/<date>/<time>/Repetition_PrisonersDilemma.jsonl \
  --match-index 6 \
  --output outputs/<date>/<time>/deepseek_self_play.json
```

- `--input` *(required)*: path to the JSONL log produced by the run.
- `--match-index` *(optional)*: 1-based index of the match to export. Omit to export every match in the log as a JSON list.
- `--output` *(optional)*: destination path. If omitted, the script prints the JSON to stdout.

All paths are resolved relative to the repository root when the command is executed from there.
