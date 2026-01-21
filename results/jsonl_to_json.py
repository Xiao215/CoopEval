import json
import sys


def convert_jsonl_to_json(input_path: str):
    if not input_path.endswith(".jsonl"):
        print("Error: Input file must have a .jsonl extension.")
        return

    output_path = input_path.replace(".jsonl", ".json")

    try:
        with open(input_path, "r", encoding="utf-8") as infile:
            data = [json.loads(line) for line in infile if line.strip()]

        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(data, outfile, indent=2)

        print(f"✅ Converted {input_path} → {output_path}")

    except FileNotFoundError:
        print(f"Error: File not found: {input_path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python jsonl_to_json.py <path_to_file.jsonl>")
    else:
        convert_jsonl_to_json(sys.argv[1])
