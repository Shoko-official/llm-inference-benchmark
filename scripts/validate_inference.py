from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from jsonschema import validate, ValidationError

ROOT = Path(__file__).resolve().parents[1]

def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)

def load_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        fail(f"Failed to parse JSON from {path}: {e}")

def validate_run_file(run_path: Path, schema_path: Path) -> None:
    schema = load_json(schema_path)
    data = load_json(run_path)
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        fail(f"Inference run validation error for {run_path.name}: {e.message}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LLM inference benchmark logs against schemas")
    parser.add_argument("--run", type=str, help="Path to a run log JSON file to validate")
    parser.add_argument("--schema", type=str, help="Path to the inference run JSON schema")
    
    args = parser.parse_args()
    
    schema_path = Path(args.schema) if args.schema else ROOT / "benchmark" / "schemas" / "inference_run.json"
    
    if args.run:
        validate_run_file(Path(args.run), schema_path)
        print(f"Successfully validated run file: {args.run}")
    else:
        # Find all JSON files in benchmark/ excluding schemas/
        bench_dir = ROOT / "benchmark"
        found = False
        for path in bench_dir.rglob("*.json"):
            if "schemas" in path.parts:
                continue
            if "profiles" in path.parts:
                continue
            validate_run_file(path, schema_path)
            print(f"Successfully validated run file: {path.relative_to(ROOT)}")
            found = True
        if not found:
            print("No inference run files found to validate.")

if __name__ == "__main__":
    main()
