from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from jsonschema import validate, ValidationError

ROOT = Path(__file__).resolve().parents[1]

PROFILE_SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "benchmark_profile.json"
PROFILES_DIR = ROOT / "benchmark" / "profiles"


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


def validate_profile(profile_path: Path, schema_path: Path) -> None:
    schema = load_json(schema_path)
    data = load_json(profile_path)
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        fail(f"Profile validation error for {profile_path.name}: {e.message}")

    # Additional semantic checks
    pc = data.get("prompt_config", {})
    if pc.get("min_prompt_tokens", 0) > pc.get("max_prompt_tokens", 0):
        fail(f"{profile_path.name}: min_prompt_tokens > max_prompt_tokens")
    if pc.get("min_completion_tokens", 0) > pc.get("max_completion_tokens", 0):
        fail(f"{profile_path.name}: min_completion_tokens > max_completion_tokens")

    cc = data.get("concurrency_config", {})
    if cc.get("concurrent_requests", 0) > cc.get("total_requests", 0):
        fail(f"{profile_path.name}: concurrent_requests > total_requests")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark profile configuration files")
    parser.add_argument("--profile", type=str, help="Path to a specific profile JSON file")
    parser.add_argument("--schema", type=str, help="Path to the benchmark profile JSON schema")
    args = parser.parse_args()

    schema_path = Path(args.schema) if args.schema else PROFILE_SCHEMA_PATH

    if args.profile:
        validate_profile(Path(args.profile), schema_path)
        print(f"Successfully validated profile: {args.profile}")
    else:
        if not PROFILES_DIR.is_dir():
            print("No profiles directory found.")
            return
        found = False
        for path in PROFILES_DIR.rglob("*.json"):
            validate_profile(path, schema_path)
            print(f"Successfully validated profile: {path.relative_to(ROOT)}")
            found = True
        if not found:
            print("No benchmark profile files found to validate.")


if __name__ == "__main__":
    main()
