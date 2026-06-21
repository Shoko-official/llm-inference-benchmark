from __future__ import annotations

"""validate_profile.py - Validate benchmark profile JSON files against schema.

Checks:
- Profile conforms to benchmark/schemas/benchmark_profile.json
- min_prompt_tokens <= max_prompt_tokens
- min_completion_tokens <= max_completion_tokens
- concurrent_requests >= 1 and total_requests >= concurrent_requests
- base_ttft_ms and time_per_token_ms are non-negative
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from jsonschema import validate, ValidationError
except ImportError:
    print("Error: jsonschema is required. Run: pip install jsonschema", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA = ROOT / "benchmark" / "schemas" / "benchmark_profile.json"


def fail(message: str) -> None:
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def load_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"File not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        fail(f"Invalid JSON in {path}: {e}")


def validate_schema(data: dict, schema: dict, source: str) -> None:
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        fail(f"Schema validation error in {source}: {e.message}")


def validate_profile_logic(data: dict, source: str) -> None:
    """Apply semantic checks beyond JSON Schema."""
    pc = data.get("prompt_config", {})
    cc = data.get("concurrency_config", {})
    lc = data.get("latency_config", {})

    if pc.get("min_prompt_tokens", 1) > pc.get("max_prompt_tokens", 1):
        fail(f"{source}: min_prompt_tokens must be <= max_prompt_tokens")

    if pc.get("min_completion_tokens", 1) > pc.get("max_completion_tokens", 1):
        fail(f"{source}: min_completion_tokens must be <= max_completion_tokens")

    if cc.get("total_requests", 1) < cc.get("concurrent_requests", 1):
        fail(f"{source}: total_requests must be >= concurrent_requests")

    if lc.get("base_ttft_ms", 0) < 0:
        fail(f"{source}: base_ttft_ms must be >= 0")

    if lc.get("time_per_token_ms", 0) < 0:
        fail(f"{source}: time_per_token_ms must be >= 0")


def validate_profile_file(profile_path: Path, schema: dict) -> None:
    data = load_json(profile_path)
    source = profile_path.name
    validate_schema(data, schema, source)
    validate_profile_logic(data, source)
    print(f"  ok: {profile_path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate benchmark profile JSON files against schema"
    )
    parser.add_argument(
        "profiles",
        nargs="*",
        help="Path(s) to profile JSON file(s). If omitted, scans benchmark/profiles/",
    )
    parser.add_argument(
        "--schema",
        type=str,
        help="Path to the benchmark profile schema (default: benchmark/schemas/benchmark_profile.json)",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema) if args.schema else PROFILE_SCHEMA
    if not schema_path.is_file():
        fail(f"Schema not found: {schema_path}")

    schema = load_json(schema_path)

    if args.profiles:
        paths = [Path(p) for p in args.profiles]
    else:
        profiles_dir = ROOT / "benchmark" / "profiles"
        if not profiles_dir.is_dir():
            fail(f"Profiles directory not found: {profiles_dir}")
        paths = sorted(profiles_dir.glob("*.json"))
        if not paths:
            fail(f"No profile JSON files found in {profiles_dir}")

    print(f"Validating {len(paths)} profile(s) against {schema_path.name}...")
    for path in paths:
        validate_profile_file(path, schema)

    print(f"All {len(paths)} profile(s) valid.")


if __name__ == "__main__":
    main()
