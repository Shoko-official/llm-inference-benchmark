from __future__ import annotations

import argparse
import json
from pathlib import Path

def generate_mock_data(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mock_run = {
        "run_id": "mock_inference_run_001",
        "model_id": "mock_llm_v1",
        "timestamp": "2026-06-21T18:00:00Z",
        "metrics": {
            "throughput_tokens_per_sec": 45.2,
            "mean_latency_ms": 1200.5,
            "mean_ttft_ms": 250.2,
            "concurrent_requests": 4
        },
        "requests": [
            {
                "request_id": "req_001",
                "prompt_tokens": 128,
                "completion_tokens": 256,
                "latency_ms": 1150.0,
                "ttft_ms": 240.0
            },
            {
                "request_id": "req_002",
                "prompt_tokens": 64,
                "completion_tokens": 128,
                "latency_ms": 950.0,
                "ttft_ms": 220.0
            },
            {
                "request_id": "req_003",
                "prompt_tokens": 256,
                "completion_tokens": 512,
                "latency_ms": 1450.0,
                "ttft_ms": 280.0
            },
            {
                "request_id": "req_004",
                "prompt_tokens": 128,
                "completion_tokens": 256,
                "latency_ms": 1220.5,
                "ttft_ms": 260.4
            },
            {
                "request_id": "req_005",
                "prompt_tokens": 96,
                "completion_tokens": 192,
                "latency_ms": 1232.0,
                "ttft_ms": 250.8
            }
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mock_run, f, indent=2)

    print(f"Generated mock inference run file: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic inference benchmark data for testing")
    parser.add_argument("--output", type=str, help="Destination path for the mock run JSON")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    
    output_path = Path(args.output) if args.output else root_dir / "benchmark" / "mock_run.json"
    
    generate_mock_data(output_path)

if __name__ == "__main__":
    main()
