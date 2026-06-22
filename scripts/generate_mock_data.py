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

    # Generate mock traces
    mock_traces = [
        # req_001
        {
            "span_id": "8a0f2b3c4d5e6f7a",
            "trace_id": "0123456789abcdef0123456789abcdef",
            "parent_span_id": "N/A",
            "name": "request_queue_latency",
            "start_time": "2026-06-21T18:00:00.000Z",
            "end_time": "2026-06-21T18:00:00.015Z",
            "duration_ms": 15.0,
            "service_name": "infer",
            "status": "ok"
        },
        {
            "span_id": "7b1e2c3d4e5f6a7b",
            "trace_id": "0123456789abcdef0123456789abcdef",
            "parent_span_id": "8a0f2b3c4d5e6f7a",
            "name": "generate_tokens",
            "start_time": "2026-06-21T18:00:00.015Z",
            "end_time": "2026-06-21T18:00:01.165Z",
            "duration_ms": 1150.0,
            "service_name": "infer",
            "status": "ok"
        },
        # req_002
        {
            "span_id": "2222222222222222",
            "trace_id": "22222222222222222222222222222222",
            "parent_span_id": "N/A",
            "name": "request_queue_latency",
            "start_time": "2026-06-21T18:00:01.000Z",
            "end_time": "2026-06-21T18:00:01.005Z",
            "duration_ms": 5.0,
            "service_name": "infer",
            "status": "ok"
        },
        {
            "span_id": "2222222222222223",
            "trace_id": "22222222222222222222222222222222",
            "parent_span_id": "2222222222222222",
            "name": "generate_tokens",
            "start_time": "2026-06-21T18:00:01.005Z",
            "end_time": "2026-06-21T18:00:01.955Z",
            "duration_ms": 950.0,
            "service_name": "infer",
            "status": "ok"
        },
        # req_003
        {
            "span_id": "3333333333333333",
            "trace_id": "33333333333333333333333333333333",
            "parent_span_id": "N/A",
            "name": "request_queue_latency",
            "start_time": "2026-06-21T18:00:02.000Z",
            "end_time": "2026-06-21T18:00:02.020Z",
            "duration_ms": 20.0,
            "service_name": "infer",
            "status": "ok"
        },
        {
            "span_id": "3333333333333334",
            "trace_id": "33333333333333333333333333333333",
            "parent_span_id": "3333333333333333",
            "name": "generate_tokens",
            "start_time": "2026-06-21T18:00:02.020Z",
            "end_time": "2026-06-21T18:00:03.470Z",
            "duration_ms": 1450.0,
            "service_name": "infer",
            "status": "ok"
        },
        # req_004
        {
            "span_id": "4444444444444444",
            "trace_id": "44444444444444444444444444444444",
            "parent_span_id": "N/A",
            "name": "request_queue_latency",
            "start_time": "2026-06-21T18:00:03.000Z",
            "end_time": "2026-06-21T18:00:03.010Z",
            "duration_ms": 10.0,
            "service_name": "infer",
            "status": "ok"
        },
        {
            "span_id": "4444444444444445",
            "trace_id": "44444444444444444444444444444444",
            "parent_span_id": "4444444444444444",
            "name": "generate_tokens",
            "start_time": "2026-06-21T18:00:03.010Z",
            "end_time": "2026-06-21T18:00:04.2305Z",
            "duration_ms": 1220.5,
            "service_name": "infer",
            "status": "ok"
        },
        # req_005
        {
            "span_id": "5555555555555555",
            "trace_id": "55555555555555555555555555555555",
            "parent_span_id": "N/A",
            "name": "request_queue_latency",
            "start_time": "2026-06-21T18:00:04.000Z",
            "end_time": "2026-06-21T18:00:04.015Z",
            "duration_ms": 15.0,
            "service_name": "infer",
            "status": "ok"
        },
        {
            "span_id": "5555555555555556",
            "trace_id": "55555555555555555555555555555555",
            "parent_span_id": "5555555555555555",
            "name": "generate_tokens",
            "start_time": "2026-06-21T18:00:04.015Z",
            "end_time": "2026-06-21T18:00:05.247Z",
            "duration_ms": 1232.0,
            "service_name": "infer",
            "status": "ok"
        }
    ]

    traces_path = output_path.parent / f"{output_path.name.replace('.json', '')}_traces.json"
    with open(traces_path, "w", encoding="utf-8") as f:
        json.dump(mock_traces, f, indent=2)

    print(f"Generated mock traces file: {traces_path}")

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
