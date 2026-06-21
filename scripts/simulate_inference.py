from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.simulator import LatencySimulator

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark Simulator CLI")
    parser.add_argument("--run-id", type=str, help="Unique run ID")
    parser.add_argument("--model-id", type=str, default="mock_llm_v1", help="Model identifier")
    parser.add_argument("--num-requests", type=int, default=50, help="Number of simulated requests")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent request streams")
    parser.add_argument("--output", type=str, help="Output JSON run log path")
    parser.add_argument("--base-ttft", type=float, default=150.0, help="Base TTFT in ms")
    parser.add_argument("--time-per-token", type=float, default=12.0, help="Time per token in ms")
    
    args = parser.parse_args()
    
    run_id = args.run_id if args.run_id else f"run_{uuid.uuid4().hex[:8]}"
    output_path = Path(args.output) if args.output else ROOT / "benchmark" / "mock_run.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sim = LatencySimulator(
        base_ttft_ms=args.base_ttft,
        time_per_token_ms=args.time_per_token
    )
    
    run_data = sim.simulate_run(
        run_id=run_id,
        model_id=args.model_id,
        num_requests=args.num_requests,
        concurrent_requests=args.concurrency
    )
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2)
        
    try:
        rel_out = output_path.relative_to(ROOT)
    except ValueError:
        rel_out = output_path
    print(f"Simulator completed. Saved {len(run_data['requests'])} requests to {rel_out}")

if __name__ == "__main__":
    main()
