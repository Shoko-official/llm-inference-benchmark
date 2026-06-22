#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.simulator import LatencySimulator

def main() -> int:
    parser = argparse.ArgumentParser(description="Run LLM latency/throughput scaling simulation across concurrency levels")
    parser.add_argument("--profile", type=str, help="Path to profile JSON file (default: benchmark/profiles/profile_high_concurrency.json)")
    parser.add_argument("--output", type=str, help="Path to save the scaling report (default: results/concurrency_scaling_report.json)")
    
    args = parser.parse_args()
    
    profile_path = Path(args.profile) if args.profile else ROOT / "benchmark" / "profiles" / "profile_high_concurrency.json"
    output_path = Path(args.output) if args.output else ROOT / "results" / "concurrency_scaling_report.json"
    
    if not profile_path.is_file():
        print(f"Error: Profile not found at {profile_path}", file=sys.stderr)
        return 1
        
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except Exception as e:
        print(f"Error: Failed to parse profile: {e}", file=sys.stderr)
        return 1
        
    profile_id = profile.get("profile_id", "custom_profile")
    model_id = profile.get("model_id", "mock-model-v1")
    concurrency_config = profile.get("concurrency_config", {})
    latency_config = profile.get("latency_config", {})
    
    total_requests = concurrency_config.get("total_requests", 100)
    
    concurrency_levels = [1, 2, 4, 8, 16]
    
    # Initialize simulator
    sim = LatencySimulator(
        base_ttft_ms=latency_config.get("base_ttft_ms", 150.0),
        time_per_token_ms=latency_config.get("time_per_token_ms", 12.0)
    )
    
    runs = []
    print(f"Starting scaling simulation for profile '{profile_id}' (Model: '{model_id}')")
    print(f"Simulating {total_requests} requests per concurrency level...")
    print(f"{'Concurrency':<12} | {'Throughput (tok/s)':<20} | {'Mean Latency (ms)':<20} | {'Mean TTFT (ms)':<15}")
    print("-" * 75)
    
    for c in concurrency_levels:
        run_data = sim.simulate_run(
            run_id=f"run_c{c}",
            model_id=model_id,
            num_requests=total_requests,
            concurrent_requests=c
        )
        
        metrics = run_data["metrics"]
        throughput = metrics["throughput_tokens_per_sec"]
        mean_latency = metrics["mean_latency_ms"]
        mean_ttft = metrics["mean_ttft_ms"]
        
        runs.append({
            "concurrency": c,
            "throughput_tokens_per_sec": throughput,
            "mean_latency_ms": mean_latency,
            "mean_ttft_ms": mean_ttft,
            "total_requests": total_requests
        })
        
        print(f"{c:<12} | {throughput:<20.2f} | {mean_latency:<20.2f} | {mean_ttft:<15.2f}")
        
    report = {
        "profile_id": profile_id,
        "model_id": model_id,
        "concurrency_runs": runs
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nScaling report successfully saved to: {output_path.resolve()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
