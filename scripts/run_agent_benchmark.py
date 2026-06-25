#!/usr/bin/env python3
"""
Run agent tool-call latency benchmark and write results to results/agent_tool_call_benchmark.json.

Usage:
    python scripts/run_agent_benchmark.py [--tasks N] [--turns N] [--seed N]

Closes: https://github.com/Shoko-official/llm-inference-benchmark/issues/43
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running from repo root or scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.agent_simulator import AgentSimulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent tool-call latency benchmark runner")
    parser.add_argument("--tasks", type=int, default=50, help="Number of agent tasks to simulate")
    parser.add_argument("--turns", type=int, default=5, help="Max ReAct turns per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    simulator = AgentSimulator(
        base_ttft_ms=150.0,
        time_per_token_ms=12.0,
        prompt_processing_rate_ms=0.4,
        context_growth_penalty=1.002,  # 0.2% overhead per turn for KV-cache growth
    )

    print(f"Running agent benchmark: {args.tasks} tasks, max {args.turns} turns, seed={args.seed}")

    results = simulator.run_benchmark(
        benchmark_id="agent-tool-call-benchmark-v1",
        num_tasks=args.tasks,
        max_turns_per_task=args.turns,
        tool_call_probability=0.75,
    )

    # Write output
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "agent_tool_call_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    m = results["aggregate_metrics"]
    lat = m["total_latency_ms"]
    tool = m["tool_latency_ms"]
    print(
        f"\nResults written to {out_path}\n"
        f"  Total latency  p50={lat['p50']}ms  p95={lat['p95']}ms  p99={lat['p99']}ms\n"
        f"  Tool latency   p50={tool['p50']}ms  p95={tool['p95']}ms  p99={tool['p99']}ms\n"
        f"  Mean turns/task: {m['mean_turns_per_task']}\n"
        f"  Mean tool overhead: {m['mean_tool_overhead_pct']}%"
    )


if __name__ == "__main__":
    main()
