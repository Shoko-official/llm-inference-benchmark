#!/usr/bin/env python3
"""
Multi-agent parallel execution concurrency benchmark.

Simulates N agents running simultaneously and measures:
- Per-agent latency distribution under load
- Aggregate throughput degradation as N grows
- Contention overhead vs. ideal scaling

Closes: https://github.com/Shoko-official/llm-inference-benchmark/issues/44
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.agent_simulator import AgentSimulator


def run_concurrency_sweep(
    concurrency_levels: list[int],
    tasks_per_agent: int = 10,
    max_turns: int = 4,
    seed: int = 42,
) -> dict:
    """Sweep over concurrency levels and collect per-level latency stats.

    For each concurrency level N, we simulate N×tasks_per_agent tasks
    using shared LLM capacity — modeled by inflating the context_growth_penalty
    to reflect contention (each additional agent adds a small queue delay).
    """
    results_by_level: list[dict] = []

    for n_agents in concurrency_levels:
        # Contention model: each concurrent agent adds ~2% overhead to processing
        # (simplified queuing penalty — not a full M/M/c model)
        contention_factor = 1.0 + (n_agents - 1) * 0.02
        random.seed(seed)

        sim = AgentSimulator(
            base_ttft_ms=150.0 * contention_factor,   # queue penalty
            time_per_token_ms=12.0 * contention_factor,
            prompt_processing_rate_ms=0.4,
            context_growth_penalty=1.002,
        )

        total_tasks = n_agents * tasks_per_agent
        benchmark = sim.run_benchmark(
            benchmark_id=f"multi-agent-concurrency-n{n_agents}",
            num_tasks=total_tasks,
            max_turns_per_task=max_turns,
            tool_call_probability=0.70,
        )

        m = benchmark["aggregate_metrics"]
        lat = m["total_latency_ms"]
        ideal_throughput = 1.0  # normalized to n_agents=1
        # Per-agent throughput: tasks_per_agent / (p95 latency in seconds)
        # This measures how much work a single agent can do in the p95 time window
        # and correctly captures degradation as contention grows
        throughput_score = tasks_per_agent / (lat["p95"] / 1000.0) if lat["p95"] > 0 else 0.0

        results_by_level.append(
            {
                "n_agents": n_agents,
                "total_tasks": total_tasks,
                "contention_factor": round(contention_factor, 4),
                "latency_ms": lat,
                "tool_latency_ms": m["tool_latency_ms"],
                "mean_turns_per_task": m["mean_turns_per_task"],
                "mean_tool_overhead_pct": m["mean_tool_overhead_pct"],
                "throughput_tasks_per_sec": round(throughput_score, 4),
            }
        )

    # Compute throughput degradation ratio vs. single-agent baseline
    baseline_tput = results_by_level[0]["throughput_tasks_per_sec"] if results_by_level else 1.0
    for r in results_by_level:
        r["throughput_degradation_pct"] = round(
            (1.0 - r["throughput_tasks_per_sec"] / baseline_tput) * 100, 2
        ) if baseline_tput > 0 else 0.0

    return {
        "benchmark_id": "multi-agent-concurrency-v1",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "concurrency_levels": concurrency_levels,
            "tasks_per_agent": tasks_per_agent,
            "max_turns": max_turns,
            "seed": seed,
        },
        "results": results_by_level,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent concurrency benchmark")
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16],
        help="Concurrency levels (number of agents) to sweep"
    )
    parser.add_argument("--tasks-per-agent", type=int, default=10)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Multi-agent concurrency sweep: {args.levels} agents")
    results = run_concurrency_sweep(
        concurrency_levels=args.levels,
        tasks_per_agent=args.tasks_per_agent,
        max_turns=args.turns,
        seed=args.seed,
    )

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "multi_agent_concurrency.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nResults written to {out_path}\n")
    print(f"{'Agents':>8} {'p50 ms':>10} {'p95 ms':>10} {'Throughput degrad.':>20}")
    print("-" * 55)
    for r in results["results"]:
        lat = r["latency_ms"]
        print(
            f"{r['n_agents']:>8} {lat['p50']:>10.1f} {lat['p95']:>10.1f}"
            f" {r['throughput_degradation_pct']:>19.1f}%"
        )


if __name__ == "__main__":
    main()
