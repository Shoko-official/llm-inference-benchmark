#!/usr/bin/env python3
"""
KV-cache efficiency benchmark for growing agent context windows.

Simulates repeated tool-call turns where context grows across the conversation,
measuring KV-cache hit rate (estimated) vs. memory footprint as context length
increases.

Design notes:
  - KV-cache hit rate model: cache captures tokens already processed in prior
    turns. Hit rate = prior_context / total_context.
  - Memory model: KV memory ≈ 2 × layers × heads × head_dim × seq_len × 2 bytes
    (float16). We parameterize with a simplified kv_bytes_per_token.
  - We sweep context window sizes from a small (512-token) conversation to a
    large (16k-token) multi-turn agent session.

Closes: https://github.com/Shoko-official/llm-inference-benchmark/issues/45
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


# Model constants (representative of a 7B-param model, 32 layers)
KV_BYTES_PER_TOKEN_FP16 = 512  # 2 (K+V) × 32 layers × 8 heads × head_dim=64 × 2 bytes


def estimate_kv_stats(
    total_context_tokens: int,
    prior_context_tokens: int,
    kv_bytes_per_token: int = KV_BYTES_PER_TOKEN_FP16,
) -> dict:
    """Estimate KV-cache hit rate and memory usage.

    Parameters
    ----------
    total_context_tokens : int
        All tokens in the current context window (including new tokens).
    prior_context_tokens : int
        Tokens that were already processed in prior turns (cache candidates).
    kv_bytes_per_token : int
        Bytes per token in KV-cache (depends on model architecture and dtype).
    """
    hit_rate = prior_context_tokens / total_context_tokens if total_context_tokens > 0 else 0.0
    memory_bytes = total_context_tokens * kv_bytes_per_token
    return {
        "hit_rate": round(hit_rate, 4),
        "memory_mb": round(memory_bytes / (1024 * 1024), 3),
        "total_tokens": total_context_tokens,
        "cached_tokens": prior_context_tokens,
        "new_tokens": total_context_tokens - prior_context_tokens,
    }


def run_kv_cache_benchmark(
    num_tasks: int = 50,
    max_turns: int = 8,
    seed: int = 42,
) -> dict:
    """Run agent tasks and track KV-cache efficiency per turn.

    For each task, we simulate a full ReAct trajectory and at each turn
    compute the estimated KV hit rate and memory footprint.
    """
    random.seed(seed)
    sim = AgentSimulator(
        base_ttft_ms=150.0,
        time_per_token_ms=12.0,
        prompt_processing_rate_ms=0.4,
        context_growth_penalty=1.002,
    )

    per_turn_stats: list[dict] = []

    for i in range(num_tasks):
        task = sim.simulate_agent_trajectory(
            task_id=f"task_{i:04d}",
            max_turns=max_turns,
            tool_call_probability=0.80,
        )

        prior_context = 128  # system prompt
        for turn in task["turns"]:
            total_ctx = turn["context_tokens_after"]
            kv = estimate_kv_stats(total_ctx, prior_context)
            per_turn_stats.append(
                {
                    "task_id": task["task_id"],
                    "turn": turn["turn"],
                    "total_latency_ms": turn["turn_total_ms"],
                    **kv,
                }
            )
            prior_context = total_ctx  # accumulate

    # Aggregate by turn index
    max_turn_seen = max(s["turn"] for s in per_turn_stats) if per_turn_stats else 0
    by_turn: list[dict] = []
    for t in range(1, max_turn_seen + 1):
        turn_entries = [s for s in per_turn_stats if s["turn"] == t]
        if not turn_entries:
            continue
        hit_rates = [s["hit_rate"] for s in turn_entries]
        memories = [s["memory_mb"] for s in turn_entries]
        latencies = [s["total_latency_ms"] for s in turn_entries]
        tokens = [s["total_tokens"] for s in turn_entries]
        by_turn.append(
            {
                "turn": t,
                "num_tasks": len(turn_entries),
                "mean_hit_rate": round(sum(hit_rates) / len(hit_rates), 4),
                "mean_memory_mb": round(sum(memories) / len(memories), 3),
                "mean_total_tokens": round(sum(tokens) / len(tokens), 1),
                "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
            }
        )

    # Compute overall aggregate
    all_hit_rates = [s["hit_rate"] for s in per_turn_stats]
    all_memories = [s["memory_mb"] for s in per_turn_stats]
    aggregate = {
        "mean_hit_rate": round(sum(all_hit_rates) / len(all_hit_rates), 4),
        "max_memory_mb": round(max(all_memories), 3),
        "p95_memory_mb": round(sorted(all_memories)[int(len(all_memories) * 0.95)], 3),
        "mean_memory_mb": round(sum(all_memories) / len(all_memories), 3),
    }

    return {
        "benchmark_id": "kv-cache-agent-context-v1",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "num_tasks": num_tasks,
            "max_turns": max_turns,
            "seed": seed,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN_FP16,
            "model_assumption": "7B params, 32 layers, fp16 KV-cache",
        },
        "aggregate": aggregate,
        "by_turn": by_turn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="KV-cache agent context benchmark")
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"KV-cache benchmark: {args.tasks} tasks, {args.turns} max turns")
    results = run_kv_cache_benchmark(args.tasks, args.turns, args.seed)

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "kv_cache_agent_context.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    agg = results["aggregate"]
    print(
        f"\nResults written to {out_path}\n"
        f"  Mean KV hit rate:  {agg['mean_hit_rate']:.1%}\n"
        f"  Mean memory/turn:  {agg['mean_memory_mb']} MB\n"
        f"  p95 memory/turn:   {agg['p95_memory_mb']} MB\n"
        f"  Max memory/turn:   {agg['max_memory_mb']} MB\n"
    )
    print(f"\n{'Turn':>6} {'Tasks':>6} {'Hit Rate':>10} {'Mem (MB)':>10} {'Tokens':>8} {'Latency ms':>12}")
    print("-" * 60)
    for row in results["by_turn"]:
        print(
            f"{row['turn']:>6} {row['num_tasks']:>6} {row['mean_hit_rate']:>10.1%}"
            f" {row['mean_memory_mb']:>10.2f} {row['mean_total_tokens']:>8.0f}"
            f" {row['mean_latency_ms']:>12.1f}"
        )


if __name__ == "__main__":
    main()
