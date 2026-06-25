"""
Agent Runtime Benchmark Simulator
===================================
Models the latency profile of LLM-based agents executing multi-step
ReAct-style planning loops with external tool calls.

Design reference: ReAct (Yao et al., 2023), Toolformer (Schick et al., 2023)
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

TOOL_LATENCY_PROFILES: dict[str, dict[str, float]] = {
    "search": {"mean_ms": 320.0, "std_ms": 80.0},
    "calculator": {"mean_ms": 5.0, "std_ms": 1.0},
    "code_interpreter": {"mean_ms": 1800.0, "std_ms": 600.0},
    "retrieval": {"mean_ms": 180.0, "std_ms": 45.0},
    "file_io": {"mean_ms": 40.0, "std_ms": 12.0},
}


class AgentSimulator:
    """
    Simulates the turn-by-turn execution of a ReAct-style agent:
    each turn = Thought (LLM generate) + Action (tool call) + Observation (tool result).

    Parameters
    ----------
    base_ttft_ms : float
        Base time-to-first-token in milliseconds for the underlying LLM.
    time_per_token_ms : float
        Incremental generation time per output token in milliseconds.
    prompt_processing_rate_ms : float
        Time in milliseconds to process each prompt token.
    context_growth_penalty : float
        Multiplicative factor applied to prompt_processing_rate_ms as context
        grows across turns (models the KV-cache miss cost when context exceeds
        the cache budget).
    """

    def __init__(
        self,
        base_ttft_ms: float = 150.0,
        time_per_token_ms: float = 12.0,
        prompt_processing_rate_ms: float = 0.4,
        context_growth_penalty: float = 1.002,
    ) -> None:
        self.base_ttft_ms = base_ttft_ms
        self.time_per_token_ms = time_per_token_ms
        self.prompt_processing_rate_ms = prompt_processing_rate_ms
        self.context_growth_penalty = context_growth_penalty

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _llm_latency(self, prompt_tokens: int, completion_tokens: int, turn: int) -> tuple[float, float]:
        """Return (ttft_ms, total_latency_ms) for one LLM generation call.

        Accounts for context window growth penalty across turns.
        """
        penalty = self.context_growth_penalty ** turn
        noise_ttft = random.gauss(0.0, 15.0)
        ttft_ms = (
            self.base_ttft_ms
            + prompt_tokens * self.prompt_processing_rate_ms * penalty
            + max(0.0, noise_ttft)
        )
        noise_gen = random.gauss(0.0, 20.0)
        generation_ms = completion_tokens * self.time_per_token_ms + max(0.0, noise_gen)
        return round(ttft_ms, 2), round(ttft_ms + generation_ms, 2)

    @staticmethod
    def _tool_latency(tool_name: str) -> float:
        """Return simulated latency in ms for a single tool call."""
        profile = TOOL_LATENCY_PROFILES.get(tool_name, {"mean_ms": 200.0, "std_ms": 60.0})
        latency = random.gauss(profile["mean_ms"], profile["std_ms"])
        return round(max(1.0, latency), 2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_agent_trajectory(
        self,
        task_id: str,
        max_turns: int = 5,
        tools: list[str] | None = None,
        tool_call_probability: float = 0.75,
    ) -> dict[str, Any]:
        """Simulate one complete agent task trajectory.

        Each turn consists of:
          1. Think: LLM generates a reasoning trace (thought tokens)
          2. Act: LLM generates a tool call or final answer
          3. Observe: Tool executes and returns a result (if tool was called)

        Parameters
        ----------
        task_id : str
            Unique identifier for this task run.
        max_turns : int
            Maximum number of think-act-observe cycles before forced stop.
        tools : list[str] | None
            Pool of tool names available to the agent. Defaults to all profiles.
        tool_call_probability : float
            Probability that any given turn triggers a tool call vs. final answer.
        """
        if tools is None:
            tools = list(TOOL_LATENCY_PROFILES.keys())

        trace_id = uuid.uuid4().hex
        turns: list[dict[str, Any]] = []
        cumulative_context_tokens = 128  # system prompt baseline
        total_latency_ms = 0.0
        total_tool_latency_ms = 0.0
        actual_turns = 0

        for turn_idx in range(max_turns):
            actual_turns = turn_idx + 1

            # --- Think step (reasoning trace generation) ---
            thought_prompt_tokens = cumulative_context_tokens
            thought_completion_tokens = random.randint(30, 120)
            ttft_ms, think_latency_ms = self._llm_latency(
                thought_prompt_tokens, thought_completion_tokens, turn=turn_idx
            )

            # --- Act step (tool call or final answer) ---
            act_prompt_tokens = thought_prompt_tokens + thought_completion_tokens
            act_completion_tokens = random.randint(20, 80)
            _, act_latency_ms = self._llm_latency(
                act_prompt_tokens, act_completion_tokens, turn=turn_idx
            )

            # Decide: tool call or final answer?
            is_final_answer = (turn_idx == max_turns - 1) or (
                random.random() > tool_call_probability
            )
            tool_called: str | None = None
            tool_latency_ms = 0.0

            if not is_final_answer and tools:
                tool_called = random.choice(tools)
                tool_latency_ms = self._tool_latency(tool_called)
                total_tool_latency_ms += tool_latency_ms
                # Observation tokens added to context
                observation_tokens = random.randint(50, 200)
                cumulative_context_tokens = act_prompt_tokens + act_completion_tokens + observation_tokens
            else:
                # Final answer — no tool call
                cumulative_context_tokens = act_prompt_tokens + act_completion_tokens
                is_final_answer = True  # force stop

            turn_total_ms = think_latency_ms + act_latency_ms + tool_latency_ms
            total_latency_ms += turn_total_ms

            turns.append(
                {
                    "turn": turn_idx + 1,
                    "think_latency_ms": think_latency_ms,
                    "act_latency_ms": act_latency_ms,
                    "tool_called": tool_called,
                    "tool_latency_ms": tool_latency_ms,
                    "ttft_ms": ttft_ms,
                    "turn_total_ms": round(turn_total_ms, 2),
                    "context_tokens_after": cumulative_context_tokens,
                    "is_final_answer": is_final_answer,
                }
            )

            if is_final_answer:
                break

        llm_only_ms = total_latency_ms - total_tool_latency_ms
        tool_overhead_pct = (
            (total_tool_latency_ms / total_latency_ms * 100) if total_latency_ms > 0 else 0.0
        )

        return {
            "task_id": task_id,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "num_turns": actual_turns,
            "total_latency_ms": round(total_latency_ms, 2),
            "llm_only_latency_ms": round(llm_only_ms, 2),
            "total_tool_latency_ms": round(total_tool_latency_ms, 2),
            "tool_overhead_pct": round(tool_overhead_pct, 2),
            "turns": turns,
        }

    def run_benchmark(
        self,
        benchmark_id: str,
        num_tasks: int = 50,
        max_turns_per_task: int = 5,
        tools: list[str] | None = None,
        tool_call_probability: float = 0.75,
    ) -> dict[str, Any]:
        """Run a full agent benchmark over N independent tasks.

        Returns aggregate statistics (p50/p95/p99 latencies) alongside
        per-task trajectories.
        """
        tasks = []
        for i in range(num_tasks):
            task = self.simulate_agent_trajectory(
                task_id=f"task_{i:04d}",
                max_turns=max_turns_per_task,
                tools=tools,
                tool_call_probability=tool_call_probability,
            )
            tasks.append(task)

        latencies = sorted(t["total_latency_ms"] for t in tasks)
        tool_latencies = sorted(t["total_tool_latency_ms"] for t in tasks)
        turns_counts = [t["num_turns"] for t in tasks]

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = int(len(data) * p / 100)
            return round(data[min(idx, len(data) - 1)], 2)

        mean_turns = round(sum(turns_counts) / len(turns_counts), 2) if turns_counts else 0.0
        tool_overhead_pcts = [t["tool_overhead_pct"] for t in tasks]
        mean_tool_overhead_pct = (
            round(sum(tool_overhead_pcts) / len(tool_overhead_pcts), 2) if tool_overhead_pcts else 0.0
        )

        return {
            "benchmark_id": benchmark_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "config": {
                "num_tasks": num_tasks,
                "max_turns_per_task": max_turns_per_task,
                "tools_available": tools or list(TOOL_LATENCY_PROFILES.keys()),
                "tool_call_probability": tool_call_probability,
            },
            "aggregate_metrics": {
                "total_latency_ms": {
                    "p50": percentile(latencies, 50),
                    "p95": percentile(latencies, 95),
                    "p99": percentile(latencies, 99),
                    "mean": round(sum(latencies) / len(latencies), 2),
                },
                "tool_latency_ms": {
                    "p50": percentile(tool_latencies, 50),
                    "p95": percentile(tool_latencies, 95),
                    "p99": percentile(tool_latencies, 99),
                    "mean": round(sum(tool_latencies) / len(tool_latencies), 2),
                },
                "mean_turns_per_task": mean_turns,
                "mean_tool_overhead_pct": mean_tool_overhead_pct,
            },
            "tasks": tasks,
        }
