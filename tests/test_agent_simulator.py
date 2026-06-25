"""Tests for the agent runtime benchmark simulator (M8-INFER-01)."""
from __future__ import annotations

import random
import pytest
from benchmark.agent_simulator import AgentSimulator, TOOL_LATENCY_PROFILES


@pytest.fixture
def sim() -> AgentSimulator:
    random.seed(0)
    return AgentSimulator(
        base_ttft_ms=150.0,
        time_per_token_ms=12.0,
        prompt_processing_rate_ms=0.4,
        context_growth_penalty=1.002,
    )


class TestAgentTrajectory:
    def test_trajectory_has_required_fields(self, sim: AgentSimulator) -> None:
        random.seed(1)
        result = sim.simulate_agent_trajectory("task_test_01", max_turns=3)
        assert "task_id" in result
        assert "trace_id" in result
        assert "num_turns" in result
        assert "total_latency_ms" in result
        assert "turns" in result

    def test_trajectory_num_turns_bounded(self, sim: AgentSimulator) -> None:
        random.seed(2)
        for max_turns in [1, 3, 5]:
            result = sim.simulate_agent_trajectory("t", max_turns=max_turns)
            assert result["num_turns"] <= max_turns

    def test_trajectory_total_latency_is_sum_of_turns(self, sim: AgentSimulator) -> None:
        random.seed(3)
        result = sim.simulate_agent_trajectory("t", max_turns=4)
        computed = sum(turn["turn_total_ms"] for turn in result["turns"])
        assert abs(result["total_latency_ms"] - computed) < 0.01

    def test_tool_latency_non_negative(self, sim: AgentSimulator) -> None:
        random.seed(4)
        result = sim.simulate_agent_trajectory("t", max_turns=6, tool_call_probability=1.0)
        for turn in result["turns"]:
            assert turn["tool_latency_ms"] >= 0.0

    def test_final_answer_turn_has_no_tool(self, sim: AgentSimulator) -> None:
        random.seed(5)
        result = sim.simulate_agent_trajectory("t", max_turns=6)
        final_turns = [t for t in result["turns"] if t["is_final_answer"]]
        assert len(final_turns) >= 1
        for t in final_turns:
            assert t["tool_called"] is None

    def test_tool_overhead_pct_bounded(self, sim: AgentSimulator) -> None:
        random.seed(6)
        result = sim.simulate_agent_trajectory("t", max_turns=5, tool_call_probability=0.9)
        assert 0.0 <= result["tool_overhead_pct"] <= 100.0

    def test_context_tokens_grows_across_turns(self, sim: AgentSimulator) -> None:
        random.seed(7)
        result = sim.simulate_agent_trajectory("t", max_turns=5, tool_call_probability=1.0)
        tokens = [t["context_tokens_after"] for t in result["turns"]]
        # Context should grow monotonically when tool calls happen
        for i in range(1, len(tokens)):
            assert tokens[i] >= tokens[i - 1]


class TestBenchmarkRun:
    def test_benchmark_has_aggregate_metrics(self, sim: AgentSimulator) -> None:
        random.seed(10)
        result = sim.run_benchmark("bench_test", num_tasks=20, max_turns_per_task=4)
        metrics = result["aggregate_metrics"]
        assert "total_latency_ms" in metrics
        assert "tool_latency_ms" in metrics
        assert "mean_turns_per_task" in metrics
        assert "mean_tool_overhead_pct" in metrics

    def test_benchmark_percentile_ordering(self, sim: AgentSimulator) -> None:
        random.seed(11)
        result = sim.run_benchmark("bench_test", num_tasks=30, max_turns_per_task=4)
        lat = result["aggregate_metrics"]["total_latency_ms"]
        assert lat["p50"] <= lat["p95"] <= lat["p99"]

    def test_benchmark_task_count(self, sim: AgentSimulator) -> None:
        random.seed(12)
        result = sim.run_benchmark("bench_test", num_tasks=10, max_turns_per_task=3)
        assert len(result["tasks"]) == 10

    def test_benchmark_reproducible_with_seed(self) -> None:
        random.seed(42)
        sim1 = AgentSimulator()
        r1 = sim1.run_benchmark("bench", num_tasks=5)

        random.seed(42)
        sim2 = AgentSimulator()
        r2 = sim2.run_benchmark("bench", num_tasks=5)

        assert r1["aggregate_metrics"]["total_latency_ms"]["p50"] == \
               r2["aggregate_metrics"]["total_latency_ms"]["p50"]

    def test_zero_tool_probability_no_tool_calls(self) -> None:
        random.seed(20)
        sim = AgentSimulator()
        result = sim.run_benchmark("bench", num_tasks=10, tool_call_probability=0.0)
        for task in result["tasks"]:
            assert task["total_tool_latency_ms"] == 0.0
            assert task["tool_overhead_pct"] == 0.0


class TestToolLatencyProfiles:
    def test_all_tools_have_required_fields(self) -> None:
        for name, profile in TOOL_LATENCY_PROFILES.items():
            assert "mean_ms" in profile, f"Tool '{name}' missing mean_ms"
            assert "std_ms" in profile, f"Tool '{name}' missing std_ms"
            assert profile["mean_ms"] > 0

    def test_tool_latency_positive(self) -> None:
        for tool in TOOL_LATENCY_PROFILES:
            # _tool_latency is a staticmethod, call it via the class directly
            lat = AgentSimulator._tool_latency(tool)  # noqa: SLF001
            assert lat > 0.0
