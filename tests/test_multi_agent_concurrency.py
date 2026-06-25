"""Tests for the multi-agent concurrency benchmark (M8-INFER-02)."""
from __future__ import annotations

import random
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.run_multi_agent_concurrency import run_concurrency_sweep


class TestConcurrencySweep:
    def test_output_has_required_fields(self) -> None:
        random.seed(0)
        result = run_concurrency_sweep([1, 2], tasks_per_agent=5, max_turns=2, seed=0)
        assert "benchmark_id" in result
        assert "timestamp" in result
        assert "config" in result
        assert "results" in result

    def test_result_count_matches_levels(self) -> None:
        random.seed(1)
        levels = [1, 2, 4]
        result = run_concurrency_sweep(levels, tasks_per_agent=5, seed=1)
        assert len(result["results"]) == len(levels)

    def test_n_agents_matches_level(self) -> None:
        random.seed(2)
        levels = [1, 4, 8]
        result = run_concurrency_sweep(levels, tasks_per_agent=5, seed=2)
        for entry, expected_n in zip(result["results"], levels):
            assert entry["n_agents"] == expected_n

    def test_latency_percentile_ordering(self) -> None:
        random.seed(3)
        result = run_concurrency_sweep([1, 2, 4], tasks_per_agent=10, seed=3)
        for entry in result["results"]:
            lat = entry["latency_ms"]
            assert lat["p50"] <= lat["p95"] <= lat["p99"]

    def test_baseline_has_zero_degradation(self) -> None:
        random.seed(4)
        result = run_concurrency_sweep([1, 2, 4], tasks_per_agent=10, seed=4)
        assert result["results"][0]["throughput_degradation_pct"] == 0.0

    def test_contention_factor_increases_with_agents(self) -> None:
        random.seed(5)
        result = run_concurrency_sweep([1, 2, 4, 8], tasks_per_agent=5, seed=5)
        factors = [r["contention_factor"] for r in result["results"]]
        for i in range(1, len(factors)):
            assert factors[i] > factors[i - 1]

    def test_tool_latency_present(self) -> None:
        random.seed(6)
        result = run_concurrency_sweep([1, 2], tasks_per_agent=5, seed=6)
        for entry in result["results"]:
            assert "tool_latency_ms" in entry
            assert "p50" in entry["tool_latency_ms"]

    def test_single_agent_baseline(self) -> None:
        """Single agent should have no contention penalty."""
        random.seed(7)
        result = run_concurrency_sweep([1], tasks_per_agent=10, seed=7)
        assert result["results"][0]["contention_factor"] == 1.0
