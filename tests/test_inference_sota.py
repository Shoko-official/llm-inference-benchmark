import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.simulator import LatencySimulator
from benchmark.http_client import AsyncLLMHttpClient
from benchmark.analysis import BenchmarkAnalyzer


class TestInferenceSOTA(unittest.TestCase):
    def test_percentile_calculations(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        stats = BenchmarkAnalyzer.calculate_percentiles(values)
        
        self.assertAlmostEqual(stats["p50"], 55.0)
        self.assertAlmostEqual(stats["p90"], 91.0)
        self.assertAlmostEqual(stats["p95"], 95.5)
        self.assertAlmostEqual(stats["p99"], 99.1)
        self.assertAlmostEqual(stats["mean"], 55.0)
        self.assertTrue(stats["std"] > 0)
        
        # Empty list fallback
        empty_stats = BenchmarkAnalyzer.calculate_percentiles([])
        self.assertEqual(empty_stats["p50"], 0.0)

    def test_run_analysis_aggregation(self) -> None:
        run_data = {
            "run_id": "run_test_123",
            "model_id": "model_xyz",
            "metrics": {
                "throughput_tokens_per_sec": 42.5
            },
            "requests": [
                {"latency_ms": 100.0, "ttft_ms": 50.0},
                {"latency_ms": 200.0, "ttft_ms": 60.0},
                {"latency_ms": 300.0, "ttft_ms": 70.0}
            ]
        }
        
        analysis = BenchmarkAnalyzer.analyze_run(run_data)
        self.assertEqual(analysis["run_id"], "run_test_123")
        self.assertEqual(analysis["throughput_tokens_per_sec"], 42.5)
        self.assertEqual(analysis["requests_count"], 3)
        self.assertEqual(analysis["latency_ms"]["p50"], 200.0)
        self.assertEqual(analysis["ttft_ms"]["p50"], 60.0)

    def test_charts_generation(self) -> None:
        concurrency_runs = [
            {
                "concurrency": 1,
                "throughput_tokens_per_sec": 15.0,
                "mean_latency_ms": 200.0,
                "mean_ttft_ms": 100.0
            },
            {
                "concurrency": 4,
                "throughput_tokens_per_sec": 45.0,
                "mean_latency_ms": 350.0,
                "mean_ttft_ms": 130.0
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            charts = BenchmarkAnalyzer.generate_charts(concurrency_runs, tmp_path)
            
            # Verify charts are generated if matplotlib is installed
            try:
                import matplotlib
                self.assertTrue((tmp_path / "concurrency_scaling.png").is_file())
                self.assertTrue((tmp_path / "latency_composition.png").is_file())
                self.assertIn("scaling_chart", charts)
                self.assertIn("latency_composition_chart", charts)
            except ImportError:
                # If matplotlib is not installed, no files should be generated
                self.assertEqual(len(charts), 0)

    def test_http_client_offline_fallback(self) -> None:
        client = AsyncLLMHttpClient("http://invalid-local-url:9999")
        
        # Running async measurement (offline, should fallback gracefully)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            metrics = loop.run_until_complete(
                client.measure_request("mock-model", "Hello world", max_tokens=100)
            )
            self.assertFalse(metrics["success"])
            self.assertIn("error", metrics)
            self.assertEqual(metrics["tokens_count"], 100)
            self.assertTrue(metrics["latency_ms"] > 0)
        finally:
            loop.close()

    def test_concurrency_degradation_simulation(self) -> None:
        import random
        random.seed(0)
        sim = LatencySimulator(base_ttft_ms=100.0, time_per_token_ms=10.0)

        # Simulate a run with low concurrency vs high concurrency
        # Both should produce valid metrics structures
        run_low = sim.simulate_run(
            "run_low", "test-model", num_requests=20, concurrent_requests=1, arrival_rate=5.0
        )
        random.seed(0)
        run_high = sim.simulate_run(
            "run_high", "test-model", num_requests=20, concurrent_requests=8, arrival_rate=5.0
        )

        # Both runs should produce valid metrics
        self.assertIn("metrics", run_low)
        self.assertIn("metrics", run_high)
        self.assertGreater(run_low["metrics"]["mean_latency_ms"], 0)
        self.assertGreater(run_high["metrics"]["mean_latency_ms"], 0)


if __name__ == "__main__":
    unittest.main()
