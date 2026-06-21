import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.simulator import LatencySimulator

class TestLatencySimulator(unittest.TestCase):
    def test_simulator_run_structure(self) -> None:
        sim = LatencySimulator()
        run_data = sim.simulate_run("run_test", "model_test", num_requests=5, concurrent_requests=2)
        
        self.assertEqual(run_data["run_id"], "run_test")
        self.assertEqual(run_data["model_id"], "model_test")
        self.assertEqual(len(run_data["requests"]), 5)
        self.assertEqual(run_data["metrics"]["concurrent_requests"], 2)
        
        for req in run_data["requests"]:
            self.assertTrue(req["request_id"].startswith("req_"))
            self.assertTrue(req["prompt_tokens"] > 0)
            self.assertTrue(req["completion_tokens"] > 0)
            self.assertTrue(req["latency_ms"] > req["ttft_ms"])
            self.assertTrue(req["ttft_ms"] > 0)

    def test_simulator_parameters(self) -> None:
        sim_low = LatencySimulator(base_ttft_ms=10.0, time_per_token_ms=1.0)
        sim_high = LatencySimulator(base_ttft_ms=1000.0, time_per_token_ms=50.0)
        
        req_low = sim_low.simulate_request("req_low", 10, 10)
        req_high = sim_high.simulate_request("req_high", 10, 10)
        
        self.assertTrue(req_high["ttft_ms"] > req_low["ttft_ms"])
        self.assertTrue(req_high["latency_ms"] > req_low["latency_ms"])

    def test_concurrency_scaling(self) -> None:
        sim = LatencySimulator(base_ttft_ms=50.0, time_per_token_ms=5.0)
        
        # Heavy load (50 requests arriving at high rate)
        # Verify that higher concurrency results in higher overall throughput
        run_low = sim.simulate_run("low_concurrency", "model_a", num_requests=20, concurrent_requests=1, arrival_rate=100.0)
        run_high = sim.simulate_run("high_concurrency", "model_a", num_requests=20, concurrent_requests=4, arrival_rate=100.0)
        
        t_low = run_low["metrics"]["throughput_tokens_per_sec"]
        t_high = run_high["metrics"]["throughput_tokens_per_sec"]
        
        print(f"\nThroughput (concurrency=1): {t_low} tokens/sec")
        print(f"Throughput (concurrency=4): {t_high} tokens/sec")
        
        self.assertTrue(t_high > t_low, f"Expected higher concurrency to scale throughput. Got: high={t_high}, low={t_low}")

if __name__ == "__main__":
    unittest.main()
