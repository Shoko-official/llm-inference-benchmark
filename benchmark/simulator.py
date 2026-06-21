from __future__ import annotations

import random
from datetime import datetime, timezone

class LatencySimulator:
    def __init__(
        self,
        base_ttft_ms: float = 150.0,
        time_per_token_ms: float = 12.0,
        prompt_processing_rate_ms: float = 0.4
    ):
        self.base_ttft_ms = base_ttft_ms
        self.time_per_token_ms = time_per_token_ms
        self.prompt_processing_rate_ms = prompt_processing_rate_ms

    def simulate_request(self, request_id: str, prompt_tokens: int, completion_tokens: int) -> dict:
        # TTFT: base + prompt processing + small noise
        noise_ttft = random.uniform(5.0, 50.0)
        ttft_ms = self.base_ttft_ms + (prompt_tokens * self.prompt_processing_rate_ms) + noise_ttft

        # Latency: TTFT + generation time + small noise
        noise_gen = random.uniform(10.0, 100.0)
        generation_time_ms = completion_tokens * self.time_per_token_ms + noise_gen
        latency_ms = ttft_ms + generation_time_ms

        return {
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": round(latency_ms, 2),
            "ttft_ms": round(ttft_ms, 2)
        }

    def simulate_run(
        self,
        run_id: str,
        model_id: str,
        num_requests: int = 50,
        concurrent_requests: int = 4
    ) -> dict:
        requests = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0.0
        total_ttft = 0.0

        for i in range(num_requests):
            req_id = f"req_{i:03d}"
            prompt_tokens = random.randint(32, 256)
            completion_tokens = random.randint(64, 512)
            
            req_data = self.simulate_request(req_id, prompt_tokens, completion_tokens)
            requests.append(req_data)
            
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_latency += req_data["latency_ms"]
            total_ttft += req_data["ttft_ms"]

        mean_latency = total_latency / num_requests if num_requests else 0.0
        mean_ttft = total_ttft / num_requests if num_requests else 0.0

        # Throughput = total generated tokens / total simulated generation wall time
        # Simulated wall time with concurrency: approximate total simulated time
        # as total generation time / concurrency.
        # Wall time = total_latency_ms / 1000 / concurrency
        approx_wall_time = (total_latency / 1000.0) / concurrent_requests
        throughput = total_completion_tokens / approx_wall_time if approx_wall_time > 0 else 0.0

        return {
            "run_id": run_id,
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metrics": {
                "throughput_tokens_per_sec": round(throughput, 2),
                "mean_latency_ms": round(mean_latency, 2),
                "mean_ttft_ms": round(mean_ttft, 2),
                "concurrent_requests": concurrent_requests
            },
            "requests": requests
        }
