from __future__ import annotations

import random
import heapq
from datetime import datetime, timezone, timedelta
import uuid
from typing import List, Dict, Any, Optional

class LatencySimulator:
    def __init__(
        self,
        base_ttft_ms: float = 150.0,
        time_per_token_ms: float = 12.0,
        prompt_processing_rate_ms: float = 0.4
    ) -> None:
        self.base_ttft_ms = base_ttft_ms
        self.time_per_token_ms = time_per_token_ms
        self.prompt_processing_rate_ms = prompt_processing_rate_ms
        self.last_spans: List[Dict[str, Any]] = []

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
        concurrent_requests: int = 4,
        arrival_rate: float = 10.0
    ) -> dict:
        """
        Simulate a concurrent benchmark run using a discrete-event queue scheduler.
        """
        # Event type constants
        ARRIVAL = 0
        COMPLETION = 1

        # Generate requests and their arrivals (Poisson process)
        raw_requests = []
        events = []
        t = 0.0
        for i in range(num_requests):
            req_id = f"req_{i:03d}"
            prompt_tokens = random.randint(32, 256)
            completion_tokens = random.randint(64, 512)
            
            # Poisson arrival intervals
            interval = random.expovariate(arrival_rate) if arrival_rate > 0 else 0.0
            t += interval
            
            req_data = {
                "request_id": req_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "arrival_time": t
            }
            raw_requests.append(req_data)
            heapq.heappush(events, (t, ARRIVAL, req_id))

        # Simulation states
        active_count = 0
        waiting_queue = [] # Queue of (arrival_time, req_id)
        completed_requests = []
        
        req_lookup = {r["request_id"]: r for r in raw_requests}
        
        start_wall_time = None
        end_wall_time = 0.0
        
        while events:
            ev_time, ev_type, req_id = heapq.heappop(events)
            
            if ev_type == ARRIVAL:
                if active_count < concurrent_requests:
                    active_count += 1
                    req = req_lookup[req_id]
                    
                    # Run request
                    sim = self.simulate_request(req_id, req["prompt_tokens"], req["completion_tokens"])
                    sim["arrival_time"] = round(req["arrival_time"], 4)
                    sim["start_time"] = round(ev_time, 4)
                    sim["queue_time_ms"] = 0.0
                    
                    if start_wall_time is None:
                        start_wall_time = ev_time
                    
                    # Schedule completion
                    latency_sec = sim["latency_ms"] / 1000.0
                    heapq.heappush(events, (ev_time + latency_sec, COMPLETION, req_id))
                    completed_requests.append(sim)
                else:
                    req = req_lookup[req_id]
                    waiting_queue.append((req["arrival_time"], req_id))
            
            elif ev_type == COMPLETION:
                end_wall_time = max(end_wall_time, ev_time)
                
                if waiting_queue:
                    # Start next queued request
                    arr_time, next_req_id = waiting_queue.pop(0)
                    req = req_lookup[next_req_id]
                    
                    # Calculate queue delay
                    queue_time_sec = ev_time - arr_time
                    sim = self.simulate_request(next_req_id, req["prompt_tokens"], req["completion_tokens"])
                    sim["arrival_time"] = round(arr_time, 4)
                    sim["start_time"] = round(ev_time, 4)
                    sim["queue_time_ms"] = round(queue_time_sec * 1000.0, 2)
                    
                    # Schedule completion
                    latency_sec = sim["latency_ms"] / 1000.0
                    heapq.heappush(events, (ev_time + latency_sec, COMPLETION, next_req_id))
                    completed_requests.append(sim)
                else:
                    active_count -= 1

        # Calculate metrics
        wall_time_sec = end_wall_time - (start_wall_time or 0.0)
        total_completion_tokens = sum(r["completion_tokens"] for r in raw_requests)
        
        # Real throughput calculation
        throughput = total_completion_tokens / wall_time_sec if wall_time_sec > 0 else 0.0
        
        total_latency = sum(r["latency_ms"] for r in completed_requests)
        total_ttft = sum(r["ttft_ms"] for r in completed_requests)
        
        mean_latency = total_latency / num_requests if num_requests else 0.0
        mean_ttft = total_ttft / num_requests if num_requests else 0.0

        # Format final output conforming to JSON schema
        cleaned_requests = []
        for req in completed_requests:
            cleaned_requests.append({
                "request_id": req["request_id"],
                "prompt_tokens": req["prompt_tokens"],
                "completion_tokens": req["completion_tokens"],
                "latency_ms": req["latency_ms"],
                "ttft_ms": req["ttft_ms"]
            })

        run_data = {
            "run_id": run_id,
            "model_id": model_id,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metrics": {
                "throughput_tokens_per_sec": round(throughput, 2),
                "mean_latency_ms": round(mean_latency, 2),
                "mean_ttft_ms": round(mean_ttft, 2),
                "concurrent_requests": concurrent_requests
            },
            "requests": cleaned_requests
        }

        # Generate spans matching core span.json schema
        base_time = datetime.fromisoformat(run_data["timestamp"].replace("Z", "+00:00"))
        spans = []
        for req in completed_requests:
            trace_id = uuid.uuid4().hex
            
            # Queue span
            queue_span_id = uuid.uuid4().hex[:16]
            queue_start = base_time + timedelta(seconds=req["arrival_time"])
            queue_end = base_time + timedelta(seconds=req["start_time"])
            
            queue_span = {
                "span_id": queue_span_id,
                "trace_id": trace_id,
                "parent_span_id": "N/A",
                "name": "request_queue_latency",
                "start_time": queue_start.isoformat().replace("+00:00", "Z"),
                "end_time": queue_end.isoformat().replace("+00:00", "Z"),
                "duration_ms": round(req.get("queue_time_ms", 0.0), 2),
                "service_name": "infer",
                "status": "ok",
                "attributes": {
                    "request_id": req["request_id"]
                }
            }
            spans.append(queue_span)
            
            # Generation span
            gen_span_id = uuid.uuid4().hex[:16]
            gen_start = queue_end
            gen_end = gen_start + timedelta(milliseconds=req["latency_ms"])
            
            gen_span = {
                "span_id": gen_span_id,
                "trace_id": trace_id,
                "parent_span_id": queue_span_id,
                "name": "generate_tokens",
                "start_time": gen_start.isoformat().replace("+00:00", "Z"),
                "end_time": gen_end.isoformat().replace("+00:00", "Z"),
                "duration_ms": round(req["latency_ms"], 2),
                "service_name": "infer",
                "status": "ok",
                "attributes": {
                    "request_id": req["request_id"],
                    "prompt_tokens": req["prompt_tokens"],
                    "completion_tokens": req["completion_tokens"]
                }
            }
            spans.append(gen_span)

        self.last_spans = spans
        return run_data
