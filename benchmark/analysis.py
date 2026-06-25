"""
analysis.py — Statistical analysis and visualization for inference benchmarks.
Calculates percentiles and generates performance charts using matplotlib.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

class BenchmarkAnalyzer:
    """
    Analyzes benchmark runs, calculating key performance metrics (percentiles, throughput)
    and generating visualization charts.
    """
    @staticmethod
    def calculate_percentiles(values: List[float]) -> Dict[str, float]:
        """Calculates standard percentiles (P50, P90, P95, P99) for a list of values."""
        if not values:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "std": 0.0}
            
        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
        }

    @staticmethod
    def analyze_run(run_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes details of a single benchmark run."""
        requests = run_data.get("requests", [])
        latencies = [r["latency_ms"] for r in requests if "latency_ms" in r]
        ttfts = [r["ttft_ms"] for r in requests if "ttft_ms" in r]
        
        latency_stats = BenchmarkAnalyzer.calculate_percentiles(latencies)
        ttft_stats = BenchmarkAnalyzer.calculate_percentiles(ttfts)
        
        return {
            "run_id": run_data.get("run_id"),
            "model_id": run_data.get("model_id"),
            "throughput_tokens_per_sec": run_data.get("metrics", {}).get("throughput_tokens_per_sec", 0.0),
            "latency_ms": latency_stats,
            "ttft_ms": ttft_stats,
            "requests_count": len(requests)
        }

    @staticmethod
    def generate_charts(
        concurrency_runs: List[Dict[str, Any]],
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Generates throughput-scaling and latency charts and saves them to disk.
        concurrency_runs contains runs at different concurrency levels.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Try importing matplotlib safely
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            logger = logging.getLogger("analysis")
            logger.warning("matplotlib not available. Skipping chart generation.")
            return {}
            
        # Sort runs by concurrency
        runs = sorted(concurrency_runs, key=lambda x: x.get("concurrency", 0))
        concurrencies = [r.get("concurrency", 0) for r in runs]
        throughputs = [r.get("throughput_tokens_per_sec", 0.0) for r in runs]
        
        mean_latencies = [r.get("mean_latency_ms", 0.0) for r in runs]
        mean_ttfts = [r.get("mean_ttft_ms", 0.0) for r in runs]
        
        # 1. Generate Concurrency vs Throughput & Latency Chart
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        color = "tab:blue"
        ax1.set_xlabel("Concurrency (Simultaneous Request Workers)")
        ax1.set_ylabel("Throughput (Tokens/sec)", color=color)
        ax1.plot(concurrencies, throughputs, marker="o", color=color, linewidth=2, label="Throughput")
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.6)
        
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Latency / TTFT (ms)", color=color)
        ax2.plot(concurrencies, mean_latencies, marker="s", linestyle="--", color=color, linewidth=1.5, label="End-to-End Latency")
        ax2.plot(concurrencies, mean_ttfts, marker="^", linestyle=":", color="tab:orange", linewidth=1.5, label="TTFT")
        ax2.tick_params(axis="y", labelcolor=color)
        
        plt.title("LLM Inference Performance Scaling Profile")
        fig.tight_layout()
        
        scaling_chart_path = output_dir / "concurrency_scaling.png"
        plt.savefig(scaling_chart_path, dpi=150)
        plt.close()
        
        # 2. Latency Share Pie/Bar Chart for typical run (using first run if available)
        fig, ax = plt.subplots(figsize=(8, 4))
        if runs:
            target_run = runs[-1] # Highest concurrency run
            categories = ["TTFT (Time to First Token)", "Generation (Remaining Tokens)"]
            ttft_avg = target_run.get("mean_ttft_ms", 150.0)
            gen_avg = max(0.0, target_run.get("mean_latency_ms", 500.0) - ttft_avg)
            
            ax.pie(
                [ttft_avg, gen_avg],
                labels=categories,
                autopct="%1.1f%%",
                startangle=140,
                colors=["#ff9999", "#66b3ff"],
                explode=(0.05, 0)
            )
            ax.set_title(f"Latency Composition (Concurrency = {target_run.get('concurrency')})")
            
        latency_composition_path = output_dir / "latency_composition.png"
        plt.savefig(latency_composition_path, dpi=150)
        plt.close()
        
        return {
            "scaling_chart": scaling_chart_path,
            "latency_composition_chart": latency_composition_path
        }
