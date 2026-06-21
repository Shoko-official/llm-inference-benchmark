from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VALID_PROFILE = {
    "profile_id": "test_profile",
    "description": "Test profile for unit tests.",
    "model_id": "test-model",
    "prompt_config": {
        "min_prompt_tokens": 32,
        "max_prompt_tokens": 256,
        "min_completion_tokens": 64,
        "max_completion_tokens": 512
    },
    "concurrency_config": {
        "concurrent_requests": 4,
        "total_requests": 50
    },
    "latency_config": {
        "base_ttft_ms": 150.0,
        "time_per_token_ms": 12.0
    }
}

SCHEMA_PATH = ROOT / "benchmark" / "schemas" / "benchmark_profile.json"


class TestBenchmarkProfiles(unittest.TestCase):

    def _write_tmp_profile(self, data: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        )
        json.dump(data, tmp)
        tmp.close()
        return Path(tmp.name)

    def _run_validator(self, profile_path: str) -> tuple[int, str, str]:
        import subprocess
        script = ROOT / "scripts" / "validate_profiles.py"
        res = subprocess.run(
            [sys.executable, str(script), "--profile", profile_path],
            capture_output=True, text=True
        )
        return res.returncode, res.stdout, res.stderr

    def test_valid_profile(self) -> None:
        path = self._write_tmp_profile(VALID_PROFILE)
        try:
            rc, out, err = self._run_validator(str(path))
            self.assertEqual(rc, 0, f"Validator failed: {err}")
            self.assertIn("Successfully validated", out)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_required_field(self) -> None:
        bad = dict(VALID_PROFILE)
        del bad["model_id"]
        path = self._write_tmp_profile(bad)
        try:
            rc, out, err = self._run_validator(str(path))
            self.assertNotEqual(rc, 0, "Should fail on missing model_id")
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_prompt_range(self) -> None:
        bad = json.loads(json.dumps(VALID_PROFILE))
        bad["prompt_config"]["min_prompt_tokens"] = 512
        bad["prompt_config"]["max_prompt_tokens"] = 32
        path = self._write_tmp_profile(bad)
        try:
            rc, out, err = self._run_validator(str(path))
            self.assertNotEqual(rc, 0, "Should fail when min_prompt > max_prompt")
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_concurrency_range(self) -> None:
        bad = json.loads(json.dumps(VALID_PROFILE))
        bad["concurrency_config"]["concurrent_requests"] = 100
        bad["concurrency_config"]["total_requests"] = 10
        path = self._write_tmp_profile(bad)
        try:
            rc, out, err = self._run_validator(str(path))
            self.assertNotEqual(rc, 0, "Should fail when concurrent > total")
        finally:
            path.unlink(missing_ok=True)

    def test_sample_profiles_valid(self) -> None:
        """All shipped sample profiles must pass validation."""
        profiles_dir = ROOT / "benchmark" / "profiles"
        self.assertTrue(profiles_dir.is_dir(), "profiles/ dir missing")
        profile_files = list(profiles_dir.rglob("*.json"))
        self.assertGreater(len(profile_files), 0, "No sample profiles found")
        for pf in profile_files:
            with self.subTest(profile=pf.name):
                rc, out, err = self._run_validator(str(pf))
                self.assertEqual(rc, 0, f"Sample profile {pf.name} invalid: {err}")

    def test_concurrency_scaling_profile(self) -> None:
        """high_concurrency profile has concurrent_requests >= 8."""
        profile_path = ROOT / "benchmark" / "profiles" / "profile_high_concurrency.json"
        self.assertTrue(profile_path.is_file(), "high_concurrency profile missing")
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertGreaterEqual(
            data["concurrency_config"]["concurrent_requests"], 8,
            "High-concurrency profile should have >= 8 concurrent requests"
        )


if __name__ == "__main__":
    unittest.main()
