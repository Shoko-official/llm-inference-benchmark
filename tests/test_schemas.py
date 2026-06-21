import json
import sys
import unittest
from pathlib import Path
from jsonschema import validate, ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class TestInferenceSchemas(unittest.TestCase):
    def setUp(self) -> None:
        self.schema_path = ROOT / "benchmark" / "schemas" / "inference_run.json"
        with open(self.schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def test_valid_mock_run(self) -> None:
        mock_path = ROOT / "benchmark" / "mock_run.json"
        self.assertTrue(mock_path.is_file())
        with open(mock_path, "r", encoding="utf-8") as f:
            mock_data = json.load(f)
        validate(instance=mock_data, schema=self.schema)

    def test_invalid_run_missing_required(self) -> None:
        invalid_run = {
            "run_id": "RUN-001",
            "model_id": "model-xyz",
            "metrics": {} # missing fields
        }
        with self.assertRaises(ValidationError):
            validate(instance=invalid_run, schema=self.schema)

    def test_invalid_run_extra_field(self) -> None:
        invalid_run = {
            "run_id": "RUN-001",
            "model_id": "model-xyz",
            "metrics": {
                "throughput_tokens_per_sec": 45.2,
                "mean_latency_ms": 1200.5,
                "mean_ttft_ms": 250.2,
                "concurrent_requests": 4
            },
            "requests": [],
            "extra_field": "not_allowed"
        }
        with self.assertRaises(ValidationError):
            validate(instance=invalid_run, schema=self.schema)

    def test_invalid_metric_value(self) -> None:
        invalid_run = {
            "run_id": "RUN-001",
            "model_id": "model-xyz",
            "metrics": {
                "throughput_tokens_per_sec": 45.2,
                "mean_latency_ms": -10.0, # invalid negative
                "mean_ttft_ms": 250.2,
                "concurrent_requests": 4
            },
            "requests": []
        }
        with self.assertRaises(ValidationError):
            validate(instance=invalid_run, schema=self.schema)

if __name__ == "__main__":
    unittest.main()
