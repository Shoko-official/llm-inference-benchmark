import json
import sys
import tempfile
import unittest
import subprocess
from pathlib import Path
from jsonschema import validate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class TestSimulatorCLI(unittest.TestCase):
    def test_cli_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_file = tmp_path / "simulated_run.json"
            
            # Run simulate_inference.py via subprocess
            cli_path = ROOT / "scripts" / "simulate_inference.py"
            res = subprocess.run([
                sys.executable,
                str(cli_path),
                "--run-id", "test_cli_run",
                "--model-id", "test_model",
                "--num-requests", "10",
                "--concurrency", "2",
                "--output", str(output_file)
            ], capture_output=True, text=True)
            
            self.assertEqual(res.returncode, 0, f"CLI failed: {res.stderr}\n{res.stdout}")
            self.assertTrue(output_file.is_file())
            
            # Load and validate
            with open(output_file, "r", encoding="utf-8") as f:
                run_data = json.load(f)
                
            schema_path = ROOT / "benchmark" / "schemas" / "inference_run.json"
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
                
            validate(instance=run_data, schema=schema)
            self.assertEqual(run_data["run_id"], "test_cli_run")
            self.assertEqual(run_data["model_id"], "test_model")
            self.assertEqual(len(run_data["requests"]), 10)

if __name__ == "__main__":
    unittest.main()
