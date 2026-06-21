# Concurrency Simulator Integration

## Overview

The `benchmark/simulator.py` module provides event-driven request concurrency
simulation and throughput estimation. It integrates with the validation
pipeline and the CLI entrypoint.

## Architecture

```
scripts/simulate_inference.py   <- CLI entrypoint
         |
         v
benchmark/simulator.py          <- LatencySimulator (core logic)
         |
         v
benchmark/schemas/inference_run.json  <- output schema validation
```

## Integration Points

| Component | File | Role |
|-----------|------|------|
| CLI entrypoint | `scripts/simulate_inference.py` | Parses arguments, calls simulator, outputs JSON |
| Simulator | `benchmark/simulator.py` | Event-driven queue, throughput calculation |
| Schema | `benchmark/schemas/inference_run.json` | Validates output structure |
| Validation | `scripts/validate_inference.py` | Validates `benchmark/mock_run.json` |
| Repo validator | `scripts/validate_repo.py` | Runs all checks end-to-end |

## Running

```bash
# Run the full validation + test suite
python scripts/validate_repo.py

# Simulate inference directly
python scripts/simulate_inference.py --concurrency 4 --tokens 512 --requests 100

# Validate mock run data
python scripts/validate_inference.py
```

## Concurrency Scaling

The simulator models queue-based throughput scaling:
- At `concurrency=1` baseline throughput is established.
- Scaling factor is sublinear (approx. `concurrency^0.9`).
- Output metrics follow the `inference_run.json` schema strictly.

## Acceptance Criteria (M3-INFER-04)

- [x] `python scripts/validate_repo.py` exits 0
- [x] All 8 unit tests pass (`pytest`)
- [x] CLI (`simulate_inference.py`) produces schema-valid JSON
- [x] Concurrency scaling verified in `test_concurrency_scaling`
