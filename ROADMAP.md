# Roadmap

This roadmap starts with governance only. Inference content starts after
Milestone 0 is complete and the repository has validation and CI.

## Milestone 0: Project Governance

Goal: establish a small, reviewable operating model for the inference benchmark repository.

### Issues

1. [#1 Create inference benchmark governance documentation](https://github.com/Shoko-official/llm-inference-benchmark/issues/1)
2. [#2 Add issue and PR templates](https://github.com/Shoko-official/llm-inference-benchmark/issues/2)
3. [#3 Add minimal validation, CI, and folder structure](https://github.com/Shoko-official/llm-inference-benchmark/issues/3)

### Execution Order

1. Complete issue #1 before templates or validation.
2. Complete issue #2 before content changes.
3. Complete issue #3 before inference content.

No inference content should be added during Milestone 0.

## Acceptance Criteria

Milestone 0 is complete when:

* repository role is documented;
* roadmap exists;
* contribution and review rules exist;
* issue and PR templates exist;
* minimal validation commands exist;
* minimal CI exists;
* initial folders exist;
* no inference content has been added.

## Later Milestones

Expected sequence:

1. Define a minimal inference metric and benchmark logging schema.
2. Implement latency and TTFT mock generators.
3. Add throughput estimation and request concurrency simulators.
4. Implement validation rules checking benchmark profiles.
5. Create validation for benchmark output datasets.
6. Connect benchmark results to evaluation harness metrics.

Any change to inference structures, benchmark models, or evidence rules must happen in a
dedicated issue.
