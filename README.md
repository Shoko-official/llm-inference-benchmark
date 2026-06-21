# LLM Inference Benchmark

`llm-inference-benchmark` defines the serving and inference benchmark workspace for the
Modern LLM Systems 2026 / arXiv Report program.

This repository will organize latency, throughput, TTFT (time-to-first-token), and concurrent request evaluation metrics.

It is not the research ledger, paper repository, RAG implementation, evaluation harness, agent runtime, memory layer, or observability stack.

## Repository Role

This repository owns:

* inference benchmark governance;
* serving benchmarking scripts;
* throughput and TTFT simulation hooks;
* benchmark validation and CI once introduced;
* paper-facing inference tables and evaluation support when backed by approved evidence.

The central project board is:

* [Modern LLM Systems 2026 / arXiv Report](https://github.com/users/Shoko-official/projects/4)

## Current Scope

Milestone 0 is limited to governance.

Included:

* repository scope;
* roadmap;
* contribution rules;
* review rules.

Out of scope:

* production serving setups;
* custom execution runtimes;
* hardware configuration tuning;
* final paper drafting;
* agent memory, RAG indexing, or evaluation dataset definition.

## Evidence Policy

Future benchmark claims must reference approved research ledger material or stay
clearly marked as unresolved planning notes.

Unsupported claims must not be used as paper-ready benchmark content.

## Figure Policy

Allowed source formats:

* Mermaid text diagrams for workflows, architecture maps, dependency graphs, and
  concept maps.
* Python-generated images for visualizations that are not practical in Mermaid.

Not allowed by default:

* web images;
* screenshots unless explicitly approved;
* hand-drawn images;
* Figma, Canva, or PowerPoint exports;
* manually authored complex SVGs;
* binary figures without clear source;
* orphan figures.

## License

This repository is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
