# Contributing to RAGHub

RAGHub welcomes focused bug fixes, tests, documentation improvements, and
retrieval experiments whose claims are backed by reproducible evidence.

## Development setup

Requirements are Go 1.25 or newer, Docker with Compose, and optionally an
OpenAI-compatible 1024-dimensional embedding endpoint for Dense and Hybrid
evaluation.

```bash
make db-up
make verify
make test-integration
```

Before submitting a pull request, run:

```bash
make verify-release
```

Changes to frozen datasets require a new dataset version. Do not alter existing
queries, gold references, or frozen bytes in response to observed rankings.
Retrieval-quality claims must name the dataset hash, clean Git revision,
configuration, provider limitations, and exact command used.

## Pull requests

- Keep each pull request scoped to one reviewable change.
- Add or update tests for behavior changes.
- Update API, README, ADR, and experiment documentation when contracts change.
- Do not commit generated evaluation manifests containing private queries or
  documents.
- Confirm that the branch contains no credentials, customer data, or private
  infrastructure addresses.

By submitting a contribution, you agree that it is licensed under Apache-2.0.
