# RAGHub

[![CI](https://github.com/ksana-ai/raghub/actions/workflows/ci.yml/badge.svg)](https://github.com/ksana-ai/raghub/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

> **Experimental preview:** RAGHub is suitable for local evaluation and
> synthetic-data experiments. It is not production-ready and must not be
> exposed directly to untrusted networks.

RAGHub is a Go-native retrieval platform built as an evidence-backed project,
not a chat-demo wrapper. The current repository implements three independently
measurable retrieval baselines:

`Markdown ingestion -> versioned PostgreSQL + pgvector storage -> FTS, exact Dense, or fail-closed Hybrid RRF -> tenant/ACL filtering -> citations -> three-way offline evaluation`

## Current status

Implemented in this slice:

- deterministic, heading-aware Markdown chunking;
- immutable document versions with idempotent re-ingestion;
- separate `raw_text` and `indexed_text` fields;
- PostgreSQL weighted full-text search with a GIN index;
- a real batched OpenAI-compatible embedding client, configured by default for
  LM Studio `text-embedding-bge-m3` at 1024 dimensions;
- immutable embedding profiles with explicit provider/model/dimension and
  document/query recipe provenance;
- atomic vector writes before document-version activation, plus idempotent
  vector backfill for an existing FTS-ingested version;
- exact pgvector cosine search over a materialized authorized/current-version
  set;
- explicit `fts`, `dense`, and `hybrid` modes with query-embedding, database,
  and fusion traces;
- fail-closed reciprocal rank fusion over independently authorized FTS and
  Dense candidate sets, preregistered with candidate floors 20/20 and
  `rrf_k=60`;
- current-version, tenant, and principal ACL filters in the SQL query;
- exact document-version/chunk/source references in every hit;
- per-stage score/rank and latency traces;
- versioned lexical, paired, and preregistered hybrid smoke datasets;
- a frozen 50-query hard benchmark with multi-gold, ACL, and tenant cases;
- exact internal FTS/Dense candidate evidence for every Hybrid evaluation
  request, including forbidden-candidate safety checks;
- strict JSON evaluation plus backward-compatible pairwise and strict
  FTS/Dense/Hybrid comparison and candidate-diagnosis manifests;
- HitRate@K, standard Recall@K, MRR, binary nDCG@K, p50, and p95;
- unit and opt-in PostgreSQL integration tests.

Not implemented yet:

- approximate nearest-neighbor indexes or production-scale performance proof;
- reranking;
- answer generation, citation verification, or an LLM judge;
- Contextual Retrieval, Agentic RAG, RAPTOR, or GraphRAG;
- production authentication, deployment, or production performance evidence.

The distinction matters: the current result is an FTS, exact Dense, and Hybrid
RRF smoke baseline, not a production RAG system.

## Quick start

Requirements:

- Go 1.25 or newer; the module recommends the security-patched Go 1.27.1
  toolchain;
- Docker with Compose.
- LM Studio serving `text-embedding-bge-m3` through the OpenAI-compatible
  `/v1/embeddings` API for ingestion, Dense search, and Hybrid search.

Start PostgreSQL:

```bash
docker compose up -d postgres
export RAGHUB_DATABASE_URL='postgres://raghub:raghub@localhost:55432/raghub?sslmode=disable'
export RAGHUB_EMBEDDING_ENDPOINT='http://127.0.0.1:1234/v1/embeddings'
```

Alternatively, with LM Studio already listening on the host, build and start
both the API and PostgreSQL in containers:

```bash
docker compose --profile app up --build
```

The container setup maps the API to `http://localhost:8080` and reaches the
embedding server through `host.docker.internal`. The development database
credentials in `compose.yaml` are intentionally local-only and must not be
reused in shared or production environments.

If port `55432` is occupied, set `RAGHUB_POSTGRES_PORT` for Compose and use the
same port in `RAGHUB_DATABASE_URL`.

Start the API and apply the embedded migration:

```bash
go run ./cmd/raghub-api -migrate
```

In another terminal, ingest a tenant-wide Markdown document:

```bash
curl --fail-with-body \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{
    "document_id": "deployment-guide",
    "title": "Deployment Guide",
    "source_uri": "https://docs.example.test/deployment",
    "content": "# Deployment\n\nUse blue-green deployment for zero-downtime releases.",
    "metadata": {"team": "platform"}
  }' \
  http://localhost:8080/v1/documents
```

Search it with the lexical baseline:

```bash
curl --fail-with-body \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"query":"blue green deployment","top_k":5,"mode":"fts"}' \
  http://localhost:8080/v1/search
```

Run the Dense baseline by changing `mode`:

```bash
curl --fail-with-body \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"query":"How can releases avoid an outage?","top_k":5,"mode":"dense"}' \
  http://localhost:8080/v1/search
```

Run the fixed Hybrid RRF baseline with `mode=hybrid`:

```bash
curl --fail-with-body \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"query":"How can releases avoid an outage?","top_k":5,"mode":"hybrid"}' \
  http://localhost:8080/v1/search
```

Hybrid requests `max(top_k, 20)` authorized candidates from each branch,
applies RRF with `k=60`, and returns FTS, Dense, and RRF ranks/scores. If either
retrieval branch fails, the whole Hybrid request fails rather than silently
changing the retrieval protocol.

`X-Tenant-ID` and `X-Principal-ID` are development authorization context.
They are deliberately outside the JSON payload, but they are still spoofable
headers until a trusted authentication layer derives them from credentials.
`/readyz` checks both PostgreSQL and the configured embedding model; successful
process liveness alone is available at `/healthz`.

## Evaluation

The preregistered benchmark v1 contains 50 fixed queries over 44 documents,
including eight two-gold cases. It covers exact disambiguation, semantic
paraphrases, Chinese-English retrieval, near-duplicate distractors,
multi-relevant recall, ACL filtering, and tenant isolation. Its frozen
exact-byte SHA-256 is
`aa44175b9ae656d97473a8340ebac59bc1432d7cee90e51432c2b4f89e61f85f`.

After committing benchmark and evaluator changes, run the complete clean-
revision protocol:

```bash
make eval-benchmark
```

This produces FTS, Dense, and Hybrid Top-5 manifests and their strict
three-way comparison. It also runs standalone FTS and Dense at Top 20, then
produces a candidate diagnosis. Classification uses the exact authorization-
filtered branch candidates captured by each Hybrid request; Top-20 standalone
runs provide supporting branch metrics. The diagnosis distinguishes:

- `fusion_ordering_gap`: every missing gold chunk was generated by at least
  one branch, so a reranker could potentially help;
- `candidate_generation_gap`: none of the missing gold chunks entered either
  branch candidate set, so a reranker cannot help;
- `mixed_gap`: only some missing gold chunks were generated;
- `complete`: Hybrid retrieved every gold chunk in Top 5.

Candidate sets are written only to evaluation manifests. They are excluded
from public API JSON, while forbidden candidates still fail the evaluation
safety gate even when absent from final Hybrid hits.

The preregistered v3 smoke dataset contains 20 fixed queries spanning exact
identifiers, semantic paraphrases, Chinese-English retrieval, near-duplicate
distractors, ACL filtering, and tenant isolation. Run each retriever over its
exact bytes while developing:

```bash
mkdir -p artifacts/evals
go run ./cmd/raghub-eval \
  -migrate \
  -mode fts \
  -dataset datasets/smoke/v3.json \
  -output artifacts/evals/v3-fts.json

go run ./cmd/raghub-eval \
  -mode dense \
  -dataset datasets/smoke/v3.json \
  -output artifacts/evals/v3-dense.json

go run ./cmd/raghub-eval \
  -mode hybrid \
  -dataset datasets/smoke/v3.json \
  -output artifacts/evals/v3-hybrid.json
```

The individual runs are useful as pre-commit acceptance checks. A trusted
three-way comparison additionally requires all manifests to identify the same
clean commit. After committing an accepted stage, generate and compare all
three runs with the fail-fast target:

```bash
make eval-three-way
```

The target checks for a clean committed revision before any retrieval run. It
builds the evaluator binary so Go embeds the checked VCS revision, then writes
the FTS, Dense, Hybrid, and three-way comparison artifacts under
`artifacts/evals/` by default. `make eval-v3-regression` and
`make eval-v2-regression` run the same three-way protocol over the retained
smoke corpora; `make eval-all` runs the 50-query benchmark followed by both
regressions.
`make eval-paired` remains an alias for the current three-way protocol.

Each report-v3 manifest records the dataset hash, retriever/config identity,
runtime information, aggregate metrics, per-query ranks/scores, and internal
candidate IDs/ranks. The comparison tool
refuses a three-way report unless retriever identities, corpus, dataset hash,
TopK, query identities, gold/forbidden references, runtime, clean revision,
smoke status, and safety gates agree. These reports remain `smoke`: they prove
that one fixed local path ran, but are too synthetic to support a general
retrieval-quality claim. Query text and gold references in v3 must not be
changed in response to observed rankings; changes require a new dataset
version. The same freeze rule applies to benchmark v1.

LM Studio reports the configured model name but not a verifiable weight
revision. `profile_id` is therefore an explicit operator-managed vector-space
boundary, not proof that model weights are cryptographically pinned.

Metric names are intentionally explicit:

- `hit_rate_at_k`: fraction of queries with at least one gold chunk in top K;
- `recall_at_k`: mean `retrieved gold / all gold` at K;
- `mrr`: reciprocal rank of the first gold chunk;
- `ndcg_at_k`: binary-relevance ranking quality at K.

## Verification

```bash
go test ./...
go vet ./...
RAGHUB_TEST_DATABASE_URL="$RAGHUB_DATABASE_URL" \
  go test -count=1 ./internal/store/postgres
```

Or use the make targets:

```bash
make db-up
make verify
make test-integration
make eval-all
```

The release-oriented gate additionally runs the race detector, verifies module
checksums, checks formatting, and scans reachable Go code for known
vulnerabilities:

```bash
GOSUMDB=sum.golang.org make verify-release
```

Integration tests skip when `RAGHUB_TEST_DATABASE_URL` is absent; a skipped test
is not PostgreSQL runtime evidence.

## API and design

- [OpenAPI contract](api/openapi.yaml)
- [ADR 0001: measurable PostgreSQL FTS slice](docs/adr/0001-first-retrieval-slice.md)
- [ADR 0002: exact pgvector Dense baseline](docs/adr/0002-exact-pgvector-dense-baseline.md)
- [ADR 0003: fail-closed Hybrid RRF baseline](docs/adr/0003-hybrid-rrf-baseline.md)
- [ADR 0004: hard benchmark and exact candidate diagnosis](docs/adr/0004-hard-benchmark-candidate-diagnosis.md)
- [Historical benchmark v1 experiment](docs/experiments/2026-08-20-benchmark-v1.md)
- [Current release-candidate benchmark](docs/experiments/2026-09-02-benchmark-v1-release-candidate.md)
- [v0.1.0-alpha release readiness](docs/releases/v0.1.0-alpha-readiness.md)

## Project policy

- [Apache-2.0 license](LICENSE)
- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Changelog](CHANGELOG.md)

Opening the source does not change the current operating boundary. Before any
real-data or internet-facing deployment, add trusted identity, rate and
resource limits, secrets management, transport security, observability,
backup/restore procedures, retention controls, and realistic load and quality
validation. The checked-in benchmark is synthetic and its results do not
establish general retrieval quality or production performance.
- [Database migrations](migrations/)

The local module path is currently `raghub`. It should be changed to the final
public repository path once that namespace is chosen.

## Next slices

1. Analyze benchmark v1 with the preregistered reranker gate; do not edit its
   gold after the first run.
2. Add reranking only if at least three Hybrid queries are incomplete and at
   least half of missing gold chunks already exist in the branch candidate
   union.
3. Otherwise improve candidate generation and add long-document, version-
   transition, and redacted real bad-case datasets.
4. Evaluate tenant-aware ANN only when corpus scale justifies it, comparing its
   recall against the exact Dense baseline.
5. Add OpenTelemetry spans, load tests, CI, and a verified identity boundary.

Generation and Agentic RAG come after retrieval evidence, not before it.
