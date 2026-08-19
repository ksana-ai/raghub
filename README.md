# RAGHub

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
- strict JSON evaluation plus backward-compatible pairwise and strict
  FTS/Dense/Hybrid comparison manifests;
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

- Go 1.24;
- Docker with Compose.
- LM Studio serving `text-embedding-bge-m3` through the OpenAI-compatible
  `/v1/embeddings` API for ingestion, Dense search, and Hybrid search.

Start PostgreSQL:

```bash
docker compose up -d postgres
export RAGHUB_DATABASE_URL='postgres://raghub:raghub@localhost:55432/raghub?sslmode=disable'
export RAGHUB_EMBEDDING_ENDPOINT='http://127.0.0.1:1234/v1/embeddings'
```

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
`artifacts/evals/` by default. `make eval-v2-regression` runs the same three-way
protocol over the retained v2 corpus; `make eval-all` runs both v3 and v2.
`make eval-paired` remains an alias for the current three-way protocol.

Each manifest records the dataset hash, retriever/config identity, runtime
information, aggregate metrics, and per-query ranks/scores. The comparison tool
refuses a three-way report unless retriever identities, corpus, dataset hash,
TopK, query identities, gold/forbidden references, runtime, clean revision,
smoke status, and safety gates agree. These reports remain `smoke`: they prove
that one fixed local path ran, but are too small to support a general
retrieval-quality claim. Query text and gold references in v3 must not be
changed in response to observed rankings; changes require a new dataset
version.

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

Integration tests skip when `RAGHUB_TEST_DATABASE_URL` is absent; a skipped test
is not PostgreSQL runtime evidence.

## API and design

- [OpenAPI contract](api/openapi.yaml)
- [ADR 0001: measurable PostgreSQL FTS slice](docs/adr/0001-first-retrieval-slice.md)
- [ADR 0002: exact pgvector Dense baseline](docs/adr/0002-exact-pgvector-dense-baseline.md)
- [ADR 0003: fail-closed Hybrid RRF baseline](docs/adr/0003-hybrid-rrf-baseline.md)
- [Database migrations](migrations/)

The local module path is currently `raghub`. It should be changed to the final
public repository path once that namespace is chosen.

## Next slices

1. Analyze preregistered Hybrid bad cases without changing v3 gold after the
   first run.
2. Expand to 50-100 preregistered gold queries, including semantic rewrites,
   version transitions, ACL negatives, and cross-tenant leakage gates.
3. Add reranking only if Hybrid failure cases justify the added stage.
4. Evaluate tenant-aware ANN only when corpus scale justifies it, comparing its
   recall against the exact Dense baseline.
5. Add OpenTelemetry spans, load tests, CI, and a verified identity boundary.

Generation and Agentic RAG come after retrieval evidence, not before it.
