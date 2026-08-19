# RAGHub

RAGHub is a Go-native retrieval platform built as an evidence-backed project,
not a chat-demo wrapper. The current repository implements two independently
measurable retrieval baselines:

`Markdown ingestion -> versioned PostgreSQL + pgvector storage -> FTS or exact Dense retrieval -> tenant/ACL filtering -> citations -> paired offline evaluation`

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
- explicit `fts` and `dense` modes with query-embedding and database traces;
- current-version, tenant, and principal ACL filters in the SQL query;
- exact document-version/chunk/source references in every hit;
- per-stage score/rank and latency traces;
- versioned lexical and paired semantic/multilingual smoke datasets;
- strict JSON evaluation and paired-comparison manifests;
- HitRate@K, standard Recall@K, MRR, binary nDCG@K, p50, and p95;
- unit and opt-in PostgreSQL integration tests.

Not implemented yet:

- approximate nearest-neighbor indexes or production-scale performance proof;
- hybrid RRF or reranking;
- answer generation, citation verification, or an LLM judge;
- Contextual Retrieval, Agentic RAG, RAPTOR, or GraphRAG;
- production authentication, deployment, or production performance evidence.

The distinction matters: the current result is an FTS baseline plus an exact
Dense smoke baseline, not a completed hybrid or production RAG system.

## Quick start

Requirements:

- Go 1.24;
- Docker with Compose.
- LM Studio serving `text-embedding-bge-m3` through the OpenAI-compatible
  `/v1/embeddings` API for ingestion and Dense search.

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

`X-Tenant-ID` and `X-Principal-ID` are development authorization context.
They are deliberately outside the JSON payload, but they are still spoofable
headers until a trusted authentication layer derives them from credentials.
`/readyz` checks both PostgreSQL and the configured embedding model; successful
process liveness alone is available at `/healthz`.

## Evaluation

Run either retriever over the same exact-byte v2 dataset while developing:

```bash
mkdir -p artifacts/evals
go run ./cmd/raghub-eval \
  -migrate \
  -mode fts \
  -dataset datasets/smoke/v2.json \
  -output artifacts/evals/v2-fts.json

go run ./cmd/raghub-eval \
  -mode dense \
  -dataset datasets/smoke/v2.json \
  -output artifacts/evals/v2-dense.json

```

The individual runs are useful as pre-commit acceptance checks. A trusted
paired comparison additionally requires both manifests to identify the same
clean commit. After committing an accepted stage, generate and compare both
runs with the fail-fast target:

```bash
make eval-paired
```

The target checks for a clean committed revision before making either model
run, then writes the FTS, Dense, and comparison artifacts under
`artifacts/evals/` by default.

Each manifest records the dataset hash, retriever/config identity, runtime
information, aggregate metrics, and per-query ranks/scores. The comparison tool
refuses to pair reports unless corpus, dataset hash, TopK, query identities,
gold/forbidden references, smoke status, and safety gates agree. These reports
remain `smoke`: they prove that one fixed local path ran, but are too small to
support a general retrieval-quality claim.

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
make eval-paired
```

Integration tests skip when `RAGHUB_TEST_DATABASE_URL` is absent; a skipped test
is not PostgreSQL runtime evidence.

## API and design

- [OpenAPI contract](api/openapi.yaml)
- [ADR 0001: measurable PostgreSQL FTS slice](docs/adr/0001-first-retrieval-slice.md)
- [ADR 0002: exact pgvector Dense baseline](docs/adr/0002-exact-pgvector-dense-baseline.md)
- [Database migrations](migrations/)

The local module path is currently `raghub`. It should be changed to the final
public repository path once that namespace is chosen.

## Next slices

1. Add RRF with preserved sparse, dense, and fusion scores.
2. Add reranking only after hybrid failure cases justify it.
3. Expand to 50-100 preregistered gold queries, including semantic rewrites,
   version transitions, ACL negatives, and cross-tenant leakage gates.
4. Evaluate tenant-aware ANN only when corpus scale justifies it, comparing its
   recall against the exact Dense baseline.
5. Add OpenTelemetry spans, load tests, CI, and a verified identity boundary.

Generation and Agentic RAG come after retrieval evidence, not before it.
