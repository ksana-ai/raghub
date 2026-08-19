# RAGHub

RAGHub is a Go-native retrieval platform built as an evidence-backed project,
not a chat-demo wrapper. The current repository implements the first measurable
vertical slice:

`Markdown ingestion -> versioned PostgreSQL storage -> FTS retrieval -> tenant/ACL filtering -> citations -> offline evaluation`

## Current status

Implemented in this slice:

- deterministic, heading-aware Markdown chunking;
- immutable document versions with idempotent re-ingestion;
- separate `raw_text` and `indexed_text` fields;
- PostgreSQL weighted full-text search with a GIN index;
- current-version, tenant, and principal ACL filters in the SQL query;
- exact document-version/chunk/source references in every hit;
- per-stage score/rank and latency traces;
- a versioned smoke dataset and JSON evaluation manifest;
- HitRate@K, standard Recall@K, MRR, binary nDCG@K, p50, and p95;
- unit and opt-in PostgreSQL integration tests.

Not implemented yet:

- embeddings or pgvector;
- dense retrieval, RRF, or reranking;
- answer generation, citation verification, or an LLM judge;
- Contextual Retrieval, Agentic RAG, RAPTOR, or GraphRAG;
- production authentication, deployment, or production performance evidence.

The distinction matters: the current result is an FTS baseline and smoke
evaluation, not a completed hybrid RAG system.

## Quick start

Requirements:

- Go 1.24;
- Docker with Compose.

Start PostgreSQL:

```bash
docker compose up -d postgres
export RAGHUB_DATABASE_URL='postgres://raghub:raghub@localhost:55432/raghub?sslmode=disable'
```

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

Search it:

```bash
curl --fail-with-body \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: demo' \
  -d '{"query":"blue green deployment","top_k":5}' \
  http://localhost:8080/v1/search
```

`X-Tenant-ID` and `X-Principal-ID` are development authorization context.
They are deliberately outside the JSON payload, but they are still spoofable
headers until a trusted authentication layer derives them from credentials.

## Evaluation

Run the versioned smoke dataset against the same real PostgreSQL path:

```bash
mkdir -p artifacts/evals
go run ./cmd/raghub-eval \
  -migrate \
  -dataset datasets/smoke/v1.json \
  -output artifacts/evals/smoke.json
```

The generated manifest records the dataset hash, retriever/config identity,
runtime information, aggregate metrics, and per-query ranks/scores. Its status
is `smoke`: it proves that this fixed path ran, but it is too small to support a
general retrieval-quality claim.

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
make eval
```

Integration tests skip when `RAGHUB_TEST_DATABASE_URL` is absent; a skipped test
is not PostgreSQL runtime evidence.

## API and design

- [OpenAPI contract](api/openapi.yaml)
- [ADR 0001: measurable PostgreSQL FTS slice](docs/adr/0001-first-retrieval-slice.md)
- [Database migrations](migrations/)

The local module path is currently `raghub`. It should be changed to the final
public repository path once that namespace is chosen.

## Next slices

1. Add a versioned embedding provider and pgvector dense index.
2. Run FTS-only and dense-only on the same dataset.
3. Add RRF with preserved sparse, dense, and fusion scores.
4. Add reranking only after hybrid failure cases justify it.
5. Expand to 50-100 preregistered gold queries, including semantic rewrites,
   version transitions, ACL negatives, and cross-tenant leakage gates.
6. Add OpenTelemetry spans, load tests, CI, and a verified identity boundary.

Generation and Agentic RAG come after retrieval evidence, not before it.
