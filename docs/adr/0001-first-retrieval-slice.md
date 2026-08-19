# ADR 0001: Start with a measurable PostgreSQL FTS slice

- Status: accepted
- Date: 2026-08-19

## Context

RAGHub starts in an empty directory. Its target shape includes structure-aware
ingestion, PostgreSQL full-text and dense retrieval, fusion, reranking,
authorization filters, citations, and versioned evaluation. Implementing all
retrieval modes at once would make it difficult to know whether failures come
from ingestion, authorization, sparse retrieval, embeddings, fusion, or
evaluation.

The first slice therefore needs a real storage and retrieval path plus a
machine-checkable baseline. An in-memory or scripted provider would not provide
that evidence.

## Decision

Build the first vertical slice around PostgreSQL FTS:

1. ingest Markdown into deterministic, heading-aware chunks;
2. store immutable document versions and atomically select the active version;
3. retain the raw source document and store chunk `raw_text` separately from
   `indexed_text` so later contextual prefixes cannot contaminate displayed
   citations;
4. enforce tenant and principal ACL filters inside the retrieval query;
5. return immutable chunk/version/source references and FTS stage scores;
6. run a versioned smoke dataset through HitRate@K, standard Recall@K, MRR,
   binary nDCG@K, and latency percentiles;
7. record the dataset hash and runtime/configuration metadata in a JSON
   manifest.

The development API obtains tenant and principal identifiers from headers. This
is an application-boundary contract, not authentication. Production readiness
requires a verified identity layer that overwrites these values from trusted
credentials.

## Consequences

- This slice is a real FTS baseline, not a hybrid-RAG implementation.
- Dense retrieval, RRF, reranking, generation, Contextual Retrieval, and
  Agentic RAG remain explicitly out of scope.
- The chunk size is currently a rune-count heuristic. Evaluation must justify
  token-aware chunking before it replaces the baseline.
- A small smoke dataset proves that the path runs; it is not a publishable
  quality claim. A 50-100 query dataset and paired ablations are still required.
- PostgreSQL failures surface as errors and are not converted into empty search
  results.
