# ADR 0002: Exact pgvector dense retrieval baseline

- Status: Accepted
- Date: 2026-08-19

## Context

The FTS slice provides a deterministic lexical baseline, but it cannot measure
semantic paraphrases or multilingual retrieval. The next slice needs real
embeddings, versioned provenance, authorization-safe search, and a paired
comparison against FTS before adding fusion or reranking.

The local provider is LM Studio's OpenAI-compatible embeddings endpoint using
`text-embedding-bge-m3`. The observed API returns 1024-dimensional vectors and
supports batched inputs. It reports a model name, not a verifiable weight-file
revision, so the system must not claim that the underlying weights are pinned.

pgvector supports exact and approximate nearest-neighbor search. Its official
documentation states that approximate-index filtering is applied after the
index scan and that shared multi-tenant ANN indexes can affect recall. That is
an unacceptable confounder for the first Dense quality and isolation baseline.

## Decision

1. Store 1024-dimensional vectors in pgvector but perform exact cosine search.
   A materialized CTE first applies tenant, embedding-profile, active-version,
   and principal ACL predicates. The outer query orders only that authorized
   set by cosine distance and applies `LIMIT`.
2. Do not create HNSW or IVFFlat yet. Add ANN only in a separately measured
   stage with tenant partitioning or an equivalent isolation design,
   iterative-scan settings, and exact-vs-ANN recall comparisons.
3. Treat an embedding profile as an immutable vector-space boundary. Its ID
   binds provider, model name, dimensions, document recipe, and query recipe.
   Reusing an ID with different configuration fails.
4. Keep the source-document fingerprint independent from embeddings. A new
   model/profile adds derived vectors to the same immutable chunks rather than
   inventing a source-document version.
5. Embed every chunk outside the database transaction. Validate count,
   dimensions, finite values, and non-zero norm. Inside one transaction, write
   the document version, chunks, profile, and all vectors; advance
   `current_version` last. Provider failure therefore leaves the previous
   version active.
6. When an FTS-ingested active version has the same fingerprint, atomically
   backfill its missing profile vectors and return `unchanged=true`. The same
   profile/input may not silently change to a different vector.
7. Expose `fts` and `dense` as separate search modes and preserve stage scores
   and traces. Do not add RRF in this slice; both paths remain independently
   measurable baselines.
8. Compare both modes on the exact same versioned dataset bytes. Reports must
   retain dataset/config hashes, per-query hits and traces, deterministic
   leakage gates, and explicit `smoke` status.

## Consequences

- Dense search has deterministic complete recall within the authorized set,
  making quality failures attributable to the embedding/profile rather than an
  ANN candidate budget.
- Query cost is linear in the authorized vector set. This is intentional for
  the small evidence corpus and is not a production performance claim.
- Synchronous ingestion depends on the embedding provider and may repeat model
  inference on a retry, although database activation remains atomic and
  idempotent.
- A model-name-only LM Studio response is insufficient weight provenance. A
  changed model under the same profile is detected only when it produces a
  different stored vector; operators must allocate a new profile ID for an
  intentional model/revision change.
- Development tenant/principal headers remain spoofable. Production identity
  derivation is outside this slice.

## References

- [pgvector: exact/approximate search, filtering, and multitenancy](https://github.com/pgvector/pgvector)
- [pgvector Go integration guidance](https://github.com/pgvector/pgvector-go)
