# ADR 0003: Deterministic hybrid retrieval with reciprocal rank fusion

- Status: Accepted
- Date: 2026-08-19

## Context

RAGHub now has independently measurable PostgreSQL FTS and exact pgvector
Dense baselines. Their scores are not calibrated to the same scale: PostgreSQL
text-search rank and cosine similarity cannot be added or averaged without a
separate normalization experiment. The next slice needs to test whether their
ranked candidates are complementary while keeping both baselines inspectable.

The v2 smoke run is intentionally small. Dense retrieved all eight gold chunks
at Top 5 while FTS retrieved five, but that result is only a local regression
anchor. It does not establish that Dense dominates lexical search, especially
for exact identifiers, error codes, product keys, or near-duplicate passages.

## Decision

1. Add an explicit `hybrid` search mode. Keep `fts` and `dense` unchanged so
   all three paths remain independently measurable.
2. Retrieve the FTS and Dense candidate lists concurrently. A branch receives
   `max(request.top_k, configured_candidate_depth)` candidates, bounded by the
   existing maximum Top K. The preregistered defaults are 20 candidates from
   each branch.
3. Fuse candidates by reciprocal rank fusion (RRF):

   `rrf_score(chunk) = sum(1 / (rrf_k + branch_rank(chunk)))`

   The preregistered default is `rrf_k = 60`. Raw FTS and Dense scores are not
   normalized or mixed into the fusion score.
4. Deduplicate strictly by immutable `chunk_id`. Sort by descending RRF score
   and then ascending `chunk_id` for deterministic ties. If two branches return
   inconsistent source or citation fields for the same chunk, fail instead of
   silently choosing one representation.
5. Preserve every source-stage rank and score and append the final RRF rank and
   score. Emit traces in deterministic FTS, Dense, and fusion-stage order even
   though the two retrieval branches execute concurrently.
6. Fail closed if either branch fails. Cancel and await its peer before
   returning the error. Production fallback is a separate policy and must not
   make an evaluation report look like a successful Hybrid run.
7. Keep tenant, principal ACL, embedding profile, and active-version predicates
   inside each database retrieval branch. The fusion layer may only combine
   already-authorized hits and never performs a broader fetch by chunk ID.
8. Run FTS, Dense, and Hybrid over identical dataset bytes and the same actual
   corpus inventory. Comparison requires a clean shared code revision and
   mechanically revalidates ranks, metrics, latency aggregates, safety gates,
   and runtime provenance.
9. Retain v2 as a regression set. Before observing Hybrid results, add a
   versioned complementary smoke set with exact identifiers, semantic
   paraphrases, multilingual queries, near-duplicate distractors, ACL cases,
   and cross-tenant negatives. Do not tune its queries or gold chunks after a
   retrieval run.

## Consequences

- RRF provides a deterministic rank-only baseline without pretending that FTS
  and cosine scores are comparable probabilities.
- Two retrieval branches increase work per query. Parallel execution reduces
  wall-clock latency but does not reduce database or embedding cost; p50 and
  p95 must be reported rather than assumed.
- The fixed candidate depths can hide a relevant chunk that neither branch
  returns. Candidate-depth sweeps belong to a later preregistered experiment,
  not post-hoc tuning of this baseline.
- A successful smoke comparison proves this fixed fusion path and its security
  gates ran locally. It is not a production quality or scale claim.
- Reranking is deferred until categorized Hybrid failures show that improving
  the order of the fused candidate set is worth another model dependency.
- HNSW and IVFFlat remain deferred; exact Dense retrieval stays the recall
  reference for any later ANN experiment.

## References

- Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher,
  [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://doi.org/10.1145/1571941.1572114)
- [ADR 0001: measurable PostgreSQL FTS slice](0001-first-retrieval-slice.md)
- [ADR 0002: exact pgvector Dense baseline](0002-exact-pgvector-dense-baseline.md)
