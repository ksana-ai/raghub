# Benchmark v1 historical experiment report

Experiment date: 2026-08-20

Code revision: `c6e071d`

Dataset: `datasets/benchmark/v1.json`
Dataset exact-byte SHA-256:
`aa44175b9ae656d97473a8340ebac59bc1432d7cee90e51432c2b4f89e61f85f`

## Question

Does exact Dense or fixed Hybrid RRF improve retrieval on the preregistered
50-query synthetic benchmark, and do the observed misses justify introducing a
reranker?

## Configuration

- 44 synthetic documents and 50 fixed queries, including 8 two-gold cases
- Top K: 5
- PostgreSQL 17.11 with pgvector 0.8.6
- Embedding endpoint: local LM Studio, OpenAI-compatible API
- Reported model: `text-embedding-bge-m3`, 1024 dimensions
- Hybrid: authorized FTS candidates 20, authorized Dense candidates 20,
  reciprocal-rank-fusion `k=60`
- Authorization safety: tenant and principal filters plus forbidden-hit checks

LM Studio did not expose a verifiable model-weight revision. These results
therefore describe this local run, not every artifact carrying the same model
name.

## Results

| Retriever | HitRate@5 | Recall@5 | MRR | nDCG@5 | p50 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| FTS | 0.2600 | 0.2600 | 0.2600 | 0.2600 | 0.78 ms |
| Exact Dense | 1.0000 | 1.0000 | 0.9867 | 0.9844 | 31.99 ms |
| Hybrid RRF | 1.0000 | 1.0000 | 0.9867 | 0.9844 | 32.81 ms |

- Complete recall: 50/50 queries for Dense and Hybrid.
- Forbidden final hits: 0.
- Tenant or ACL leakage: 0 observed.
- Candidate diagnosis classified all Hybrid cases as `complete`.
- The preregistered reranker gate returned `eligible=false`.

## Decision

Do not add a reranker on the basis of this experiment. There were no remaining
Hybrid misses for a reranker to repair. Dense and Hybrid produced identical
quality metrics, so the experiment also does **not** establish that Hybrid is
better than Dense. FTS contributed no effective quality gain on this synthetic
corpus.

Before revisiting reranking, add realistic bad cases or an independently
versioned corpus that creates observable ordering gaps without adapting the
frozen benchmark to current rankings.

## Evidence limitations

This report preserves the accepted aggregate result and decision, but the raw
JSON manifests from the 2026-08-20 run were generated artifacts and were not
committed. They are no longer available in the original temporary evidence
location. Consequently:

- the code revision and dataset bytes remain inspectable;
- the historical aggregate is recorded but cannot be independently recomputed
  from committed raw manifests alone;
- a new run must receive its own date, clean revision, runtime metadata, and
  report rather than being presented as the original run.

The follow-up release experiment is intentionally tracked separately so that
historical and current evidence are not conflated.
