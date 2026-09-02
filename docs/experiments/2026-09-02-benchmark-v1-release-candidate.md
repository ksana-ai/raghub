# Benchmark v1 release-candidate experiment

Experiment date: 2026-09-02

Clean code revision: `ac39d2ae70b28144a28073cadec7b5c65cb9b4a7`

Dataset: `datasets/benchmark/v1.json`

Dataset exact-byte SHA-256:
`aa44175b9ae656d97473a8340ebac59bc1432d7cee90e51432c2b4f89e61f85f`

Corpus SHA-256:
`fc416bf81fc623447c3b4a175f7c4d36db9d328b7a8abeb00400b664166e5653`

Status: **accepted as synthetic, local, clean-revision release evidence**.

## Question and preregistered decision

After the open-source supply-chain upgrade, does the frozen 50-query benchmark
still pass its quality and authorization gates, and do incomplete Hybrid
results make a reranker experiment eligible?

The candidate-diagnosis gate requires at least three incomplete queries and at
least 50% of missing gold chunks to be recoverable from the union of FTS and
Dense candidates. The frozen dataset and gate were not changed for this run.

## Runtime and configuration

- Go: 1.27.1
- PostgreSQL: 17.11 (Debian 17.11-1.pgdg12+2)
- pgvector: 0.8.6
- Embedding provider: local LM Studio, OpenAI-compatible API
- Reported embedding model: `text-embedding-bge-m3`
- Embedding dimensions: 1024
- Model-weight revision: not reported by provider
- Top K: 5
- Hybrid candidates: authorized FTS 20 plus authorized Dense 20
- Fusion: reciprocal rank fusion, `k=60`, fail closed

Retriever configuration hashes:

- FTS: `fd58c25f77cde5a128dd402f4741b4389f7d8a7981b3b1f58435db37e311700c`
- Dense: `48bdb48a5bb41f27f6d7309760bbbbe68f2db8cc2eba4dbab2878df2cc666b0d`
- Hybrid: `43ea6c6518d2745defc6372c10044dd9b95671c80c09468f8153e80ed7b11dd9`

## Command

The repository was clean before execution.

```bash
GOTOOLCHAIN=go1.27.1 \
GOSUMDB=sum.golang.org \
make eval-benchmark \
  DATABASE_URL='postgres://raghub:raghub@127.0.0.1:55433/raghub?sslmode=disable'
```

Port 55433 was an isolated local override because 55432 was already allocated.
The port has no effect on retrieval configuration or metrics.

## Results

| Retriever | HitRate@5 | Recall@5 | MRR | nDCG@5 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FTS | 0.2600 | 0.2600 | 0.2600 | 0.2600 | 0.80 ms | 1.22 ms |
| Exact Dense | 1.0000 | 1.0000 | 0.9867 | 0.9844 | 34.54 ms | 52.50 ms |
| Hybrid RRF | 1.0000 | 1.0000 | 0.9867 | 0.9844 | 33.34 ms | 54.40 ms |

Safety and completeness:

- 50 queries evaluated with zero search errors.
- Zero forbidden final hits.
- Corpus-reference, corpus-isolation, search-completion, and forbidden-hit
  gates all passed for all three retrievers.
- Dense and Hybrid achieved complete recall on 50/50 queries.
- Candidate union recall was 1.0.

Candidate diagnosis:

| Classification | Queries |
| --- | ---: |
| `complete` | 50 |
| `fusion_ordering_gap` | 0 |
| `mixed_gap` | 0 |
| `candidate_generation_gap` | 0 |

Reranker experiment gate:

- incomplete queries: 0; required: 3;
- recoverable missing fraction: 0; required: 0.5;
- `eligible=false`.

## Interpretation and improvement strategy

The dependency and Go toolchain upgrades preserved the historical quality
metrics exactly on this frozen dataset. This is regression evidence for one
local path, not evidence that the new runtime caused or generally guarantees
the result.

Do not implement a reranker next. There are no misses for it to repair, and
Hybrid again has identical quality metrics to Dense. The appropriate quality
improvement step is a separately versioned, more realistic benchmark containing
independently collected hard negatives, longer multi-chunk documents, noisy
queries, and genuine ordering failures. Its acceptance criteria must be frozen
before results are observed.

The next release step is operational: push both accepted commits, observe the
configured GitHub Actions workflow, enable repository security controls, and
only then create `v0.1.0-alpha`.

## Evidence boundaries

The full report-v3 and comparison manifests remain local generated artifacts
under `artifacts/evals/` and are ignored to prevent accidental publication of
future private corpora. This committed report is the sanitized release record.
It preserves the clean revision, hashes, configurations, aggregate results,
safety gates, and decision. LM Studio did not expose a verifiable model-weight
revision, so another operator must treat a rerun as new evidence even when the
reported model name matches.
