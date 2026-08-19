package eval

import (
	"strings"
	"testing"

	"raghub/internal/model"
)

func TestCompareManifestsProducesCandidateMinusBaselineDeltas(t *testing.T) {
	t.Parallel()

	baseline := comparableManifest(t, "baseline", "postgres_fts", RankingMetrics{}, LatencyPercentiles{P50MS: 10, P95MS: 10})
	candidate := comparableManifest(t, "candidate", "postgres_dense", RankingMetrics{
		RecallAtK: 1, HitRateAtK: 1, MRR: 1, NDCGAtK: 1,
	}, LatencyPercentiles{P50MS: 15, P95MS: 15})

	comparison, err := CompareManifests(baseline, candidate)
	if err != nil {
		t.Fatalf("CompareManifests() error = %v", err)
	}
	if comparison.SchemaVersion != ComparisonSchemaVersion || comparison.Status != SmokeStatus {
		t.Fatalf("comparison must remain smoke scoped: %+v", comparison)
	}
	if comparison.Delta.Direction != "candidate-baseline" || comparison.QueryCount != 1 {
		t.Fatalf("unexpected pairing metadata: %+v", comparison)
	}
	assertClose(t, "delta recall", comparison.Delta.Metrics.RecallAtK, 1)
	assertClose(t, "delta hit rate", comparison.Delta.Metrics.HitRateAtK, 1)
	assertClose(t, "delta mrr", comparison.Delta.Metrics.MRR, 1)
	assertClose(t, "delta ndcg", comparison.Delta.Metrics.NDCGAtK, 1)
	assertClose(t, "delta p50", comparison.Delta.Latency.P50MS, 5)
	assertClose(t, "delta p95", comparison.Delta.Latency.P95MS, 5)
	if comparison.Baseline.Retriever.Name != "postgres_fts" || comparison.Candidate.Retriever.Name != "postgres_dense" {
		t.Fatalf("retriever provenance missing: %+v", comparison)
	}
	if comparison.Baseline.Runtime.DatabaseVersion != "test-db" || comparison.Candidate.Runtime.DatabaseVersion != "test-db" {
		t.Fatalf("runtime provenance missing: %+v", comparison)
	}
	if _, err := MarshalComparison(comparison); err != nil {
		t.Fatalf("MarshalComparison() error = %v", err)
	}
}

func TestCompareManifestsRejectsUnpairedOrUntrustedInputs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*Manifest)
		want   string
	}{
		{name: "report schema", mutate: func(value *Manifest) { value.SchemaVersion = "raghub.eval.report/v1" }, want: "schema_version"},
		{name: "incomplete status", mutate: func(value *Manifest) { value.Status = IncompleteStatus }, want: "status"},
		{name: "failed gate", mutate: func(value *Manifest) { value.Summary.Gates.Pass = false }, want: "gates.pass"},
		{name: "dataset name", mutate: func(value *Manifest) { value.Dataset.Name = "other" }, want: "dataset"},
		{name: "dataset version", mutate: func(value *Manifest) { value.Dataset.Version = "3.0.0" }, want: "dataset"},
		{name: "dataset sha", mutate: func(value *Manifest) { value.Dataset.SHA256 = strings.Repeat("b", 64) }, want: "dataset"},
		{name: "corpus sha", mutate: func(value *Manifest) { value.CorpusSHA256 = strings.Repeat("c", 64) }, want: "corpus_sha256"},
		{name: "top k", mutate: func(value *Manifest) { value.TopK = 10 }, want: "top_k"},
		{name: "case id", mutate: func(value *Manifest) { value.PerQuery[0].ID = "other" }, want: "case identity"},
		{name: "case category", mutate: func(value *Manifest) { value.PerQuery[0].Category = "other" }, want: "case identity"},
		{name: "case tenant", mutate: func(value *Manifest) { value.PerQuery[0].TenantID = "other" }, want: "case identity"},
		{name: "case principal", mutate: func(value *Manifest) { value.PerQuery[0].PrincipalID = "other" }, want: "case identity"},
		{name: "case query", mutate: func(value *Manifest) { value.PerQuery[0].Query = "other" }, want: "case identity"},
		{name: "gold chunks", mutate: func(value *Manifest) { value.PerQuery[0].GoldChunkIDs = []string{"different-gold:v000001:c0000"} }, want: "gold_chunk_ids"},
		{name: "forbidden chunks", mutate: func(value *Manifest) { value.PerQuery[0].ForbiddenChunkIDs = []string{"different:v000001:c0000"} }, want: "forbidden_chunk_ids"},
		{name: "config hash", mutate: func(value *Manifest) { value.Retriever.ConfigSHA256 = strings.Repeat("0", 64) }, want: "config_sha256"},
		{name: "go runtime", mutate: func(value *Manifest) { value.Runtime.GoVersion = "other-go" }, want: "runtime"},
		{name: "database runtime", mutate: func(value *Manifest) { value.Runtime.DatabaseVersion = "other-db" }, want: "runtime"},
		{name: "pgvector runtime", mutate: func(value *Manifest) { value.Runtime.VectorExtensionVersion = "other-vector" }, want: "runtime"},
		{name: "code runtime", mutate: func(value *Manifest) { value.Runtime.CodeRevision = "other-code" }, want: "runtime"},
		{name: "uncommitted revision", mutate: func(value *Manifest) { value.Runtime.CodeRevision = "uncommitted" }, want: "clean committed"},
		{name: "dirty revision", mutate: func(value *Manifest) { value.Runtime.CodeRevision = "abc123+dirty" }, want: "clean committed"},
		{name: "per-query metrics", mutate: func(value *Manifest) { value.PerQuery[0].Metrics.RecallAtK = 1 }, want: "metrics do not match"},
		{name: "summary metrics", mutate: func(value *Manifest) { value.Summary.Metrics.RecallAtK = 1 }, want: "summary metrics"},
		{name: "summary latency", mutate: func(value *Manifest) { value.Summary.Latency.P95MS = 1 }, want: "summary latency"},
		{name: "discontinuous rank", mutate: func(value *Manifest) {
			value.PerQuery[0].Hits = []HitRecord{{Rank: 2, SearchHit: model.SearchHit{ChunkID: "doc:v000001:c0000"}}}
		}, want: "ranks must be continuous"},
		{name: "duplicate hit", mutate: func(value *Manifest) {
			value.PerQuery[0].Hits = []HitRecord{
				{Rank: 1, SearchHit: model.SearchHit{ChunkID: "miss:v000001:c0000"}},
				{Rank: 2, SearchHit: model.SearchHit{ChunkID: "miss:v000001:c0000"}},
			}
		}, want: "duplicate hit"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			baseline := comparableManifest(t, "baseline", "postgres_fts", RankingMetrics{}, LatencyPercentiles{})
			candidate := comparableManifest(t, "candidate", "postgres_dense", RankingMetrics{}, LatencyPercentiles{})
			test.mutate(&candidate)
			_, err := CompareManifests(baseline, candidate)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("CompareManifests() error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestParseManifestIsStrict(t *testing.T) {
	t.Parallel()

	for _, data := range []string{
		`{"schema_version":"raghub.eval.report/v1","unknown":true}`,
		`{} {}`,
	} {
		if _, err := ParseManifest([]byte(data)); err == nil {
			t.Fatalf("ParseManifest(%q) error = nil", data)
		}
	}
}

func comparableManifest(t *testing.T, runID, retrieverName string, metrics RankingMetrics, latency LatencyPercentiles) Manifest {
	t.Helper()

	config := map[string]any{"mode": retrieverName}
	_, configSHA256, err := normalizeAndHashConfig(config)
	if err != nil {
		t.Fatalf("normalizeAndHashConfig() error = %v", err)
	}
	manifest := Manifest{
		SchemaVersion: ReportSchemaVersion,
		RunID:         runID,
		Status:        SmokeStatus,
		Command:       "raghub-eval",
		CorpusSHA256:  strings.Repeat("a", 64),
		Dataset: DatasetManifest{
			Name: "paired", Version: "2.0.0", SHA256: strings.Repeat("a", 64),
		},
		Retriever: RetrieverManifest{Name: retrieverName, Config: config, ConfigSHA256: configSHA256},
		Runtime: RuntimeManifest{
			GoVersion: "go-test", DatabaseVersion: "test-db", VectorExtensionVersion: "0.8.1", CodeRevision: "test-revision",
		},
		TopK: 5,
		Summary: Summary{
			QueryCount: 1,
			Gates: GateSummary{
				Pass: true, CorpusReferencesValid: true, CorpusIsolated: true, SearchCompleted: true, NoForbiddenHits: true,
			},
			Metrics: metrics,
			Latency: latency,
		},
		PerQuery: []QueryResult{{
			ID: "q1", Category: "semantic", TenantID: "tenant", PrincipalID: "principal", Query: "question",
			GoldChunkIDs: []string{"doc:v000001:c0000"}, ForbiddenChunkIDs: []string{"other:v000001:c0000"},
			Metrics: metrics, LatencyMS: latency.P50MS,
		}},
	}
	if metrics == (RankingMetrics{RecallAtK: 1, HitRateAtK: 1, MRR: 1, NDCGAtK: 1}) {
		manifest.PerQuery[0].Hits = []HitRecord{{
			Rank: 1, SearchHit: model.SearchHit{ChunkID: "doc:v000001:c0000"},
		}}
	}
	return manifest
}
