package eval

import (
	"fmt"
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

func TestCompareThreeManifestsProducesStrictThreeWayDeltas(t *testing.T) {
	t.Parallel()

	miss := RankingMetrics{}
	hit := RankingMetrics{RecallAtK: 1, HitRateAtK: 1, MRR: 1, NDCGAtK: 1}
	fts := comparableManifest(t, "fts-run", "postgres_fts", miss, LatencyPercentiles{P50MS: 2, P95MS: 2})
	dense := comparableManifest(t, "dense-run", "postgres_dense", hit, LatencyPercentiles{P50MS: 7, P95MS: 7})
	hybrid := comparableManifest(t, "hybrid-run", "postgres_hybrid_rrf", hit, LatencyPercentiles{P50MS: 9, P95MS: 9})
	setManifestMode(t, &fts, "fts")
	setManifestMode(t, &dense, "dense")
	setManifestMode(t, &hybrid, "hybrid")

	comparison, err := CompareThreeManifests(fts, dense, hybrid)
	if err != nil {
		t.Fatalf("CompareThreeManifests() error = %v", err)
	}
	if comparison.SchemaVersion != ThreeWayComparisonSchemaVersion || comparison.Status != SmokeStatus || comparison.QueryCount != 1 {
		t.Fatalf("unexpected three-way scope: %+v", comparison)
	}
	if comparison.FTS.Retriever.Name != "postgres_fts" || comparison.Dense.Retriever.Name != "postgres_dense" || comparison.Hybrid.Retriever.Name != "postgres_hybrid_rrf" {
		t.Fatalf("three-way retriever provenance = %+v", comparison)
	}
	if len(comparison.Categories) != 1 || comparison.Categories[0].Category != "semantic" || comparison.Categories[0].QueryCount != 1 {
		t.Fatalf("three-way category metrics = %+v", comparison.Categories)
	}
	if comparison.Deltas.DenseMinusFTS.Direction != "dense-minus-fts" || comparison.Deltas.HybridMinusFTS.Direction != "hybrid-minus-fts" || comparison.Deltas.HybridMinusDense.Direction != "hybrid-minus-dense" {
		t.Fatalf("three-way delta directions = %+v", comparison.Deltas)
	}
	assertClose(t, "dense-minus-fts recall", comparison.Deltas.DenseMinusFTS.Metrics.RecallAtK, 1)
	assertClose(t, "hybrid-minus-fts recall", comparison.Deltas.HybridMinusFTS.Metrics.RecallAtK, 1)
	assertClose(t, "hybrid-minus-dense recall", comparison.Deltas.HybridMinusDense.Metrics.RecallAtK, 0)
	assertClose(t, "hybrid-minus-dense p95", comparison.Deltas.HybridMinusDense.Latency.P95MS, 2)
	if _, err := MarshalThreeWayComparison(comparison); err != nil {
		t.Fatalf("MarshalThreeWayComparison() error = %v", err)
	}
}

func TestCompareThreeManifestsRejectsMislabeledOrUnpairedSide(t *testing.T) {
	t.Parallel()

	newTriplet := func(t *testing.T) (Manifest, Manifest, Manifest) {
		t.Helper()
		hit := RankingMetrics{RecallAtK: 1, HitRateAtK: 1, MRR: 1, NDCGAtK: 1}
		fts := comparableManifest(t, "fts-run", "postgres_fts", hit, LatencyPercentiles{})
		dense := comparableManifest(t, "dense-run", "postgres_dense", hit, LatencyPercentiles{})
		hybrid := comparableManifest(t, "hybrid-run", "postgres_hybrid_rrf", hit, LatencyPercentiles{})
		setManifestMode(t, &fts, "fts")
		setManifestMode(t, &dense, "dense")
		setManifestMode(t, &hybrid, "hybrid")
		return fts, dense, hybrid
	}

	t.Run("mislabeled hybrid", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		setManifestMode(t, &hybrid, "dense")
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "mode must be \"hybrid\"") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("hybrid dataset mismatch", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		hybrid.Dataset.Version = "different"
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "dataset") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("wrong FTS retriever name", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		fts.Retriever.Name = "custom_fts"
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "retriever name") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("tampered FTS stage score", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		fts.PerQuery[0].Hits[0].StageScores[0].Rank = 2
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "stage score") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("tampered hybrid source order", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		hybrid.PerQuery[0].Hits[0].StageScores[0], hybrid.PerQuery[0].Hits[0].StageScores[1] =
			hybrid.PerQuery[0].Hits[0].StageScores[1], hybrid.PerQuery[0].Hits[0].StageScores[0]
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "ordered fts then dense") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("tampered hybrid rrf score", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		hybrid.PerQuery[0].Hits[0].StageScores[2].Score = 0.5
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "rrf stage") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("tampered hybrid final and rrf scores", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		hybrid.PerQuery[0].Hits[0].Score = 0.5
		hybrid.PerQuery[0].Hits[0].StageScores[2].Score = 0.5
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "source ranks and rrf_k") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("source rank exceeds candidate depth", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		hybrid.PerQuery[0].Hits[0].StageScores[0].Rank = 21
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "exceeds candidate depth") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("nested dense config drift", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		hybrid.Retriever.Config["dense"].(map[string]any)["profile_id"] = "different-profile"
		rehashManifestConfig(t, &hybrid)
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "nested dense config differs") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("tampered dense trace order", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		dense.PerQuery[0].Traces[0], dense.PerQuery[0].Traces[1] = dense.PerQuery[0].Traces[1], dense.PerQuery[0].Traces[0]
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "trace stages") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("missing hybrid candidate set", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		hybrid.PerQuery[0].CandidateSets = hybrid.PerQuery[0].CandidateSets[:1]
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "candidate sets") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("hybrid source rank differs from candidate evidence", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		hybrid.PerQuery[0].CandidateSets[0].Hits[0].ChunkID = "different-chunk"
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "recorded candidate set") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
	t.Run("forbidden chunk hidden in branch candidates", func(t *testing.T) {
		fts, dense, hybrid := newTriplet(t)
		hybrid.PerQuery[0].CandidateSets[0].Hits = append(hybrid.PerQuery[0].CandidateSets[0].Hits,
			model.CandidateHit{ChunkID: hybrid.PerQuery[0].ForbiddenChunkIDs[0], Rank: 2})
		if _, err := CompareThreeManifests(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), "candidate evidence contains forbidden") {
			t.Fatalf("CompareThreeManifests() error = %v", err)
		}
	})
}

func TestValidateHybridQueryEvidenceRejectsDuplicateSourceRanksAndWrongFinalOrder(t *testing.T) {
	t.Parallel()

	config := map[string]any{"rrf_k": 60, "fts_candidate_k": 20, "dense_candidate_k": 20}
	tests := []struct {
		name  string
		hits  []HitRecord
		match string
	}{
		{
			name: "duplicate source rank",
			hits: []HitRecord{
				hybridEvidenceRecord("chunk-a", 1, "fts", 1),
				hybridEvidenceRecord("chunk-b", 2, "fts", 1),
			},
			match: "appears more than once",
		},
		{
			name: "wrong final order",
			hits: []HitRecord{
				hybridEvidenceRecord("chunk-b", 1, "fts", 2),
				hybridEvidenceRecord("chunk-a", 2, "fts", 1),
			},
			match: "ordered by score descending",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if err := validateHybridQueryEvidence(QueryResult{Hits: test.hits}, config, 5); err == nil || !strings.Contains(err.Error(), test.match) {
				t.Fatalf("validateHybridQueryEvidence() error = %v", err)
			}
		})
	}
}

func TestValidateHybridQueryEvidenceUsesExactFloatOrderBeforeChunkIDTieBreak(t *testing.T) {
	t.Parallel()

	config := map[string]any{"rrf_k": 60, "fts_candidate_k": 20, "dense_candidate_k": 20}
	hits := []HitRecord{
		hybridDualEvidenceRecord("chunk-z", 1, 6, 39),
		hybridDualEvidenceRecord("chunk-a", 2, 28, 12),
	}
	if !(hits[0].Score > hits[1].Score && hits[0].Score-hits[1].Score < 1e-12) {
		t.Fatalf("test fixture must contain distinct near-equal scores: %.18f %.18f", hits[0].Score, hits[1].Score)
	}
	if err := validateHybridQueryEvidence(QueryResult{Hits: hits}, config, 50); err != nil {
		t.Fatalf("validateHybridQueryEvidence() error = %v", err)
	}
}

func TestThreeWayCategoriesAreStableAndMechanicallyAggregated(t *testing.T) {
	t.Parallel()

	fts := Manifest{PerQuery: []QueryResult{
		{Category: "semantic", Metrics: RankingMetrics{}},
		{Category: "exact-identifier", Metrics: RankingMetrics{RecallAtK: 1}},
		{Category: "semantic", Metrics: RankingMetrics{RecallAtK: 1}},
	}}
	dense := fts
	hybrid := fts
	dense.PerQuery = append([]QueryResult(nil), fts.PerQuery...)
	hybrid.PerQuery = append([]QueryResult(nil), fts.PerQuery...)
	dense.PerQuery[0].Metrics.RecallAtK = 1
	hybrid.PerQuery[0].Metrics.RecallAtK = 1

	categories, err := threeWayCategories(fts, dense, hybrid)
	if err != nil {
		t.Fatalf("threeWayCategories() error = %v", err)
	}
	if len(categories) != 2 || categories[0].Category != "exact-identifier" || categories[1].Category != "semantic" {
		t.Fatalf("category order = %+v", categories)
	}
	if categories[1].QueryCount != 2 || categories[1].FTS.RecallAtK != 0.5 || categories[1].Dense.RecallAtK != 1 || categories[1].Hybrid.RecallAtK != 1 {
		t.Fatalf("semantic aggregation = %+v", categories[1])
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

func setManifestMode(t *testing.T, manifest *Manifest, mode string) {
	t.Helper()
	manifest.Retriever.Config["mode"] = mode
	switch mode {
	case "fts":
		manifest.PerQuery[0].Traces = []model.StageTrace{{Stage: "fts"}}
		for index := range manifest.PerQuery[0].Hits {
			hit := &manifest.PerQuery[0].Hits[index]
			hit.StageScores = []model.StageScore{{Stage: "fts", Rank: hit.Rank, Score: hit.Score}}
		}
		manifest.PerQuery[0].CandidateSets = []model.CandidateSet{candidateSetFromRecords("fts", manifest.PerQuery[0].Hits)}
	case "dense":
		manifest.PerQuery[0].Traces = []model.StageTrace{{Stage: "query_embedding"}, {Stage: "dense"}}
		for index := range manifest.PerQuery[0].Hits {
			hit := &manifest.PerQuery[0].Hits[index]
			hit.StageScores = []model.StageScore{{Stage: "dense", Rank: hit.Rank, Score: hit.Score}}
		}
		manifest.PerQuery[0].CandidateSets = []model.CandidateSet{candidateSetFromRecords("dense", manifest.PerQuery[0].Hits)}
	case "hybrid":
		manifest.Retriever.Config["fusion"] = "reciprocal_rank_fusion"
		manifest.Retriever.Config["branch_failure"] = "fail_closed"
		manifest.Retriever.Config["rrf_k"] = 60
		manifest.Retriever.Config["fts_candidate_k"] = 20
		manifest.Retriever.Config["dense_candidate_k"] = 20
		manifest.Retriever.Config["fts"] = map[string]any{"mode": "fts"}
		manifest.Retriever.Config["dense"] = map[string]any{"mode": "dense"}
		manifest.PerQuery[0].Traces = []model.StageTrace{
			{Stage: "fts"},
			{Stage: "query_embedding"},
			{Stage: "dense"},
			{Stage: "rrf_fusion"},
		}
		for index := range manifest.PerQuery[0].Hits {
			hit := &manifest.PerQuery[0].Hits[index]
			hit.Score = 2.0 / 61
			hit.StageScores = []model.StageScore{
				{Stage: "fts", Rank: 1, Score: 1},
				{Stage: "dense", Rank: 1, Score: 1},
				{Stage: "rrf", Rank: hit.Rank, Score: hit.Score},
			}
		}
		manifest.PerQuery[0].CandidateSets = []model.CandidateSet{
			candidateSetFromSourceScores("fts", manifest.PerQuery[0].Hits),
			candidateSetFromSourceScores("dense", manifest.PerQuery[0].Hits),
		}
	}
	_, configSHA256, err := normalizeAndHashConfig(manifest.Retriever.Config)
	if err != nil {
		t.Fatalf("normalizeAndHashConfig() error = %v", err)
	}
	manifest.Retriever.ConfigSHA256 = configSHA256
}

func candidateSetFromRecords(stage string, hits []HitRecord) model.CandidateSet {
	result := model.CandidateSet{Stage: stage, Hits: make([]model.CandidateHit, 0, len(hits))}
	for _, hit := range hits {
		result.Hits = append(result.Hits, model.CandidateHit{ChunkID: hit.ChunkID, Rank: hit.Rank})
	}
	return result
}

func candidateSetFromSourceScores(stage string, hits []HitRecord) model.CandidateSet {
	byRank := make(map[int]string)
	maxRank := 0
	for _, hit := range hits {
		for _, score := range hit.StageScores {
			if score.Stage == stage {
				byRank[score.Rank] = hit.ChunkID
				maxRank = max(maxRank, score.Rank)
			}
		}
	}
	result := model.CandidateSet{Stage: stage, Hits: make([]model.CandidateHit, 0, maxRank)}
	for rank := 1; rank <= maxRank; rank++ {
		chunkID := byRank[rank]
		if chunkID == "" {
			chunkID = fmt.Sprintf("%s-distractor-%d", stage, rank)
		}
		result.Hits = append(result.Hits, model.CandidateHit{ChunkID: chunkID, Rank: rank})
	}
	return result
}

func rehashManifestConfig(t *testing.T, manifest *Manifest) {
	t.Helper()
	_, configSHA256, err := normalizeAndHashConfig(manifest.Retriever.Config)
	if err != nil {
		t.Fatalf("normalizeAndHashConfig() error = %v", err)
	}
	manifest.Retriever.ConfigSHA256 = configSHA256
}

func hybridEvidenceRecord(chunkID string, finalRank int, stage string, sourceRank int) HitRecord {
	score := 1 / float64(60+sourceRank)
	return HitRecord{
		Rank: finalRank,
		SearchHit: model.SearchHit{
			ChunkID: chunkID,
			Score:   score,
			StageScores: []model.StageScore{
				{Stage: stage, Rank: sourceRank, Score: 1},
				{Stage: "rrf", Rank: finalRank, Score: score},
			},
		},
	}
}

func hybridDualEvidenceRecord(chunkID string, finalRank, ftsRank, denseRank int) HitRecord {
	score := 1/float64(60+ftsRank) + 1/float64(60+denseRank)
	return HitRecord{
		Rank: finalRank,
		SearchHit: model.SearchHit{
			ChunkID: chunkID,
			Score:   score,
			StageScores: []model.StageScore{
				{Stage: "fts", Rank: ftsRank, Score: 1},
				{Stage: "dense", Rank: denseRank, Score: 1},
				{Stage: "rrf", Rank: finalRank, Score: score},
			},
		},
	}
}
