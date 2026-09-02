package eval

import (
	"strings"
	"testing"

	"github.com/ksana-ai/raghub/internal/model"
)

func TestDiagnoseCandidateCoverageClassifiesRerankerOpportunity(t *testing.T) {
	t.Parallel()

	gold := []string{"gold-a", "gold-b"}
	fts := diagnosisManifest(t, "fts", 20, gold, []string{"gold-a"})
	dense := diagnosisManifest(t, "dense", 20, gold, []string{"gold-b"})
	hybrid := diagnosisManifest(t, "hybrid", 5, gold, nil)
	hybrid.PerQuery[0].CandidateSets = []model.CandidateSet{
		{Stage: "fts", Hits: []model.CandidateHit{{ChunkID: "gold-a", Rank: 1}}},
		{Stage: "dense", Hits: []model.CandidateHit{{ChunkID: "gold-b", Rank: 1}}},
	}

	diagnosis, err := DiagnoseCandidateCoverage(fts, dense, hybrid)
	if err != nil {
		t.Fatalf("DiagnoseCandidateCoverage() error = %v", err)
	}
	if diagnosis.SchemaVersion != CandidateDiagnosisSchemaVersion || diagnosis.Status != SmokeStatus {
		t.Fatalf("unexpected diagnosis identity: %+v", diagnosis)
	}
	if diagnosis.QueryCount != 1 || diagnosis.FinalTopK != 5 || diagnosis.FTSCandidateK != 20 || diagnosis.DenseCandidateK != 20 {
		t.Fatalf("unexpected diagnosis dimensions: %+v", diagnosis)
	}
	query := diagnosis.PerQuery[0]
	if query.Classification != DiagnosisFusionOrderingGap || query.HybridRecall != 0 || query.UnionCandidateRecall != 1 {
		t.Fatalf("unexpected query diagnosis: %+v", query)
	}
	if diagnosis.Summary.FusionOrderingGapQueries != 1 || diagnosis.Summary.RecoverableMissingGoldCount != 2 || diagnosis.Summary.UnionCandidateRecall != 1 {
		t.Fatalf("unexpected diagnosis summary: %+v", diagnosis.Summary)
	}
	if diagnosis.RerankerGate.Eligible || diagnosis.RerankerGate.IncompleteQueries != 1 || diagnosis.RerankerGate.RecoverableMissingFraction != 1 {
		t.Fatalf("unexpected reranker gate: %+v", diagnosis.RerankerGate)
	}
	data, err := MarshalCandidateDiagnosis(diagnosis)
	if err != nil || !strings.HasSuffix(string(data), "\n") {
		t.Fatalf("MarshalCandidateDiagnosis() error=%v data=%q", err, data)
	}
}

func TestRerankerExperimentGateUsesPreregisteredThresholds(t *testing.T) {
	t.Parallel()

	gate := rerankerExperimentGate(CandidateDiagnosisCounts{
		QueryCount: 5, CompleteQueries: 2, MissingGoldCount: 4, RecoverableMissingGoldCount: 2,
	})
	if !gate.Eligible || gate.MinimumIncompleteQueries != 3 || gate.MinimumRecoverableFraction != 0.5 ||
		gate.IncompleteQueries != 3 || gate.RecoverableMissingFraction != 0.5 {
		t.Fatalf("unexpected eligible gate: %+v", gate)
	}
	below := rerankerExperimentGate(CandidateDiagnosisCounts{
		QueryCount: 5, CompleteQueries: 3, MissingGoldCount: 2, RecoverableMissingGoldCount: 2,
	})
	if below.Eligible {
		t.Fatalf("two incomplete queries must not pass gate: %+v", below)
	}
}

func TestDiagnoseCandidateCoverageClassifiesCompleteMixedAndCandidateGaps(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		gold       []string
		ftsHits    []string
		denseHits  []string
		hybridHits []string
		want       string
	}{
		{name: "complete", gold: []string{"gold-a"}, hybridHits: []string{"gold-a"}, want: DiagnosisComplete},
		{name: "candidate generation", gold: []string{"gold-a"}, want: DiagnosisCandidateGeneration},
		{name: "mixed", gold: []string{"gold-a", "gold-b"}, ftsHits: []string{"gold-a"}, want: DiagnosisMixedGap},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			fts := diagnosisManifest(t, "fts", 20, test.gold, test.ftsHits)
			dense := diagnosisManifest(t, "dense", 20, test.gold, test.denseHits)
			hybrid := diagnosisManifest(t, "hybrid", 5, test.gold, test.hybridHits)
			if test.want == DiagnosisMixedGap {
				hybrid.PerQuery[0].CandidateSets = []model.CandidateSet{
					{Stage: "fts", Hits: []model.CandidateHit{{ChunkID: "gold-a", Rank: 1}}},
					{Stage: "dense", Hits: []model.CandidateHit{}},
				}
			}
			diagnosis, err := DiagnoseCandidateCoverage(fts, dense, hybrid)
			if err != nil {
				t.Fatal(err)
			}
			if diagnosis.PerQuery[0].Classification != test.want {
				t.Fatalf("classification = %q, want %q", diagnosis.PerQuery[0].Classification, test.want)
			}
		})
	}
}

func TestDiagnoseCandidateCoverageRejectsUntrustedOrMisalignedRuns(t *testing.T) {
	t.Parallel()

	newTriplet := func(t *testing.T) (Manifest, Manifest, Manifest) {
		t.Helper()
		gold := []string{"gold-a"}
		return diagnosisManifest(t, "fts", 20, gold, gold),
			diagnosisManifest(t, "dense", 20, gold, gold),
			diagnosisManifest(t, "hybrid", 5, gold, gold)
	}
	tests := []struct {
		name   string
		mutate func(*Manifest, *Manifest, *Manifest)
		want   string
	}{
		{name: "dirty revision", mutate: func(fts, _, _ *Manifest) { fts.Runtime.CodeRevision += "+dirty" }, want: "clean committed revision"},
		{name: "dataset drift", mutate: func(_, dense, _ *Manifest) { dense.Dataset.SHA256 = strings.Repeat("b", 64) }, want: "dataset"},
		{name: "corpus drift", mutate: func(_, dense, _ *Manifest) { dense.CorpusSHA256 = strings.Repeat("b", 64) }, want: "corpus_sha256"},
		{name: "candidate depth", mutate: func(fts, _, _ *Manifest) { fts.TopK = 19 }, want: "effective candidate depth"},
		{name: "nested profile drift", mutate: func(_, _, hybrid *Manifest) {
			hybrid.Retriever.Config["dense"].(map[string]any)["profile"] = "other"
			rehashManifestConfig(t, hybrid)
		}, want: "nested dense config differs"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			fts, dense, hybrid := newTriplet(t)
			test.mutate(&fts, &dense, &hybrid)
			if _, err := DiagnoseCandidateCoverage(fts, dense, hybrid); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("DiagnoseCandidateCoverage() error = %v, want %q", err, test.want)
			}
		})
	}
}

func diagnosisManifest(t *testing.T, mode string, topK int, gold, hitIDs []string) Manifest {
	t.Helper()
	metrics := EvaluateRanking(hitIDs, gold, topK)
	manifest := comparableManifest(t, mode+"-run", map[string]string{
		"fts": "postgres_fts", "dense": "postgres_dense", "hybrid": "postgres_hybrid_rrf",
	}[mode], metrics, LatencyPercentiles{})
	manifest.TopK = topK
	manifest.PerQuery[0].GoldChunkIDs = append([]string(nil), gold...)
	manifest.PerQuery[0].ForbiddenChunkIDs = nil
	manifest.PerQuery[0].Hits = make([]HitRecord, 0, len(hitIDs))
	for index, chunkID := range hitIDs {
		rank := index + 1
		score := 1 / float64(rank)
		hit := HitRecord{Rank: rank, SearchHit: model.SearchHit{ChunkID: chunkID, Score: score}}
		switch mode {
		case "fts", "dense":
			hit.StageScores = []model.StageScore{{Stage: mode, Rank: rank, Score: score}}
		case "hybrid":
			hit.Score = 2 / float64(60+rank)
			hit.StageScores = []model.StageScore{
				{Stage: "fts", Rank: rank, Score: score},
				{Stage: "dense", Rank: rank, Score: score},
				{Stage: "rrf", Rank: rank, Score: hit.Score},
			}
		}
		manifest.PerQuery[0].Hits = append(manifest.PerQuery[0].Hits, hit)
	}
	manifest.PerQuery[0].Metrics = metrics
	manifest.Summary.Metrics = metrics
	setManifestMode(t, &manifest, mode)
	return manifest
}
