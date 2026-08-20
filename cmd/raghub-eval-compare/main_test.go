package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	evalrun "raghub/internal/eval"
	"raghub/internal/model"
)

type testIngestor struct{}

func (testIngestor) Ingest(_ context.Context, document model.DocumentInput) (model.IngestResult, error) {
	return model.IngestResult{
		TenantID: document.TenantID, DocumentID: document.ID, Version: 1,
		ChunkIDs: []string{document.ID + ":v000001:c0000"},
	}, nil
}

type testSearcher struct{}

func (testSearcher) Search(_ context.Context, request model.SearchRequest) (model.SearchResult, error) {
	hit := model.SearchHit{ChunkID: "doc:v000001:c0000", Score: 1}
	switch request.Mode {
	case model.SearchModeDense:
		hit.StageScores = []model.StageScore{{Stage: "dense", Rank: 1, Score: 1}}
		return model.SearchResult{
			Hits:          []model.SearchHit{hit},
			CandidateSets: []model.CandidateSet{{Stage: "dense", Hits: []model.CandidateHit{{ChunkID: hit.ChunkID, Rank: 1}}}},
			Traces: []model.StageTrace{
				{Stage: "query_embedding"},
				{Stage: "dense"},
			},
		}, nil
	case model.SearchModeHybrid:
		hit.Score = 2.0 / 61
		hit.StageScores = []model.StageScore{
			{Stage: "fts", Rank: 1, Score: 1},
			{Stage: "dense", Rank: 1, Score: 1},
			{Stage: "rrf", Rank: 1, Score: hit.Score},
		}
		return model.SearchResult{
			Hits: []model.SearchHit{hit},
			CandidateSets: []model.CandidateSet{
				{Stage: "fts", Hits: []model.CandidateHit{{ChunkID: hit.ChunkID, Rank: 1}}},
				{Stage: "dense", Hits: []model.CandidateHit{{ChunkID: hit.ChunkID, Rank: 1}}},
			},
			Traces: []model.StageTrace{
				{Stage: "fts"},
				{Stage: "query_embedding"},
				{Stage: "dense"},
				{Stage: "rrf_fusion"},
			},
		}, nil
	default:
		hit.StageScores = []model.StageScore{{Stage: "fts", Rank: 1, Score: 1}}
		return model.SearchResult{
			Hits:          []model.SearchHit{hit},
			CandidateSets: []model.CandidateSet{{Stage: "fts", Hits: []model.CandidateHit{{ChunkID: hit.ChunkID, Rank: 1}}}},
			Traces:        []model.StageTrace{{Stage: "fts"}},
		}, nil
	}
}

type testInspector struct{}

func (testInspector) ActiveChunkInventory(_ context.Context, _ []string) ([]model.ActiveChunkInventoryEntry, error) {
	return []model.ActiveChunkInventoryEntry{{
		TenantID: "tenant", DocumentID: "doc", DocumentVersion: 1,
		DocumentFingerprint: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		ChunkID:             "doc:v000001:c0000",
		RawTextSHA256:       "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		IndexedTextSHA256:   "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
	}}, nil
}

func TestRunWritesStrictSmokeComparison(t *testing.T) {
	t.Parallel()

	directory := t.TempDir()
	baselinePath := filepath.Join(directory, "baseline.json")
	candidatePath := filepath.Join(directory, "candidate.json")
	outputPath := filepath.Join(directory, "comparison.json")
	writeTestManifest(t, baselinePath, testManifest(t, "fts", model.SearchModeFTS, 5))
	writeTestManifest(t, candidatePath, testManifest(t, "dense", model.SearchModeDense, 5))

	var stdout, stderr bytes.Buffer
	code := run([]string{"-baseline", baselinePath, "-candidate", candidatePath, "-output", outputPath}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	var comparison evalrun.Comparison
	if err := json.Unmarshal(data, &comparison); err != nil {
		t.Fatalf("decode comparison: %v", err)
	}
	if comparison.SchemaVersion != evalrun.ComparisonSchemaVersion || comparison.Status != evalrun.SmokeStatus {
		t.Fatalf("comparison escaped smoke scope: %+v", comparison)
	}
	if comparison.Baseline.Retriever.Name != "fts" || comparison.Candidate.Retriever.Name != "dense" {
		t.Fatalf("missing paired retriever provenance: %+v", comparison)
	}
}

func TestRunWritesStrictThreeWaySmokeComparison(t *testing.T) {
	t.Parallel()

	directory := t.TempDir()
	ftsPath := filepath.Join(directory, "fts.json")
	densePath := filepath.Join(directory, "dense.json")
	hybridPath := filepath.Join(directory, "hybrid.json")
	outputPath := filepath.Join(directory, "comparison.json")
	writeTestManifest(t, ftsPath, testManifest(t, "postgres_fts", model.SearchModeFTS, 5))
	writeTestManifest(t, densePath, testManifest(t, "postgres_dense", model.SearchModeDense, 5))
	writeTestManifest(t, hybridPath, testManifest(t, "postgres_hybrid_rrf", model.SearchModeHybrid, 5))

	var stdout, stderr bytes.Buffer
	code := run([]string{
		"-fts", ftsPath,
		"-dense", densePath,
		"-hybrid", hybridPath,
		"-output", outputPath,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	var comparison evalrun.ThreeWayComparison
	if err := json.Unmarshal(data, &comparison); err != nil {
		t.Fatalf("decode three-way comparison: %v", err)
	}
	if comparison.SchemaVersion != evalrun.ThreeWayComparisonSchemaVersion || comparison.Status != evalrun.SmokeStatus {
		t.Fatalf("comparison escaped smoke scope: %+v", comparison)
	}
	if comparison.FTS.Retriever.Name != "postgres_fts" || comparison.Dense.Retriever.Name != "postgres_dense" || comparison.Hybrid.Retriever.Name != "postgres_hybrid_rrf" {
		t.Fatalf("missing three-way retriever provenance: %+v", comparison)
	}
}

func TestRunWritesCandidateDiagnosis(t *testing.T) {
	t.Parallel()

	directory := t.TempDir()
	ftsPath := filepath.Join(directory, "fts-candidates.json")
	densePath := filepath.Join(directory, "dense-candidates.json")
	hybridPath := filepath.Join(directory, "hybrid-final.json")
	outputPath := filepath.Join(directory, "diagnosis.json")
	writeTestManifest(t, ftsPath, testManifest(t, "postgres_fts", model.SearchModeFTS, 20))
	writeTestManifest(t, densePath, testManifest(t, "postgres_dense", model.SearchModeDense, 20))
	writeTestManifest(t, hybridPath, testManifest(t, "postgres_hybrid_rrf", model.SearchModeHybrid, 5))

	var stdout, stderr bytes.Buffer
	code := run([]string{
		"-candidate-diagnosis",
		"-fts", ftsPath,
		"-dense", densePath,
		"-hybrid", hybridPath,
		"-output", outputPath,
	}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	var diagnosis evalrun.CandidateDiagnosis
	if err := json.Unmarshal(data, &diagnosis); err != nil {
		t.Fatalf("decode candidate diagnosis: %v", err)
	}
	if diagnosis.SchemaVersion != evalrun.CandidateDiagnosisSchemaVersion || diagnosis.Summary.CompleteQueries != 1 {
		t.Fatalf("unexpected candidate diagnosis: %+v", diagnosis)
	}
}

func TestRunRejectsMixedOrIncompleteComparisonModes(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name string
		args []string
		want string
	}{
		{name: "mixed", args: []string{"-baseline", "fts.json", "-candidate", "dense.json", "-hybrid", "hybrid.json"}, want: "cannot be combined"},
		{name: "incomplete three-way", args: []string{"-fts", "fts.json", "-dense", "dense.json"}, want: "are all required"},
		{name: "diagnosis with pair", args: []string{"-candidate-diagnosis", "-baseline", "fts.json", "-candidate", "dense.json"}, want: "requires -fts"},
		{name: "no mode", args: nil, want: "provide either"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			var stdout, stderr bytes.Buffer
			if code := run(test.args, &stdout, &stderr); code != 2 || !bytes.Contains(stderr.Bytes(), []byte(test.want)) {
				t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
			}
		})
	}
}

func TestRunRejectsMismatchWithoutWritingOutput(t *testing.T) {
	t.Parallel()

	directory := t.TempDir()
	baselinePath := filepath.Join(directory, "baseline.json")
	candidatePath := filepath.Join(directory, "candidate.json")
	outputPath := filepath.Join(directory, "comparison.json")
	writeTestManifest(t, baselinePath, testManifest(t, "fts", model.SearchModeFTS, 5))
	writeTestManifest(t, candidatePath, testManifest(t, "dense", model.SearchModeDense, 10))

	var stdout, stderr bytes.Buffer
	code := run([]string{"-baseline", baselinePath, "-candidate", candidatePath, "-output", outputPath}, &stdout, &stderr)
	if code != 1 || !bytes.Contains(stderr.Bytes(), []byte("top_k differs")) {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
	if _, err := os.Stat(outputPath); !os.IsNotExist(err) {
		t.Fatalf("output must not exist after rejected pairing; stat error=%v", err)
	}
}

func TestRunRefusesToOverwriteInput(t *testing.T) {
	t.Parallel()

	var stdout, stderr bytes.Buffer
	code := run([]string{"-baseline", "same.json", "-candidate", "candidate.json", "-output", "same.json"}, &stdout, &stderr)
	if code != 2 || !bytes.Contains(stderr.Bytes(), []byte("must not overwrite")) {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
}

func TestWriteOutputReplacesComparisonAtomically(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "comparison.json")
	if err := os.WriteFile(path, []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := writeOutput(path, []byte("new\n"), &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "new\n" {
		t.Fatalf("comparison = %q, want new output", data)
	}
}

func testManifest(t *testing.T, retrieverName string, mode model.SearchMode, topK int) evalrun.Manifest {
	t.Helper()
	retrieverConfig := map[string]any{"mode": string(mode)}
	if mode == model.SearchModeHybrid {
		retrieverConfig["fusion"] = "reciprocal_rank_fusion"
		retrieverConfig["branch_failure"] = "fail_closed"
		retrieverConfig["rrf_k"] = 60
		retrieverConfig["fts_candidate_k"] = 20
		retrieverConfig["dense_candidate_k"] = 20
		retrieverConfig["fts"] = map[string]any{"mode": "fts"}
		retrieverConfig["dense"] = map[string]any{"mode": "dense"}
	}
	dataset := evalrun.Dataset{
		SchemaVersion: evalrun.DatasetSchemaVersion,
		Name:          "paired",
		Version:       "2.0.0",
		Documents: []evalrun.DatasetDocument{{
			TenantID: "tenant", DocumentID: "doc", Title: "Doc", Content: "body",
		}},
		Queries: []evalrun.QueryCase{{
			ID: "q1", Category: "exact", TenantID: "tenant", Query: "body",
			GoldChunkIDs: []string{"doc:v000001:c0000"},
		}},
	}
	manifest, err := evalrun.NewRunner(testIngestor{}, testSearcher{}, testInspector{}).Run(context.Background(), evalrun.LoadedDataset{
		Dataset: dataset, SHA256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
	}, evalrun.Options{
		TopK: topK, SearchMode: mode, RetrieverName: retrieverName,
		RetrieverConfig: retrieverConfig, Command: "raghub-eval",
		DatabaseVersion: "test-db", VectorExtensionVersion: "0.8.1", CodeRevision: "test-revision",
	})
	if err != nil {
		t.Fatalf("build test manifest: %v", err)
	}
	return manifest
}

func writeTestManifest(t *testing.T, path string, manifest evalrun.Manifest) {
	t.Helper()
	data, err := evalrun.MarshalManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
