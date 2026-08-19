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
	return model.SearchResult{Hits: []model.SearchHit{{ChunkID: "doc:v000001:c0000"}}}, nil
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
		RetrieverConfig: map[string]any{"mode": string(mode)}, Command: "raghub-eval",
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
