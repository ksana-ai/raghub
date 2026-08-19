package eval

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"raghub/internal/model"
)

type fakeIngestor struct {
	documents          []model.DocumentInput
	err                error
	version            int
	chunkIDsByDocument map[string][]string
	resultTenantID     string
	resultDocumentID   string
}

func (f *fakeIngestor) Ingest(_ context.Context, document model.DocumentInput) (model.IngestResult, error) {
	f.documents = append(f.documents, document)
	if f.err != nil {
		return model.IngestResult{}, f.err
	}
	version := f.version
	if version == 0 {
		version = 1
	}
	chunkIDs := f.chunkIDsByDocument[document.ID]
	if chunkIDs == nil {
		chunkIDs = []string{document.ID + ":v000001:c0000"}
	}
	tenantID := document.TenantID
	if f.resultTenantID != "" {
		tenantID = f.resultTenantID
	}
	documentID := document.ID
	if f.resultDocumentID != "" {
		documentID = f.resultDocumentID
	}
	return model.IngestResult{
		TenantID:   tenantID,
		DocumentID: documentID,
		Version:    version,
		ChunkIDs:   append([]string(nil), chunkIDs...),
		CreatedAt:  time.Unix(1, 0),
	}, nil
}

type fakeSearcher struct {
	results  map[string]model.SearchResult
	errors   map[string]error
	requests []model.SearchRequest
}

func (f *fakeSearcher) Search(_ context.Context, request model.SearchRequest) (model.SearchResult, error) {
	f.requests = append(f.requests, request)
	if err := f.errors[request.Query]; err != nil {
		return model.SearchResult{}, err
	}
	return f.results[request.Query], nil
}

func TestRunnerProducesSmokeManifest(t *testing.T) {
	t.Parallel()

	dataset := Dataset{
		SchemaVersion: DatasetSchemaVersion,
		Name:          "smoke-test",
		Version:       "1.0.0",
		Documents: []DatasetDocument{{
			TenantID:   "tenant-a",
			DocumentID: "doc-a",
			Title:      "A",
			SourceURI:  "fixture://a",
			Content:    "alpha",
			Metadata:   json.RawMessage(`{"kind":"test"}`),
		}},
		Queries: []QueryCase{
			{ID: "q1", Category: "exact", TenantID: "tenant-a", Query: "first", GoldChunkIDs: []string{"doc-a:v000001:c0000"}},
			{ID: "q2", Category: "exact", TenantID: "tenant-a", Query: "second", GoldChunkIDs: []string{"doc-a:v000001:c0000"}},
		},
	}
	ingestor := &fakeIngestor{}
	searcher := &fakeSearcher{results: map[string]model.SearchResult{
		"first":  {Hits: []model.SearchHit{{ChunkID: "doc-a:v000001:c0000"}}},
		"second": {Hits: []model.SearchHit{{ChunkID: "doc-a:v000001:c0000"}}},
	}, errors: map[string]error{}}

	manifest, err := NewRunner(ingestor, searcher).Run(context.Background(), LoadedDataset{
		Dataset: dataset,
		SHA256:  "abc123",
	}, Options{
		TopK:            5,
		RetrieverName:   "fake",
		RetrieverConfig: map[string]any{"mode": "fts"},
		DatabaseVersion: "test-db",
		CodeRevision:    "deadbeef",
		Command:         "raghub-eval -dataset smoke.json",
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if manifest.RunID == "" || manifest.Status != SmokeStatus || manifest.Dataset.SHA256 != "abc123" || manifest.CorpusSHA256 != "abc123" {
		t.Fatalf("unexpected manifest identity: %+v", manifest)
	}
	if manifest.Command != "raghub-eval -dataset smoke.json" || manifest.Retriever.ConfigSHA256 == "" {
		t.Fatalf("missing command/config provenance: %+v", manifest)
	}
	if len(ingestor.documents) != 1 || len(manifest.Ingestions) != 1 {
		t.Fatalf("ingestions = %d/%d, want 1/1", len(ingestor.documents), len(manifest.Ingestions))
	}
	if manifest.Summary.QueryCount != 2 || manifest.Summary.SearchErrorCount != 0 {
		t.Fatalf("unexpected summary counts: %+v", manifest.Summary)
	}
	if manifest.Summary.ForbiddenHitCount != 0 || !manifest.Summary.Gates.Pass {
		t.Fatalf("unexpected safety gates: %+v", manifest.Summary)
	}
	assertClose(t, "mean recall", manifest.Summary.Metrics.RecallAtK, 1)
	assertClose(t, "mean hit rate", manifest.Summary.Metrics.HitRateAtK, 1)
	assertClose(t, "mean mrr", manifest.Summary.Metrics.MRR, 1)
	if manifest.Runtime.DatabaseVersion != "test-db" || manifest.Runtime.CodeRevision != "deadbeef" {
		t.Fatalf("unexpected runtime provenance: %+v", manifest.Runtime)
	}
	data, err := MarshalManifest(manifest)
	if err != nil {
		t.Fatalf("MarshalManifest() error = %v", err)
	}
	if !strings.Contains(string(data), `"unchanged": false`) {
		t.Fatalf("manifest must make first-ingest idempotency state explicit: %s", data)
	}
}

func TestRunnerMakesSearchErrorsIncomplete(t *testing.T) {
	t.Parallel()

	dataset := Dataset{
		Name: "test", Version: "1", Documents: []DatasetDocument{{
			TenantID: "tenant-a", DocumentID: "doc-a", Title: "A", Content: "alpha",
		}}, Queries: []QueryCase{{
			ID: "q1", Category: "error", TenantID: "tenant-a", Query: "broken", GoldChunkIDs: []string{"doc-a:v000001:c0000"},
		}},
	}
	searcher := &fakeSearcher{results: map[string]model.SearchResult{}, errors: map[string]error{"broken": errors.New("backend unavailable")}}
	manifest, err := NewRunner(&fakeIngestor{}, searcher).Run(context.Background(), LoadedDataset{Dataset: dataset, SHA256: "x"}, Options{TopK: 5, RetrieverName: "fake", Command: "raghub-eval"})
	if err == nil {
		t.Fatal("Run() error = nil, want incomplete evaluation error")
	}
	if manifest.Summary.SearchErrorCount != 1 || manifest.PerQuery[0].Error == "" {
		t.Fatalf("search error not recorded: %+v", manifest)
	}
	if manifest.Status != IncompleteStatus || manifest.Summary.Gates.Pass || manifest.Summary.Gates.SearchCompleted {
		t.Fatalf("search error must fail completeness gates: %+v", manifest)
	}
	if !manifest.Summary.Gates.CorpusReferencesValid || !manifest.Summary.Gates.NoForbiddenHits {
		t.Fatalf("unrelated gates should retain their result: %+v", manifest.Summary.Gates)
	}
	if manifest.Summary.Metrics != (RankingMetrics{}) {
		t.Fatalf("metrics = %+v, want zeros", manifest.Summary.Metrics)
	}
}

func TestRunnerRejectsInactiveDatasetReferencesBeforeSearch(t *testing.T) {
	t.Parallel()

	dataset := Dataset{
		Name: "test", Version: "1", Documents: []DatasetDocument{{
			TenantID: "tenant-a", DocumentID: "doc-a", Title: "A", Content: "alpha",
		}}, Queries: []QueryCase{{
			ID: "q1", Category: "stale", TenantID: "tenant-a", Query: "alpha",
			GoldChunkIDs: []string{"doc-a:v000001:c0000"}, ForbiddenChunkIDs: []string{"other:v000001:c0000"},
		}},
	}
	ingestor := &fakeIngestor{
		version:            2,
		chunkIDsByDocument: map[string][]string{"doc-a": {"doc-a:v000002:c0000"}},
	}
	searcher := &fakeSearcher{results: map[string]model.SearchResult{}, errors: map[string]error{}}
	manifest, err := NewRunner(ingestor, searcher).Run(context.Background(), LoadedDataset{Dataset: dataset, SHA256: "x"}, Options{
		TopK: 5, RetrieverName: "fake", Command: "raghub-eval",
	})
	if err == nil || !strings.Contains(err.Error(), "outside this run's active corpus") {
		t.Fatalf("Run() error = %v, want inactive-reference failure", err)
	}
	if !strings.Contains(err.Error(), "kind=gold") || !strings.Contains(err.Error(), "kind=forbidden") {
		t.Fatalf("Run() error must identify all stale reference kinds: %v", err)
	}
	if manifest.Status != IncompleteStatus || manifest.Summary.Gates.CorpusReferencesValid {
		t.Fatalf("stale corpus must be incomplete: %+v", manifest)
	}
	if len(searcher.requests) != 0 {
		t.Fatalf("search calls = %d, want 0 before corpus validation", len(searcher.requests))
	}
}

func TestRunnerForbiddenTailHitFailsSafetyGate(t *testing.T) {
	t.Parallel()

	dataset := Dataset{
		Name: "test", Version: "1", Documents: []DatasetDocument{
			{TenantID: "tenant-a", DocumentID: "public", Title: "Public", Content: "alpha"},
			{TenantID: "tenant-a", DocumentID: "private", Title: "Private", Content: "secret"},
		}, Queries: []QueryCase{{
			ID: "q1", Category: "acl", TenantID: "tenant-a", Query: "alpha",
			GoldChunkIDs: []string{"public:v000001:c0000"}, ForbiddenChunkIDs: []string{"private:v000001:c0000"},
		}},
	}
	searcher := &fakeSearcher{results: map[string]model.SearchResult{
		"alpha": {Hits: []model.SearchHit{
			{ChunkID: "public:v000001:c0000"},
			// A buggy backend returned more than TopK; the security gate must
			// still inspect this tail result.
			{ChunkID: "private:v000001:c0000"},
		}},
	}, errors: map[string]error{}}
	manifest, err := NewRunner(&fakeIngestor{}, searcher).Run(context.Background(), LoadedDataset{Dataset: dataset, SHA256: "x"}, Options{
		TopK: 1, RetrieverName: "fake", Command: "raghub-eval",
	})
	if err == nil || !strings.Contains(err.Error(), "safety gate failed") {
		t.Fatalf("Run() error = %v, want forbidden-hit failure", err)
	}
	if manifest.Status != SmokeStatus {
		t.Fatalf("status = %q, want complete smoke scope", manifest.Status)
	}
	if manifest.Summary.Gates.Pass || !manifest.Summary.Gates.SearchCompleted || manifest.Summary.Gates.NoForbiddenHits {
		t.Fatalf("unexpected gate result: %+v", manifest.Summary.Gates)
	}
	if len(manifest.PerQuery[0].Hits) != 1 || len(manifest.PerQuery[0].ForbiddenHits) != 1 {
		t.Fatalf("TopK evidence/security tail handling is wrong: %+v", manifest.PerQuery[0])
	}
}

func TestRunnerRejectsWrongIngestIdentity(t *testing.T) {
	t.Parallel()

	dataset := Dataset{
		Name: "test", Version: "1",
		Documents: []DatasetDocument{{TenantID: "tenant-a", DocumentID: "doc-a", Title: "A", Content: "alpha"}},
		Queries:   []QueryCase{{ID: "q1", Category: "exact", TenantID: "tenant-a", Query: "alpha", GoldChunkIDs: []string{"doc-a:v000001:c0000"}}},
	}
	manifest, err := NewRunner(&fakeIngestor{resultTenantID: "wrong-tenant"}, &fakeSearcher{}).Run(
		context.Background(), LoadedDataset{Dataset: dataset, SHA256: "x"}, Options{TopK: 1, RetrieverName: "fake", Command: "raghub-eval"},
	)
	if err == nil || !strings.Contains(err.Error(), "returned identity") {
		t.Fatalf("Run() error = %v, want identity failure", err)
	}
	if manifest.Status != IncompleteStatus {
		t.Fatalf("status = %q, want incomplete", manifest.Status)
	}
}

func TestNormalizeAndHashConfigIsCanonicalAndDeepCopied(t *testing.T) {
	t.Parallel()

	nested := map[string]any{"z": 2, "a": map[string]any{"y": true, "x": "one"}}
	first, firstHash, err := normalizeAndHashConfig(nested)
	if err != nil {
		t.Fatalf("normalizeAndHashConfig() error = %v", err)
	}
	_, secondHash, err := normalizeAndHashConfig(map[string]any{"a": map[string]any{"x": "one", "y": true}, "z": 2})
	if err != nil {
		t.Fatalf("normalizeAndHashConfig() reordered error = %v", err)
	}
	if firstHash != secondHash {
		t.Fatalf("canonical hashes differ by insertion order: %s != %s", firstHash, secondHash)
	}
	nested["z"] = 3
	_, persistedHash, err := normalizeAndHashConfig(first)
	if err != nil {
		t.Fatalf("rehash normalized config: %v", err)
	}
	if persistedHash != firstHash {
		t.Fatalf("normalized manifest config changed through caller alias: %s != %s", persistedHash, firstHash)
	}
	_, changedHash, err := normalizeAndHashConfig(nested)
	if err != nil {
		t.Fatalf("normalize changed config: %v", err)
	}
	if changedHash == firstHash {
		t.Fatal("config value change must change config_sha256")
	}
}
