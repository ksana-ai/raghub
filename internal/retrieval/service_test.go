package retrieval

import (
	"context"
	"errors"
	"testing"

	"raghub/internal/model"
)

type captureRetriever struct {
	request model.SearchRequest
	result  model.SearchResult
	err     error
}

type captureDenseRetriever struct {
	request model.SearchRequest
	profile model.EmbeddingProfile
	vector  []float32
	result  model.SearchResult
	err     error
}

func (r *captureDenseRetriever) SearchDense(_ context.Context, request model.SearchRequest, profile model.EmbeddingProfile, vector []float32) (model.SearchResult, error) {
	r.request = request
	r.profile = profile
	r.vector = append([]float32(nil), vector...)
	return r.result, r.err
}

type staticEmbedder struct {
	profile model.EmbeddingProfile
	vectors [][]float32
	err     error
	inputs  []string
}

func (e *staticEmbedder) Profile() model.EmbeddingProfile { return e.profile }
func (e *staticEmbedder) Embed(_ context.Context, inputs []string) ([][]float32, error) {
	e.inputs = append([]string(nil), inputs...)
	return e.vectors, e.err
}

func (r *captureRetriever) Search(_ context.Context, request model.SearchRequest) (model.SearchResult, error) {
	r.request = request
	return r.result, r.err
}

func TestServiceDefaultsTopKAndTrimsScope(t *testing.T) {
	retriever := &captureRetriever{}
	service := NewService(retriever)

	_, err := service.Search(context.Background(), model.SearchRequest{
		TenantID:    " tenant-a ",
		PrincipalID: " user:alice ",
		Query:       " deployment ",
	})
	if err != nil {
		t.Fatal(err)
	}
	if retriever.request.TenantID != "tenant-a" || retriever.request.PrincipalID != "user:alice" {
		t.Fatalf("scope was not normalized: %+v", retriever.request)
	}
	if retriever.request.Query != "deployment" || retriever.request.TopK != 5 {
		t.Fatalf("query defaults were not applied: %+v", retriever.request)
	}
	if retriever.request.Mode != model.SearchModeFTS {
		t.Fatalf("default mode = %q, want fts", retriever.request.Mode)
	}
}

func TestServiceRejectsInvalidRequest(t *testing.T) {
	tests := []model.SearchRequest{
		{TenantID: "", Query: "deployment"},
		{TenantID: "bad tenant", Query: "deployment"},
		{TenantID: "tenant-a", PrincipalID: "bad principal", Query: "deployment"},
		{TenantID: "tenant-a", Query: ""},
		{TenantID: "tenant-a", Query: "deployment", TopK: 51},
		{TenantID: "tenant-a", Query: "deployment", Mode: "unknown"},
	}
	for _, request := range tests {
		_, err := NewService(&captureRetriever{}).Search(context.Background(), request)
		if !errors.Is(err, ErrInvalidInput) {
			t.Errorf("request %+v error = %v, want ErrInvalidInput", request, err)
		}
	}
}

func TestServiceRunsDenseQueryEmbeddingAndPreservesTraces(t *testing.T) {
	profile := model.EmbeddingProfile{
		ProfileID: "dense-v1", Provider: "test", Model: "test-model", Dimensions: 2,
		DocumentRecipe: "indexed_text/v1", QueryRecipe: "raw_query/v1",
	}
	embedder := &staticEmbedder{profile: profile, vectors: [][]float32{{0.6, 0.8}}}
	dense := &captureDenseRetriever{result: model.SearchResult{
		Hits:   []model.SearchHit{{ChunkID: "guide:v000001:c0000"}},
		Traces: []model.StageTrace{{Stage: "dense", DurationMS: 2}},
	}}
	service := NewServiceWithDense(&captureRetriever{}, dense, embedder)
	result, err := service.Search(context.Background(), model.SearchRequest{
		TenantID: "tenant-a", Query: "semantic question", Mode: model.SearchModeDense,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(embedder.inputs) != 1 || embedder.inputs[0] != "semantic question" {
		t.Fatalf("embedding inputs = %#v", embedder.inputs)
	}
	if dense.profile != profile || len(dense.vector) != 2 || dense.request.Mode != model.SearchModeDense {
		t.Fatalf("dense call profile=%+v vector=%v request=%+v", dense.profile, dense.vector, dense.request)
	}
	if len(result.Traces) != 2 || result.Traces[0].Stage != "query_embedding" || result.Traces[1].Stage != "dense" {
		t.Fatalf("traces = %+v", result.Traces)
	}
}

func TestServiceRejectsInvalidDenseEmbedding(t *testing.T) {
	profile := model.EmbeddingProfile{ProfileID: "dense-v1", Model: "m", Dimensions: 2}
	for _, vectors := range [][][]float32{nil, {{1}}, {{0, 0}}} {
		service := NewServiceWithDense(&captureRetriever{}, &captureDenseRetriever{}, &staticEmbedder{profile: profile, vectors: vectors})
		_, err := service.Search(context.Background(), model.SearchRequest{
			TenantID: "tenant-a", Query: "query", Mode: model.SearchModeDense,
		})
		if err == nil {
			t.Fatalf("vectors %v unexpectedly succeeded", vectors)
		}
	}
}

func TestServiceDoesNotTurnBackendFailureIntoEmptyHits(t *testing.T) {
	backendError := errors.New("database unavailable")
	_, err := NewService(&captureRetriever{err: backendError}).Search(context.Background(), model.SearchRequest{
		TenantID: "tenant-a",
		Query:    "deployment",
	})
	if !errors.Is(err, backendError) {
		t.Fatalf("error = %v, want backend error", err)
	}
}
