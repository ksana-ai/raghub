package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"raghub/internal/model"
)

type fakeIngestor struct {
	received model.DocumentInput
	result   model.IngestResult
	err      error
}

func (f *fakeIngestor) Ingest(_ context.Context, document model.DocumentInput) (model.IngestResult, error) {
	f.received = document
	return f.result, f.err
}

type fakeSearcher struct {
	received model.SearchRequest
	result   model.SearchResult
	err      error
}

func (f *fakeSearcher) Search(_ context.Context, request model.SearchRequest) (model.SearchResult, error) {
	f.received = request
	return f.result, f.err
}

type fakeReadiness struct{ err error }

func (f fakeReadiness) Ping(context.Context) error { return f.err }

func TestIngestRequiresTenantHeader(t *testing.T) {
	handler := newTestHandler(&fakeIngestor{}, &fakeSearcher{}, fakeReadiness{})
	request := httptest.NewRequest(http.MethodPost, "/v1/documents", strings.NewReader(`{"title":"Guide","content":"hello"}`))
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusBadRequest)
	}
}

func TestIngestPassesTenantOutsidePayload(t *testing.T) {
	ingestor := &fakeIngestor{result: model.IngestResult{
		TenantID:   "tenant-a",
		DocumentID: "guide",
		Version:    1,
		CreatedAt:  time.Unix(1, 0).UTC(),
	}}
	handler := newTestHandler(ingestor, &fakeSearcher{}, fakeReadiness{})
	request := httptest.NewRequest(http.MethodPost, "/v1/documents", strings.NewReader(`{"document_id":"guide","title":"Guide","content":"hello"}`))
	request.Header.Set("X-Tenant-ID", "tenant-a")
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusCreated {
		t.Fatalf("status = %d, want %d; body=%s", response.Code, http.StatusCreated, response.Body.String())
	}
	if ingestor.received.TenantID != "tenant-a" {
		t.Fatalf("tenant = %q, want tenant-a", ingestor.received.TenantID)
	}
}

func TestSearchPassesHybridModePrincipalAndReturnsTrace(t *testing.T) {
	searcher := &fakeSearcher{result: model.SearchResult{
		Hits:   []model.SearchHit{{ChunkID: "guide:v000001:c0000", Score: 0.8}},
		Traces: []model.StageTrace{{Stage: "fts", DurationMS: 1.2}},
	}}
	handler := newTestHandler(&fakeIngestor{}, searcher, fakeReadiness{})
	request := httptest.NewRequest(http.MethodPost, "/v1/search", strings.NewReader(`{"query":"deployment","top_k":3,"mode":"hybrid"}`))
	request.Header.Set("X-Tenant-ID", "tenant-a")
	request.Header.Set("X-Principal-ID", "user:alice")
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d; body=%s", response.Code, http.StatusOK, response.Body.String())
	}
	if searcher.received.PrincipalID != "user:alice" || searcher.received.TopK != 3 {
		t.Fatalf("unexpected search request: %+v", searcher.received)
	}
	if searcher.received.Mode != model.SearchModeHybrid {
		t.Fatalf("search mode = %q, want hybrid", searcher.received.Mode)
	}
	var body model.SearchResult
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(body.Traces) != 1 || body.Traces[0].Stage != "fts" {
		t.Fatalf("unexpected traces: %+v", body.Traces)
	}
}

func TestReadyReturnsUnavailable(t *testing.T) {
	handler := newTestHandler(&fakeIngestor{}, &fakeSearcher{}, fakeReadiness{err: errors.New("down")})
	request := httptest.NewRequest(http.MethodGet, "/readyz", nil)
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusServiceUnavailable)
	}
}

func newTestHandler(ingestor Ingestor, searcher Searcher, readiness ReadinessChecker) http.Handler {
	return New(ingestor, searcher, readiness, slog.New(slog.NewTextHandler(io.Discard, nil)))
}
