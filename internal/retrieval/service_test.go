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
}

func TestServiceRejectsInvalidRequest(t *testing.T) {
	tests := []model.SearchRequest{
		{TenantID: "", Query: "deployment"},
		{TenantID: "bad tenant", Query: "deployment"},
		{TenantID: "tenant-a", PrincipalID: "bad principal", Query: "deployment"},
		{TenantID: "tenant-a", Query: ""},
		{TenantID: "tenant-a", Query: "deployment", TopK: 51},
	}
	for _, request := range tests {
		_, err := NewService(&captureRetriever{}).Search(context.Background(), request)
		if !errors.Is(err, ErrInvalidInput) {
			t.Errorf("request %+v error = %v, want ErrInvalidInput", request, err)
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
