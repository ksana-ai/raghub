package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"raghub/internal/ingest"
	"raghub/internal/model"
	"raghub/internal/retrieval"
)

const maxRequestBodyBytes = 6 << 20

type Ingestor interface {
	Ingest(ctx context.Context, document model.DocumentInput) (model.IngestResult, error)
}

type Searcher interface {
	Search(ctx context.Context, request model.SearchRequest) (model.SearchResult, error)
}

type ReadinessChecker interface {
	Ping(ctx context.Context) error
}

type API struct {
	ingestor  Ingestor
	searcher  Searcher
	readiness ReadinessChecker
	logger    *slog.Logger
}

func New(ingestor Ingestor, searcher Searcher, readiness ReadinessChecker, logger *slog.Logger) http.Handler {
	if logger == nil {
		logger = slog.Default()
	}
	api := &API{
		ingestor:  ingestor,
		searcher:  searcher,
		readiness: readiness,
		logger:    logger,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", api.health)
	mux.HandleFunc("GET /readyz", api.ready)
	mux.HandleFunc("POST /v1/documents", api.ingestDocument)
	mux.HandleFunc("POST /v1/search", api.search)
	return mux
}

type ingestRequest struct {
	DocumentID        string          `json:"document_id"`
	Title             string          `json:"title"`
	SourceURI         string          `json:"source_uri"`
	Content           string          `json:"content"`
	AllowedPrincipals []string        `json:"allowed_principals"`
	Metadata          json.RawMessage `json:"metadata"`
}

type searchRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

func (a *API) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (a *API) ready(w http.ResponseWriter, r *http.Request) {
	if a.readiness == nil {
		writeProblem(w, http.StatusServiceUnavailable, "not_ready", "readiness check is not configured")
		return
	}
	if err := a.readiness.Ping(r.Context()); err != nil {
		a.logger.ErrorContext(r.Context(), "readiness check failed", "error", err)
		writeProblem(w, http.StatusServiceUnavailable, "not_ready", "database is unavailable")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (a *API) ingestDocument(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requiredHeader(w, r, "X-Tenant-ID")
	if !ok {
		return
	}

	var request ingestRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeProblem(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	result, err := a.ingestor.Ingest(r.Context(), model.DocumentInput{
		TenantID:          tenantID,
		ID:                request.DocumentID,
		Title:             request.Title,
		SourceURI:         request.SourceURI,
		Content:           request.Content,
		AllowedPrincipals: request.AllowedPrincipals,
		Metadata:          request.Metadata,
	})
	if err != nil {
		if errors.Is(err, ingest.ErrInvalidInput) {
			writeProblem(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}
		a.logger.ErrorContext(r.Context(), "document ingestion failed", "tenant_id", tenantID, "document_id", request.DocumentID, "error", err)
		writeProblem(w, http.StatusInternalServerError, "internal_error", "document ingestion failed")
		return
	}

	status := http.StatusCreated
	if result.Unchanged {
		status = http.StatusOK
	}
	writeJSON(w, status, result)
}

func (a *API) search(w http.ResponseWriter, r *http.Request) {
	tenantID, ok := requiredHeader(w, r, "X-Tenant-ID")
	if !ok {
		return
	}

	var request searchRequest
	if err := decodeJSON(w, r, &request); err != nil {
		writeProblem(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	result, err := a.searcher.Search(r.Context(), model.SearchRequest{
		TenantID:    tenantID,
		PrincipalID: strings.TrimSpace(r.Header.Get("X-Principal-ID")),
		Query:       request.Query,
		TopK:        request.TopK,
	})
	if err != nil {
		if errors.Is(err, retrieval.ErrInvalidInput) {
			writeProblem(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}
		a.logger.ErrorContext(r.Context(), "search failed", "tenant_id", tenantID, "error", err)
		writeProblem(w, http.StatusInternalServerError, "internal_error", "search failed")
		return
	}

	writeJSON(w, http.StatusOK, result)
}

func requiredHeader(w http.ResponseWriter, r *http.Request, name string) (string, bool) {
	value := strings.TrimSpace(r.Header.Get(name))
	if value == "" {
		writeProblem(w, http.StatusBadRequest, "invalid_request", fmt.Sprintf("%s header is required", name))
		return "", false
	}
	return value, true
}

func decodeJSON(w http.ResponseWriter, r *http.Request, destination any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("decode JSON body: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("JSON body must contain exactly one object")
		}
		return fmt.Errorf("decode trailing JSON data: %w", err)
	}
	return nil
}

func writeProblem(w http.ResponseWriter, status int, code, message string) {
	writeJSONStatus(w, status, map[string]any{
		"error": map[string]string{
			"code":    code,
			"message": message,
		},
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	writeJSONStatus(w, status, value)
}

func writeJSONStatus(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
