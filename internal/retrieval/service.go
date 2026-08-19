package retrieval

import (
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"

	"raghub/internal/model"
)

const (
	defaultTopK = 5
	maxTopK     = 50
)

var (
	identifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$`)
	ErrInvalidInput   = errors.New("invalid search input")
)

// Retriever performs an authorization-scoped retrieval operation.
type Retriever interface {
	Search(ctx context.Context, request model.SearchRequest) (model.SearchResult, error)
}

type DenseRetriever interface {
	SearchDense(ctx context.Context, request model.SearchRequest, profile model.EmbeddingProfile, queryVector []float32) (model.SearchResult, error)
}

type Embedder interface {
	Profile() model.EmbeddingProfile
	Embed(ctx context.Context, inputs []string) ([][]float32, error)
}

type Service struct {
	retriever      Retriever
	denseRetriever DenseRetriever
	embedder       Embedder
}

func NewService(retriever Retriever) *Service {
	return &Service{retriever: retriever}
}

func NewServiceWithDense(retriever Retriever, denseRetriever DenseRetriever, embedder Embedder) *Service {
	return &Service{retriever: retriever, denseRetriever: denseRetriever, embedder: embedder}
}

func (s *Service) Search(ctx context.Context, request model.SearchRequest) (model.SearchResult, error) {
	request.TenantID = strings.TrimSpace(request.TenantID)
	request.PrincipalID = strings.TrimSpace(request.PrincipalID)
	request.Query = strings.TrimSpace(request.Query)

	if !identifierPattern.MatchString(request.TenantID) {
		return model.SearchResult{}, fmt.Errorf("%w: tenant ID must be 1-128 safe identifier characters", ErrInvalidInput)
	}
	if request.PrincipalID != "" && !identifierPattern.MatchString(request.PrincipalID) {
		return model.SearchResult{}, fmt.Errorf("%w: principal ID must be 1-128 safe identifier characters", ErrInvalidInput)
	}
	if request.Query == "" {
		return model.SearchResult{}, fmt.Errorf("%w: query is required", ErrInvalidInput)
	}
	if len(request.Query) > 4096 {
		return model.SearchResult{}, fmt.Errorf("%w: query exceeds 4096 bytes", ErrInvalidInput)
	}
	if request.TopK == 0 {
		request.TopK = defaultTopK
	}
	if request.TopK < 1 || request.TopK > maxTopK {
		return model.SearchResult{}, fmt.Errorf("%w: top_k must be between 1 and %d", ErrInvalidInput, maxTopK)
	}
	if request.Mode == "" {
		request.Mode = model.SearchModeFTS
	}
	switch request.Mode {
	case model.SearchModeFTS:
		if s == nil || s.retriever == nil {
			return model.SearchResult{}, errors.New("FTS retriever is not configured")
		}
		return s.retriever.Search(ctx, request)
	case model.SearchModeDense:
		return s.searchDense(ctx, request)
	default:
		return model.SearchResult{}, fmt.Errorf("%w: mode must be %q or %q", ErrInvalidInput, model.SearchModeFTS, model.SearchModeDense)
	}

}

func (s *Service) searchDense(ctx context.Context, request model.SearchRequest) (model.SearchResult, error) {
	if s == nil || s.denseRetriever == nil || s.embedder == nil {
		return model.SearchResult{}, errors.New("dense retrieval is not configured")
	}
	profile := s.embedder.Profile()
	if strings.TrimSpace(profile.ProfileID) == "" || strings.TrimSpace(profile.Model) == "" || profile.Dimensions <= 0 {
		return model.SearchResult{}, errors.New("dense retrieval embedding profile is invalid")
	}
	startedAt := time.Now()
	vectors, err := s.embedder.Embed(ctx, []string{request.Query})
	if err != nil {
		return model.SearchResult{}, fmt.Errorf("embed dense query with profile %q: %w", profile.ProfileID, err)
	}
	if len(vectors) != 1 || len(vectors[0]) != profile.Dimensions {
		return model.SearchResult{}, fmt.Errorf("embed dense query with profile %q: received shape %dx%d, want 1x%d", profile.ProfileID, len(vectors), firstVectorLength(vectors), profile.Dimensions)
	}
	var normSquared float64
	for dimension, value := range vectors[0] {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return model.SearchResult{}, fmt.Errorf("embed dense query with profile %q: dimension %d is not finite", profile.ProfileID, dimension)
		}
		normSquared += float64(value) * float64(value)
	}
	if normSquared == 0 {
		return model.SearchResult{}, fmt.Errorf("embed dense query with profile %q: cosine vector must not be zero", profile.ProfileID)
	}
	embeddingDuration := float64(time.Since(startedAt).Microseconds()) / 1000
	result, err := s.denseRetriever.SearchDense(ctx, request, profile, vectors[0])
	if err != nil {
		return model.SearchResult{}, err
	}
	result.Traces = append([]model.StageTrace{{Stage: "query_embedding", DurationMS: embeddingDuration}}, result.Traces...)
	return result, nil
}

func firstVectorLength(vectors [][]float32) int {
	if len(vectors) == 0 {
		return 0
	}
	return len(vectors[0])
}
