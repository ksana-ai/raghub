package retrieval

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"

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

type Service struct {
	retriever Retriever
}

func NewService(retriever Retriever) *Service {
	return &Service{retriever: retriever}
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

	return s.retriever.Search(ctx, request)
}
