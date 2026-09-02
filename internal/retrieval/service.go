package retrieval

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/ksana-ai/raghub/internal/model"
)

const (
	defaultTopK                      = 5
	maxTopK                          = 50
	maxHybridRRFK                    = 1_000_000
	DefaultHybridRRFK                = 60
	DefaultHybridFTSCandidateDepth   = 20
	DefaultHybridDenseCandidateDepth = 20
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

// HybridConfig fixes the independently retrieved candidate sets and the RRF
// smoothing constant. Candidate depths are bounded by the same limit as a
// direct retrieval request so hybrid cannot bypass the public query budget.
type HybridConfig struct {
	FTSCandidateDepth   int
	DenseCandidateDepth int
	RRFK                int
}

func DefaultHybridConfig() HybridConfig {
	return HybridConfig{
		FTSCandidateDepth:   DefaultHybridFTSCandidateDepth,
		DenseCandidateDepth: DefaultHybridDenseCandidateDepth,
		RRFK:                DefaultHybridRRFK,
	}
}

type Service struct {
	retriever      Retriever
	denseRetriever DenseRetriever
	embedder       Embedder
	hybridConfig   HybridConfig
}

func NewService(retriever Retriever) *Service {
	return &Service{retriever: retriever}
}

func NewServiceWithDense(retriever Retriever, denseRetriever DenseRetriever, embedder Embedder) *Service {
	return &Service{
		retriever:      retriever,
		denseRetriever: denseRetriever,
		embedder:       embedder,
		hybridConfig:   DefaultHybridConfig(),
	}
}

// NewServiceWithHybrid configures all three independently measurable search
// modes. Zero values are rejected rather than silently changing an evaluation
// protocol; callers that want the preregistered baseline should use
// DefaultHybridConfig.
func NewServiceWithHybrid(retriever Retriever, denseRetriever DenseRetriever, embedder Embedder, config HybridConfig) (*Service, error) {
	if err := validateHybridConfig(config); err != nil {
		return nil, err
	}
	return &Service{
		retriever:      retriever,
		denseRetriever: denseRetriever,
		embedder:       embedder,
		hybridConfig:   config,
	}, nil
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
		result, err := s.retriever.Search(ctx, request)
		if err != nil {
			return model.SearchResult{}, err
		}
		result.CandidateSets = []model.CandidateSet{candidateSet("fts", result.Hits)}
		return result, nil
	case model.SearchModeDense:
		return s.searchDense(ctx, request)
	case model.SearchModeHybrid:
		return s.searchHybrid(ctx, request)
	default:
		return model.SearchResult{}, fmt.Errorf(
			"%w: mode must be %q, %q, or %q",
			ErrInvalidInput,
			model.SearchModeFTS,
			model.SearchModeDense,
			model.SearchModeHybrid,
		)
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
	result.CandidateSets = []model.CandidateSet{candidateSet("dense", result.Hits)}
	return result, nil
}

type hybridBranchResult struct {
	stage  string
	result model.SearchResult
	err    error
}

func (s *Service) searchHybrid(ctx context.Context, request model.SearchRequest) (model.SearchResult, error) {
	if s == nil || s.retriever == nil {
		return model.SearchResult{}, errors.New("hybrid FTS retrieval is not configured")
	}
	if s.denseRetriever == nil || s.embedder == nil {
		return model.SearchResult{}, errors.New("hybrid dense retrieval is not configured")
	}
	if err := validateHybridConfig(s.hybridConfig); err != nil {
		return model.SearchResult{}, err
	}

	ftsRequest := request
	ftsRequest.Mode = model.SearchModeFTS
	ftsRequest.TopK = max(request.TopK, s.hybridConfig.FTSCandidateDepth)
	denseRequest := request
	denseRequest.Mode = model.SearchModeDense
	denseRequest.TopK = max(request.TopK, s.hybridConfig.DenseCandidateDepth)

	branchCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	results := make(chan hybridBranchResult, 2)
	go func() {
		result, err := s.retriever.Search(branchCtx, ftsRequest)
		results <- hybridBranchResult{stage: "fts", result: result, err: err}
	}()
	go func() {
		result, err := s.searchDense(branchCtx, denseRequest)
		results <- hybridBranchResult{stage: "dense", result: result, err: err}
	}()

	var ftsBranch, denseBranch hybridBranchResult
	failureSeen := false
	for range 2 {
		branch := <-results
		if branch.err != nil && !failureSeen {
			failureSeen = true
			cancel()
		}
		switch branch.stage {
		case "fts":
			ftsBranch = branch
		case "dense":
			denseBranch = branch
		}
	}
	if stage, err := deterministicHybridBranchError(ftsBranch.err, denseBranch.err); err != nil {
		return model.SearchResult{}, fmt.Errorf("hybrid %s branch: %w", stage, err)
	}

	fusionStartedAt := time.Now()
	hits, err := reciprocalRankFusion(
		ftsBranch.result.Hits,
		denseBranch.result.Hits,
		request.TopK,
		s.hybridConfig.RRFK,
	)
	if err != nil {
		return model.SearchResult{}, fmt.Errorf("hybrid fusion: %w", err)
	}
	traces := make([]model.StageTrace, 0, len(ftsBranch.result.Traces)+len(denseBranch.result.Traces)+1)
	traces = append(traces, ftsBranch.result.Traces...)
	traces = append(traces, denseBranch.result.Traces...)
	traces = append(traces, model.StageTrace{
		Stage:      "rrf_fusion",
		DurationMS: float64(time.Since(fusionStartedAt).Microseconds()) / 1000,
	})
	return model.SearchResult{
		Hits:   hits,
		Traces: traces,
		CandidateSets: []model.CandidateSet{
			candidateSet("fts", ftsBranch.result.Hits),
			candidateSet("dense", denseBranch.result.Hits),
		},
	}, nil
}

func candidateSet(stage string, hits []model.SearchHit) model.CandidateSet {
	result := model.CandidateSet{Stage: stage, Hits: make([]model.CandidateHit, 0, len(hits))}
	for index, hit := range hits {
		result.Hits = append(result.Hits, model.CandidateHit{ChunkID: hit.ChunkID, Rank: index + 1})
	}
	return result
}

func deterministicHybridBranchError(ftsErr, denseErr error) (string, error) {
	switch {
	case ftsErr == nil && denseErr == nil:
		return "", nil
	case ftsErr == nil:
		return "dense", denseErr
	case denseErr == nil:
		return "fts", ftsErr
	}

	// Prefer the root failure over the sibling's cancellation. When both
	// branches fail independently, prefer FTS so the same inputs produce the
	// same error text regardless of goroutine scheduling.
	ftsCanceled := isContextTermination(ftsErr)
	denseCanceled := isContextTermination(denseErr)
	switch {
	case ftsCanceled && !denseCanceled:
		return "dense", denseErr
	case !ftsCanceled && denseCanceled:
		return "fts", ftsErr
	default:
		return "fts", ftsErr
	}
}

func isContextTermination(err error) bool {
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

type fusionCandidate struct {
	hit        model.SearchHit
	ftsScore   *model.StageScore
	denseScore *model.StageScore
	rrfScore   float64
}

func reciprocalRankFusion(ftsHits, denseHits []model.SearchHit, topK, rrfK int) ([]model.SearchHit, error) {
	candidates := make(map[string]*fusionCandidate, len(ftsHits)+len(denseHits))
	if err := addFusionCandidates(candidates, ftsHits, "fts", rrfK); err != nil {
		return nil, err
	}
	if err := addFusionCandidates(candidates, denseHits, "dense", rrfK); err != nil {
		return nil, err
	}

	ordered := make([]*fusionCandidate, 0, len(candidates))
	for _, candidate := range candidates {
		ordered = append(ordered, candidate)
	}
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].rrfScore != ordered[j].rrfScore {
			return ordered[i].rrfScore > ordered[j].rrfScore
		}
		return ordered[i].hit.ChunkID < ordered[j].hit.ChunkID
	})
	if len(ordered) > topK {
		ordered = ordered[:topK]
	}

	hits := make([]model.SearchHit, 0, len(ordered))
	for index, candidate := range ordered {
		hit := candidate.hit
		hit.Score = candidate.rrfScore
		hit.StageScores = make([]model.StageScore, 0, 3)
		if candidate.ftsScore != nil {
			hit.StageScores = append(hit.StageScores, *candidate.ftsScore)
		}
		if candidate.denseScore != nil {
			hit.StageScores = append(hit.StageScores, *candidate.denseScore)
		}
		hit.StageScores = append(hit.StageScores, model.StageScore{
			Stage: "rrf", Rank: index + 1, Score: candidate.rrfScore,
		})
		hits = append(hits, hit)
	}
	return hits, nil
}

func addFusionCandidates(candidates map[string]*fusionCandidate, hits []model.SearchHit, stage string, rrfK int) error {
	seen := make(map[string]struct{}, len(hits))
	for index, hit := range hits {
		if strings.TrimSpace(hit.ChunkID) == "" {
			return fmt.Errorf("%s candidate at rank %d has an empty chunk ID", stage, index+1)
		}
		if _, exists := seen[hit.ChunkID]; exists {
			return fmt.Errorf("%s branch returned duplicate chunk %q", stage, hit.ChunkID)
		}
		seen[hit.ChunkID] = struct{}{}

		stageScore, err := candidateStageScore(hit, stage, index+1)
		if err != nil {
			return err
		}
		candidate, exists := candidates[hit.ChunkID]
		if !exists {
			candidate = &fusionCandidate{hit: hit}
			candidates[hit.ChunkID] = candidate
		} else if mismatch := hybridHitMismatch(candidate.hit, hit); mismatch != "" {
			return fmt.Errorf("chunk %q differs between FTS and dense candidates: %s", hit.ChunkID, mismatch)
		}
		switch stage {
		case "fts":
			candidate.ftsScore = &stageScore
		case "dense":
			candidate.denseScore = &stageScore
		default:
			return fmt.Errorf("unsupported fusion stage %q", stage)
		}
		candidate.rrfScore += 1 / float64(rrfK+stageScore.Rank)
	}
	return nil
}

func candidateStageScore(hit model.SearchHit, stage string, rank int) (model.StageScore, error) {
	if len(hit.StageScores) != 1 || hit.StageScores[0].Stage != stage {
		return model.StageScore{}, fmt.Errorf(
			"%s candidate %q must contain exactly one %s stage score",
			stage,
			hit.ChunkID,
			stage,
		)
	}
	found := hit.StageScores[0]
	if found.Rank != rank {
		return model.StageScore{}, fmt.Errorf(
			"%s candidate %q reports rank %d at position %d",
			stage,
			hit.ChunkID,
			found.Rank,
			rank,
		)
	}
	if math.IsNaN(found.Score) || math.IsInf(found.Score, 0) ||
		math.IsNaN(hit.Score) || math.IsInf(hit.Score, 0) {
		return model.StageScore{}, fmt.Errorf("%s candidate %q has a non-finite score", stage, hit.ChunkID)
	}
	if found.Score != hit.Score {
		return model.StageScore{}, fmt.Errorf(
			"%s candidate %q stage score does not match its final branch score",
			stage,
			hit.ChunkID,
		)
	}
	return found, nil
}

func hybridHitMismatch(left, right model.SearchHit) string {
	switch {
	case left.DocumentID != right.DocumentID:
		return "document_id"
	case left.DocumentVersion != right.DocumentVersion:
		return "document_version"
	case left.Title != right.Title:
		return "title"
	case left.SourceURI != right.SourceURI:
		return "source_uri"
	case !slices.Equal(left.HeadingPath, right.HeadingPath):
		return "heading_path"
	case left.Content != right.Content:
		return "content"
	case !bytes.Equal(left.Metadata, right.Metadata):
		return "metadata"
	default:
		return ""
	}
}

func validateHybridConfig(config HybridConfig) error {
	if config.FTSCandidateDepth < 1 || config.FTSCandidateDepth > maxTopK {
		return fmt.Errorf("invalid hybrid config: FTS candidate depth must be between 1 and %d", maxTopK)
	}
	if config.DenseCandidateDepth < 1 || config.DenseCandidateDepth > maxTopK {
		return fmt.Errorf("invalid hybrid config: dense candidate depth must be between 1 and %d", maxTopK)
	}
	if config.RRFK < 1 || config.RRFK > maxHybridRRFK {
		return fmt.Errorf("invalid hybrid config: RRF k must be between 1 and %d", maxHybridRRFK)
	}
	return nil
}

func firstVectorLength(vectors [][]float32) int {
	if len(vectors) == 0 {
		return 0
	}
	return len(vectors[0])
}
