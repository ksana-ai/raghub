package retrieval

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

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

type coordinatedRetriever struct {
	started     chan struct{}
	peerStarted <-chan struct{}
	request     model.SearchRequest
	result      model.SearchResult
}

func (r *coordinatedRetriever) Search(ctx context.Context, request model.SearchRequest) (model.SearchResult, error) {
	r.request = request
	close(r.started)
	select {
	case <-r.peerStarted:
		return r.result, nil
	case <-ctx.Done():
		return model.SearchResult{}, ctx.Err()
	}
}

type coordinatedDenseRetriever struct {
	started     chan struct{}
	peerStarted <-chan struct{}
	request     model.SearchRequest
	result      model.SearchResult
}

func (r *coordinatedDenseRetriever) SearchDense(
	ctx context.Context,
	request model.SearchRequest,
	_ model.EmbeddingProfile,
	_ []float32,
) (model.SearchResult, error) {
	r.request = request
	close(r.started)
	select {
	case <-r.peerStarted:
		return r.result, nil
	case <-ctx.Done():
		return model.SearchResult{}, ctx.Err()
	}
}

type cancelWaitingDenseRetriever struct {
	done chan struct{}
}

func (r *cancelWaitingDenseRetriever) SearchDense(
	ctx context.Context,
	_ model.SearchRequest,
	_ model.EmbeddingProfile,
	_ []float32,
) (model.SearchResult, error) {
	<-ctx.Done()
	close(r.done)
	return model.SearchResult{}, ctx.Err()
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
	if len(result.CandidateSets) != 1 || result.CandidateSets[0].Stage != "dense" ||
		len(result.CandidateSets[0].Hits) != 1 || result.CandidateSets[0].Hits[0].ChunkID != "guide:v000001:c0000" {
		t.Fatalf("dense candidate evidence = %+v", result.CandidateSets)
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

func TestServiceRunsHybridBranchesConcurrentlyAndFusesDeterministically(t *testing.T) {
	profile := model.EmbeddingProfile{ProfileID: "dense-v1", Model: "m", Dimensions: 2}
	ftsStarted := make(chan struct{})
	denseStarted := make(chan struct{})
	fts := &coordinatedRetriever{
		started:     ftsStarted,
		peerStarted: denseStarted,
		result: model.SearchResult{
			Hits: []model.SearchHit{
				hybridHit("chunk-a", 0.9, "fts", 1),
				hybridHit("chunk-b", 0.8, "fts", 2),
				hybridHit("chunk-c", 0.7, "fts", 3),
			},
			Traces: []model.StageTrace{{Stage: "fts", DurationMS: 1}},
		},
	}
	dense := &coordinatedDenseRetriever{
		started:     denseStarted,
		peerStarted: ftsStarted,
		result: model.SearchResult{
			Hits: []model.SearchHit{
				hybridHit("chunk-c", 0.95, "dense", 1),
				hybridHit("chunk-a", 0.85, "dense", 2),
				hybridHit("chunk-d", 0.75, "dense", 3),
			},
			Traces: []model.StageTrace{{Stage: "dense", DurationMS: 2}},
		},
	}
	service, err := NewServiceWithHybrid(
		fts,
		dense,
		&staticEmbedder{profile: profile, vectors: [][]float32{{0.6, 0.8}}},
		DefaultHybridConfig(),
	)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	result, err := service.Search(ctx, model.SearchRequest{
		TenantID: "tenant-a", Query: "hybrid question", TopK: 4, Mode: model.SearchModeHybrid,
	})
	if err != nil {
		t.Fatal(err)
	}
	if fts.request.Mode != model.SearchModeFTS || fts.request.TopK != DefaultHybridFTSCandidateDepth {
		t.Fatalf("FTS branch request = %+v", fts.request)
	}
	if dense.request.Mode != model.SearchModeDense || dense.request.TopK != DefaultHybridDenseCandidateDepth {
		t.Fatalf("dense branch request = %+v", dense.request)
	}
	assertChunkOrder(t, result.Hits, []string{"chunk-a", "chunk-c", "chunk-b", "chunk-d"})
	assertStages(t, result.Hits[0].StageScores, []string{"fts", "dense", "rrf"})
	if result.Hits[0].StageScores[0].Rank != 1 || result.Hits[0].StageScores[0].Score != 0.9 {
		t.Fatalf("FTS evidence was not preserved: %+v", result.Hits[0].StageScores)
	}
	if result.Hits[0].StageScores[1].Rank != 2 || result.Hits[0].StageScores[1].Score != 0.85 {
		t.Fatalf("dense evidence was not preserved: %+v", result.Hits[0].StageScores)
	}
	wantRRF := 1.0/61 + 1.0/62
	if math.Abs(result.Hits[0].Score-wantRRF) > 1e-12 || result.Hits[0].StageScores[2].Score != result.Hits[0].Score {
		t.Fatalf("RRF score = %.16f stages=%+v, want %.16f", result.Hits[0].Score, result.Hits[0].StageScores, wantRRF)
	}
	if result.Hits[0].StageScores[2].Rank != 1 {
		t.Fatalf("final RRF rank = %d, want 1", result.Hits[0].StageScores[2].Rank)
	}
	traceStages := make([]string, 0, len(result.Traces))
	for _, trace := range result.Traces {
		traceStages = append(traceStages, trace.Stage)
	}
	wantTraces := []string{"fts", "query_embedding", "dense", "rrf_fusion"}
	if !equalStrings(traceStages, wantTraces) {
		t.Fatalf("trace stages = %v, want %v", traceStages, wantTraces)
	}
	if len(result.CandidateSets) != 2 || result.CandidateSets[0].Stage != "fts" || result.CandidateSets[1].Stage != "dense" {
		t.Fatalf("hybrid candidate set stages = %+v", result.CandidateSets)
	}
	assertCandidateIDs(t, result.CandidateSets[0], []string{"chunk-a", "chunk-b", "chunk-c"})
	assertCandidateIDs(t, result.CandidateSets[1], []string{"chunk-c", "chunk-a", "chunk-d"})
}

func TestServiceHybridUsesLargerOfTopKAndConfiguredCandidateDepth(t *testing.T) {
	config := HybridConfig{FTSCandidateDepth: 2, DenseCandidateDepth: 3, RRFK: 10}
	fts := &captureRetriever{}
	dense := &captureDenseRetriever{}
	service, err := NewServiceWithHybrid(
		fts,
		dense,
		&staticEmbedder{
			profile: model.EmbeddingProfile{ProfileID: "dense-v1", Model: "m", Dimensions: 2},
			vectors: [][]float32{{1, 0}},
		},
		config,
	)
	if err != nil {
		t.Fatal(err)
	}
	_, err = service.Search(context.Background(), model.SearchRequest{
		TenantID: "tenant-a", Query: "query", TopK: 4, Mode: model.SearchModeHybrid,
	})
	if err != nil {
		t.Fatal(err)
	}
	if fts.request.TopK != 4 || dense.request.TopK != 4 {
		t.Fatalf("branch TopK = %d/%d, want 4/4", fts.request.TopK, dense.request.TopK)
	}
}

func TestServiceHybridFailsAndWaitsWhenEitherBranchFails(t *testing.T) {
	backendError := errors.New("FTS database unavailable")
	dense := &cancelWaitingDenseRetriever{done: make(chan struct{})}
	service := NewServiceWithDense(
		&captureRetriever{err: backendError},
		dense,
		&staticEmbedder{
			profile: model.EmbeddingProfile{ProfileID: "dense-v1", Model: "m", Dimensions: 2},
			vectors: [][]float32{{1, 0}},
		},
	)
	_, err := service.Search(context.Background(), model.SearchRequest{
		TenantID: "tenant-a", Query: "query", Mode: model.SearchModeHybrid,
	})
	if !errors.Is(err, backendError) {
		t.Fatalf("error = %v, want FTS backend error", err)
	}
	select {
	case <-dense.done:
	default:
		t.Fatal("hybrid returned before the canceled dense branch finished")
	}
}

func TestServiceHybridChoosesFTSErrorDeterministicallyWhenBothBranchesFail(t *testing.T) {
	ftsError := errors.New("FTS root failure")
	denseError := errors.New("dense root failure")
	for range 50 {
		service := NewServiceWithDense(
			&captureRetriever{err: ftsError},
			&captureDenseRetriever{err: denseError},
			&staticEmbedder{
				profile: model.EmbeddingProfile{ProfileID: "dense-v1", Model: "m", Dimensions: 2},
				vectors: [][]float32{{1, 0}},
			},
		)
		_, err := service.Search(context.Background(), model.SearchRequest{
			TenantID: "tenant-a", Query: "query", Mode: model.SearchModeHybrid,
		})
		if !errors.Is(err, ftsError) {
			t.Fatalf("error = %v, want deterministic FTS failure", err)
		}
	}
}

func TestReciprocalRankFusionUsesChunkIDTieBreak(t *testing.T) {
	hits, err := reciprocalRankFusion(
		[]model.SearchHit{hybridHit("chunk-z", 0.9, "fts", 1)},
		[]model.SearchHit{hybridHit("chunk-a", 0.9, "dense", 1)},
		2,
		DefaultHybridRRFK,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertChunkOrder(t, hits, []string{"chunk-a", "chunk-z"})
}

func TestReciprocalRankFusionRejectsBackendContractViolations(t *testing.T) {
	tests := []struct {
		name      string
		ftsHits   []model.SearchHit
		denseHits []model.SearchHit
	}{
		{
			name: "duplicate chunk in branch",
			ftsHits: []model.SearchHit{
				hybridHit("chunk-a", 0.9, "fts", 1),
				hybridHit("chunk-a", 0.8, "fts", 2),
			},
		},
		{
			name:    "inconsistent content across branches",
			ftsHits: []model.SearchHit{hybridHit("chunk-a", 0.9, "fts", 1)},
			denseHits: []model.SearchHit{func() model.SearchHit {
				hit := hybridHit("chunk-a", 0.9, "dense", 1)
				hit.Content = "different immutable content"
				return hit
			}()},
		},
		{
			name:    "inconsistent source locator across branches",
			ftsHits: []model.SearchHit{hybridHit("chunk-a", 0.9, "fts", 1)},
			denseHits: []model.SearchHit{func() model.SearchHit {
				hit := hybridHit("chunk-a", 0.9, "dense", 1)
				hit.SourceURI = "https://other.example.test/chunk-a"
				return hit
			}()},
		},
		{
			name: "reported rank differs from position",
			ftsHits: []model.SearchHit{
				hybridHit("chunk-a", 0.9, "fts", 2),
			},
		},
		{
			name: "non-finite source score",
			ftsHits: []model.SearchHit{
				hybridHit("chunk-a", math.NaN(), "fts", 1),
			},
		},
		{
			name: "missing source stage score",
			ftsHits: []model.SearchHit{func() model.SearchHit {
				hit := hybridHit("chunk-a", 0.9, "fts", 1)
				hit.StageScores = nil
				return hit
			}()},
		},
		{
			name: "extra source stage score",
			ftsHits: []model.SearchHit{func() model.SearchHit {
				hit := hybridHit("chunk-a", 0.9, "fts", 1)
				hit.StageScores = append(hit.StageScores, model.StageScore{Stage: "other", Rank: 1, Score: 0.9})
				return hit
			}()},
		},
		{
			name: "wrong source stage",
			ftsHits: []model.SearchHit{
				hybridHit("chunk-a", 0.9, "dense", 1),
			},
		},
		{
			name: "source stage score differs from branch score",
			ftsHits: []model.SearchHit{func() model.SearchHit {
				hit := hybridHit("chunk-a", 0.9, "fts", 1)
				hit.StageScores[0].Score = 0.8
				return hit
			}()},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := reciprocalRankFusion(test.ftsHits, test.denseHits, 5, DefaultHybridRRFK)
			if err == nil {
				t.Fatal("invalid backend candidates unexpectedly fused")
			}
		})
	}
}

func TestNewServiceWithHybridRejectsInvalidConfiguration(t *testing.T) {
	tests := []HybridConfig{
		{},
		{FTSCandidateDepth: 51, DenseCandidateDepth: 20, RRFK: 60},
		{FTSCandidateDepth: 20, DenseCandidateDepth: 51, RRFK: 60},
		{FTSCandidateDepth: 20, DenseCandidateDepth: 20, RRFK: 0},
	}
	for _, config := range tests {
		if _, err := NewServiceWithHybrid(nil, nil, nil, config); err == nil {
			t.Fatalf("config %+v unexpectedly accepted", config)
		}
	}
}

func hybridHit(chunkID string, score float64, stage string, rank int) model.SearchHit {
	return model.SearchHit{
		ChunkID:         chunkID,
		DocumentID:      "document-" + chunkID,
		DocumentVersion: 1,
		Title:           "Title " + chunkID,
		SourceURI:       "https://example.test/" + chunkID,
		HeadingPath:     []string{"Section"},
		Content:         "Content " + chunkID,
		Score:           score,
		StageScores:     []model.StageScore{{Stage: stage, Rank: rank, Score: score}},
		Metadata:        []byte(`{"kind":"test"}`),
	}
}

func assertChunkOrder(t *testing.T, hits []model.SearchHit, want []string) {
	t.Helper()
	got := make([]string, 0, len(hits))
	for _, hit := range hits {
		got = append(got, hit.ChunkID)
	}
	if !equalStrings(got, want) {
		t.Fatalf("chunk order = %v, want %v", got, want)
	}
}

func assertCandidateIDs(t *testing.T, candidateSet model.CandidateSet, want []string) {
	t.Helper()
	if len(candidateSet.Hits) != len(want) {
		t.Fatalf("candidate set %q length = %d, want %d", candidateSet.Stage, len(candidateSet.Hits), len(want))
	}
	for index, chunkID := range want {
		if candidateSet.Hits[index].ChunkID != chunkID || candidateSet.Hits[index].Rank != index+1 {
			t.Fatalf("candidate set %q hit %d = %+v, want chunk=%q rank=%d", candidateSet.Stage, index, candidateSet.Hits[index], chunkID, index+1)
		}
	}
}

func assertStages(t *testing.T, scores []model.StageScore, want []string) {
	t.Helper()
	got := make([]string, 0, len(scores))
	for _, score := range scores {
		got = append(got, score.Stage)
	}
	if !equalStrings(got, want) {
		t.Fatalf("stage scores = %v, want %v", got, want)
	}
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}
