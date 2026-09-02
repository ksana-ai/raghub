package eval

import (
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"github.com/ksana-ai/raghub/internal/model"
)

const (
	CandidateDiagnosisSchemaVersion    = "raghub.eval.candidate-diagnosis/v1"
	RerankerMinimumIncompleteQueries   = 3
	RerankerMinimumRecoverableFraction = 0.5

	DiagnosisComplete            = "complete"
	DiagnosisFusionOrderingGap   = "fusion_ordering_gap"
	DiagnosisMixedGap            = "mixed_gap"
	DiagnosisCandidateGeneration = "candidate_generation_gap"
)

// CandidateDiagnosis separates failures that a reranker could potentially
// repair from failures where neither retrieval branch generated the missing
// gold chunks. It is diagnostic evidence, not a release verdict.
type CandidateDiagnosis struct {
	SchemaVersion   string                       `json:"schema_version"`
	Status          string                       `json:"status"`
	Dataset         DatasetManifest              `json:"dataset"`
	CorpusSHA256    string                       `json:"corpus_sha256"`
	FinalTopK       int                          `json:"final_top_k"`
	FTSCandidateK   int                          `json:"fts_candidate_k"`
	DenseCandidateK int                          `json:"dense_candidate_k"`
	QueryCount      int                          `json:"query_count"`
	FTS             ComparisonSide               `json:"fts_candidate_replay"`
	Dense           ComparisonSide               `json:"dense_candidate_replay"`
	Hybrid          ComparisonSide               `json:"hybrid_final"`
	Summary         CandidateDiagnosisCounts     `json:"summary"`
	RerankerGate    RerankerExperimentGate       `json:"reranker_experiment_gate"`
	Categories      []CandidateDiagnosisCategory `json:"categories"`
	PerQuery        []CandidateDiagnosisQuery    `json:"per_query"`
}

type RerankerExperimentGate struct {
	MinimumIncompleteQueries   int     `json:"minimum_incomplete_queries"`
	MinimumRecoverableFraction float64 `json:"minimum_recoverable_missing_fraction"`
	IncompleteQueries          int     `json:"incomplete_queries"`
	RecoverableMissingFraction float64 `json:"recoverable_missing_fraction"`
	Eligible                   bool    `json:"eligible"`
}

type CandidateDiagnosisCounts struct {
	QueryCount                    int     `json:"query_count"`
	CompleteQueries               int     `json:"complete_queries"`
	FusionOrderingGapQueries      int     `json:"fusion_ordering_gap_queries"`
	MixedGapQueries               int     `json:"mixed_gap_queries"`
	CandidateGenerationGapQueries int     `json:"candidate_generation_gap_queries"`
	MissingGoldCount              int     `json:"missing_gold_count"`
	RecoverableMissingGoldCount   int     `json:"recoverable_missing_gold_count"`
	UnionCandidateRecall          float64 `json:"union_candidate_recall"`
}

type CandidateDiagnosisCategory struct {
	Category string                   `json:"category"`
	Summary  CandidateDiagnosisCounts `json:"summary"`
}

type CandidateDiagnosisQuery struct {
	ID                    string   `json:"id"`
	Category              string   `json:"category"`
	GoldChunkIDs          []string `json:"gold_chunk_ids"`
	HybridGoldChunkIDs    []string `json:"hybrid_gold_chunk_ids"`
	MissingGoldChunkIDs   []string `json:"missing_gold_chunk_ids"`
	FTSCandidateGoldIDs   []string `json:"fts_candidate_gold_ids"`
	DenseCandidateGoldIDs []string `json:"dense_candidate_gold_ids"`
	UnionCandidateGoldIDs []string `json:"union_candidate_gold_ids"`
	HybridRecall          float64  `json:"hybrid_recall"`
	UnionCandidateRecall  float64  `json:"union_candidate_recall"`
	Classification        string   `json:"classification"`
}

// DiagnoseCandidateCoverage validates standalone branch replays at Hybrid's
// configured depths, then classifies misses from the exact candidate sets
// captured by the Hybrid Top K run itself.
func DiagnoseCandidateCoverage(ftsCandidates, denseCandidates, hybridFinal Manifest) (CandidateDiagnosis, error) {
	for _, side := range []struct {
		label    string
		manifest Manifest
		name     string
		mode     string
	}{
		{label: "FTS candidates", manifest: ftsCandidates, name: "postgres_fts", mode: "fts"},
		{label: "Dense candidates", manifest: denseCandidates, name: "postgres_dense", mode: "dense"},
		{label: "Hybrid final", manifest: hybridFinal, name: "postgres_hybrid_rrf", mode: "hybrid"},
	} {
		if err := validateComparableManifest(side.label, side.manifest); err != nil {
			return CandidateDiagnosis{}, err
		}
		if err := validateThreeWaySide(side.label, side.manifest, side.name, side.mode); err != nil {
			return CandidateDiagnosis{}, err
		}
	}
	if err := validateDiagnosisProvenance(ftsCandidates, denseCandidates, hybridFinal); err != nil {
		return CandidateDiagnosis{}, err
	}
	if err := validateThreeWayBranchConfigs(ftsCandidates, denseCandidates, hybridFinal); err != nil {
		return CandidateDiagnosis{}, err
	}

	ftsFloor, _ := configInteger(hybridFinal.Retriever.Config["fts_candidate_k"])
	denseFloor, _ := configInteger(hybridFinal.Retriever.Config["dense_candidate_k"])
	wantFTSK := max(hybridFinal.TopK, ftsFloor)
	wantDenseK := max(hybridFinal.TopK, denseFloor)
	if ftsCandidates.TopK != wantFTSK {
		return CandidateDiagnosis{}, fmt.Errorf("candidate diagnosis: FTS top_k=%d, want Hybrid effective candidate depth %d", ftsCandidates.TopK, wantFTSK)
	}
	if denseCandidates.TopK != wantDenseK {
		return CandidateDiagnosis{}, fmt.Errorf("candidate diagnosis: Dense top_k=%d, want Hybrid effective candidate depth %d", denseCandidates.TopK, wantDenseK)
	}

	queries := make([]CandidateDiagnosisQuery, 0, len(hybridFinal.PerQuery))
	categoryQueries := make(map[string][]CandidateDiagnosisQuery)
	categoryNames := make([]string, 0)
	for index := range hybridFinal.PerQuery {
		ftsQuery := ftsCandidates.PerQuery[index]
		denseQuery := denseCandidates.PerQuery[index]
		hybridQuery := hybridFinal.PerQuery[index]
		if err := compareQueryCaseIdentity(index, ftsQuery, hybridQuery); err != nil {
			return CandidateDiagnosis{}, err
		}
		if err := compareQueryCaseIdentity(index, denseQuery, hybridQuery); err != nil {
			return CandidateDiagnosis{}, err
		}
		diagnosis := diagnoseQuery(hybridQuery)
		queries = append(queries, diagnosis)
		if _, exists := categoryQueries[diagnosis.Category]; !exists {
			categoryNames = append(categoryNames, diagnosis.Category)
		}
		categoryQueries[diagnosis.Category] = append(categoryQueries[diagnosis.Category], diagnosis)
	}

	slices.Sort(categoryNames)
	categories := make([]CandidateDiagnosisCategory, 0, len(categoryNames))
	for _, category := range categoryNames {
		categories = append(categories, CandidateDiagnosisCategory{
			Category: category,
			Summary:  summarizeDiagnosisQueries(categoryQueries[category]),
		})
	}

	summary := summarizeDiagnosisQueries(queries)
	return CandidateDiagnosis{
		SchemaVersion:   CandidateDiagnosisSchemaVersion,
		Status:          SmokeStatus,
		Dataset:         hybridFinal.Dataset,
		CorpusSHA256:    hybridFinal.CorpusSHA256,
		FinalTopK:       hybridFinal.TopK,
		FTSCandidateK:   ftsCandidates.TopK,
		DenseCandidateK: denseCandidates.TopK,
		QueryCount:      len(queries),
		FTS:             comparisonSide(ftsCandidates),
		Dense:           comparisonSide(denseCandidates),
		Hybrid:          comparisonSide(hybridFinal),
		Summary:         summary,
		RerankerGate:    rerankerExperimentGate(summary),
		Categories:      categories,
		PerQuery:        queries,
	}, nil
}

func rerankerExperimentGate(summary CandidateDiagnosisCounts) RerankerExperimentGate {
	gate := RerankerExperimentGate{
		MinimumIncompleteQueries:   RerankerMinimumIncompleteQueries,
		MinimumRecoverableFraction: RerankerMinimumRecoverableFraction,
		IncompleteQueries:          summary.QueryCount - summary.CompleteQueries,
	}
	if summary.MissingGoldCount > 0 {
		gate.RecoverableMissingFraction = float64(summary.RecoverableMissingGoldCount) / float64(summary.MissingGoldCount)
	}
	gate.Eligible = gate.IncompleteQueries >= gate.MinimumIncompleteQueries &&
		gate.RecoverableMissingFraction >= gate.MinimumRecoverableFraction
	return gate
}

func validateDiagnosisProvenance(fts, dense, hybrid Manifest) error {
	if fts.Dataset != dense.Dataset || fts.Dataset != hybrid.Dataset {
		return errors.New("candidate diagnosis: dataset name/version/sha256 differ")
	}
	if fts.CorpusSHA256 != dense.CorpusSHA256 || fts.CorpusSHA256 != hybrid.CorpusSHA256 {
		return errors.New("candidate diagnosis: corpus_sha256 differs")
	}
	if fts.Runtime != dense.Runtime || fts.Runtime != hybrid.Runtime {
		return errors.New("candidate diagnosis: runtime go/database/pgvector/code revision differs")
	}
	if len(fts.PerQuery) != len(dense.PerQuery) || len(fts.PerQuery) != len(hybrid.PerQuery) {
		return errors.New("candidate diagnosis: per_query lengths differ")
	}
	return nil
}

func diagnoseQuery(hybrid QueryResult) CandidateDiagnosisQuery {
	ftsHits := candidateSetIDSet(hybrid.CandidateSets, "fts")
	denseHits := candidateSetIDSet(hybrid.CandidateSets, "dense")
	hybridHits := hitIDSet(hybrid.Hits)

	result := CandidateDiagnosisQuery{
		ID:                    hybrid.ID,
		Category:              hybrid.Category,
		GoldChunkIDs:          append([]string(nil), hybrid.GoldChunkIDs...),
		HybridGoldChunkIDs:    []string{},
		MissingGoldChunkIDs:   []string{},
		FTSCandidateGoldIDs:   []string{},
		DenseCandidateGoldIDs: []string{},
		UnionCandidateGoldIDs: []string{},
	}
	for _, goldID := range hybrid.GoldChunkIDs {
		_, inFTS := ftsHits[goldID]
		_, inDense := denseHits[goldID]
		_, inHybrid := hybridHits[goldID]
		if inHybrid {
			result.HybridGoldChunkIDs = append(result.HybridGoldChunkIDs, goldID)
		} else {
			result.MissingGoldChunkIDs = append(result.MissingGoldChunkIDs, goldID)
		}
		if inFTS {
			result.FTSCandidateGoldIDs = append(result.FTSCandidateGoldIDs, goldID)
		}
		if inDense {
			result.DenseCandidateGoldIDs = append(result.DenseCandidateGoldIDs, goldID)
		}
		if inFTS || inDense {
			result.UnionCandidateGoldIDs = append(result.UnionCandidateGoldIDs, goldID)
		}
	}
	goldCount := float64(len(result.GoldChunkIDs))
	result.HybridRecall = float64(len(result.HybridGoldChunkIDs)) / goldCount
	result.UnionCandidateRecall = float64(len(result.UnionCandidateGoldIDs)) / goldCount

	recoverableMissing := 0
	unionGold := make(map[string]struct{}, len(result.UnionCandidateGoldIDs))
	for _, chunkID := range result.UnionCandidateGoldIDs {
		unionGold[chunkID] = struct{}{}
	}
	for _, chunkID := range result.MissingGoldChunkIDs {
		if _, exists := unionGold[chunkID]; exists {
			recoverableMissing++
		}
	}
	switch {
	case len(result.MissingGoldChunkIDs) == 0:
		result.Classification = DiagnosisComplete
	case recoverableMissing == len(result.MissingGoldChunkIDs):
		result.Classification = DiagnosisFusionOrderingGap
	case recoverableMissing == 0:
		result.Classification = DiagnosisCandidateGeneration
	default:
		result.Classification = DiagnosisMixedGap
	}
	return result
}

func candidateSetIDSet(candidateSets []model.CandidateSet, stage string) map[string]struct{} {
	for _, candidateSet := range candidateSets {
		if candidateSet.Stage != stage {
			continue
		}
		result := make(map[string]struct{}, len(candidateSet.Hits))
		for _, hit := range candidateSet.Hits {
			result[hit.ChunkID] = struct{}{}
		}
		return result
	}
	return map[string]struct{}{}
}

func hitIDSet(hits []HitRecord) map[string]struct{} {
	result := make(map[string]struct{}, len(hits))
	for _, hit := range hits {
		result[hit.ChunkID] = struct{}{}
	}
	return result
}

func summarizeDiagnosisQueries(queries []CandidateDiagnosisQuery) CandidateDiagnosisCounts {
	summary := CandidateDiagnosisCounts{QueryCount: len(queries)}
	unionRecallSum := 0.0
	for _, query := range queries {
		switch query.Classification {
		case DiagnosisComplete:
			summary.CompleteQueries++
		case DiagnosisFusionOrderingGap:
			summary.FusionOrderingGapQueries++
		case DiagnosisMixedGap:
			summary.MixedGapQueries++
		case DiagnosisCandidateGeneration:
			summary.CandidateGenerationGapQueries++
		}
		summary.MissingGoldCount += len(query.MissingGoldChunkIDs)
		unionRecallSum += query.UnionCandidateRecall
		unionSet := make(map[string]struct{}, len(query.UnionCandidateGoldIDs))
		for _, chunkID := range query.UnionCandidateGoldIDs {
			unionSet[chunkID] = struct{}{}
		}
		for _, chunkID := range query.MissingGoldChunkIDs {
			if _, exists := unionSet[chunkID]; exists {
				summary.RecoverableMissingGoldCount++
			}
		}
	}
	if len(queries) > 0 {
		summary.UnionCandidateRecall = unionRecallSum / float64(len(queries))
	}
	return summary
}

func MarshalCandidateDiagnosis(diagnosis CandidateDiagnosis) ([]byte, error) {
	data, err := json.MarshalIndent(diagnosis, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal candidate diagnosis: %w", err)
	}
	return append(data, '\n'), nil
}
