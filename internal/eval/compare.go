package eval

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"slices"
	"strings"
)

const (
	ComparisonSchemaVersion         = "raghub.eval.comparison/v1"
	ThreeWayComparisonSchemaVersion = "raghub.eval.comparison/v2"
	threeWayHybridRRFK              = 60
	threeWayFTSCandidateK           = 20
	threeWayDenseCandidateK         = 20
)

// Comparison is a paired smoke-run comparison. It deliberately carries no
// complete/pass verdict: smoke scope and deterministic run gates are preserved
// separately from metric deltas.
type Comparison struct {
	SchemaVersion string          `json:"schema_version"`
	Status        string          `json:"status"`
	Dataset       DatasetManifest `json:"dataset"`
	CorpusSHA256  string          `json:"corpus_sha256"`
	TopK          int             `json:"top_k"`
	QueryCount    int             `json:"query_count"`
	Baseline      ComparisonSide  `json:"baseline"`
	Candidate     ComparisonSide  `json:"candidate"`
	Delta         ComparisonDelta `json:"delta"`
}

type ComparisonSide struct {
	RunID     string             `json:"run_id"`
	Retriever RetrieverManifest  `json:"retriever"`
	Runtime   RuntimeManifest    `json:"runtime"`
	Metrics   RankingMetrics     `json:"metrics"`
	Latency   LatencyPercentiles `json:"latency"`
}

type ComparisonDelta struct {
	Direction string             `json:"direction"`
	Metrics   RankingMetrics     `json:"metrics"`
	Latency   LatencyPercentiles `json:"latency"`
}

// ThreeWayComparison is the strict FTS/Dense/Hybrid comparison artifact. The
// v1 pairwise shape remains available for earlier evidence and ad-hoc ablation
// comparisons.
type ThreeWayComparison struct {
	SchemaVersion string                  `json:"schema_version"`
	Status        string                  `json:"status"`
	Dataset       DatasetManifest         `json:"dataset"`
	CorpusSHA256  string                  `json:"corpus_sha256"`
	TopK          int                     `json:"top_k"`
	QueryCount    int                     `json:"query_count"`
	FTS           ComparisonSide          `json:"fts"`
	Dense         ComparisonSide          `json:"dense"`
	Hybrid        ComparisonSide          `json:"hybrid"`
	Categories    []ThreeWayCategory      `json:"categories"`
	Deltas        ThreeWayComparisonDelta `json:"deltas"`
}

type ThreeWayCategory struct {
	Category   string         `json:"category"`
	QueryCount int            `json:"query_count"`
	FTS        RankingMetrics `json:"fts"`
	Dense      RankingMetrics `json:"dense"`
	Hybrid     RankingMetrics `json:"hybrid"`
}

type ThreeWayComparisonDelta struct {
	DenseMinusFTS    ComparisonDelta `json:"dense_minus_fts"`
	HybridMinusFTS   ComparisonDelta `json:"hybrid_minus_fts"`
	HybridMinusDense ComparisonDelta `json:"hybrid_minus_dense"`
}

func LoadManifest(path string) (Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Manifest{}, fmt.Errorf("read manifest: %w", err)
	}
	return ParseManifest(data)
}

// ParseManifest rejects unknown fields and any bytes after the single JSON
// object. Pairing semantics are validated by CompareManifests.
func ParseManifest(data []byte) (Manifest, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	var manifest Manifest
	if err := decoder.Decode(&manifest); err != nil {
		return Manifest{}, fmt.Errorf("decode manifest: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return Manifest{}, errors.New("decode manifest: input must contain exactly one JSON object")
		}
		return Manifest{}, fmt.Errorf("decode manifest trailing data: %w", err)
	}
	return manifest, nil
}

func CompareManifests(baseline, candidate Manifest) (Comparison, error) {
	if err := validateComparableManifest("baseline", baseline); err != nil {
		return Comparison{}, err
	}
	if err := validateComparableManifest("candidate", candidate); err != nil {
		return Comparison{}, err
	}
	if baseline.Dataset != candidate.Dataset {
		return Comparison{}, errors.New("pair manifests: dataset name/version/sha256 differ")
	}
	if baseline.CorpusSHA256 != candidate.CorpusSHA256 {
		return Comparison{}, errors.New("pair manifests: corpus_sha256 differs")
	}
	if baseline.TopK != candidate.TopK {
		return Comparison{}, errors.New("pair manifests: top_k differs")
	}
	if baseline.Runtime != candidate.Runtime {
		return Comparison{}, errors.New("pair manifests: runtime go/database/pgvector/code revision differs")
	}
	if len(baseline.PerQuery) != len(candidate.PerQuery) {
		return Comparison{}, fmt.Errorf("pair manifests: per_query length differs: %d != %d", len(baseline.PerQuery), len(candidate.PerQuery))
	}
	for index := range baseline.PerQuery {
		if err := compareQueryCaseIdentity(index, baseline.PerQuery[index], candidate.PerQuery[index]); err != nil {
			return Comparison{}, err
		}
	}

	return Comparison{
		SchemaVersion: ComparisonSchemaVersion,
		Status:        SmokeStatus,
		Dataset:       baseline.Dataset,
		CorpusSHA256:  baseline.CorpusSHA256,
		TopK:          baseline.TopK,
		QueryCount:    len(baseline.PerQuery),
		Baseline:      comparisonSide(baseline),
		Candidate:     comparisonSide(candidate),
		Delta: ComparisonDelta{
			Direction: "candidate-baseline",
			Metrics:   subtractMetrics(candidate.Summary.Metrics, baseline.Summary.Metrics),
			Latency:   subtractLatency(candidate.Summary.Latency, baseline.Summary.Latency),
		},
	}, nil
}

// CompareThreeManifests validates each report independently, verifies every
// pair has identical inputs and runtime provenance, and then labels each side
// by its explicit retriever mode. No score is promoted to a release verdict.
func CompareThreeManifests(fts, dense, hybrid Manifest) (ThreeWayComparison, error) {
	if _, err := CompareManifests(fts, dense); err != nil {
		return ThreeWayComparison{}, fmt.Errorf("compare FTS and Dense: %w", err)
	}
	if _, err := CompareManifests(fts, hybrid); err != nil {
		return ThreeWayComparison{}, fmt.Errorf("compare FTS and Hybrid: %w", err)
	}
	if _, err := CompareManifests(dense, hybrid); err != nil {
		return ThreeWayComparison{}, fmt.Errorf("compare Dense and Hybrid: %w", err)
	}
	if err := validateThreeWaySide("FTS", fts, "postgres_fts", "fts"); err != nil {
		return ThreeWayComparison{}, err
	}
	if err := validateThreeWaySide("Dense", dense, "postgres_dense", "dense"); err != nil {
		return ThreeWayComparison{}, err
	}
	if err := validateThreeWaySide("Hybrid", hybrid, "postgres_hybrid_rrf", "hybrid"); err != nil {
		return ThreeWayComparison{}, err
	}
	if err := validateThreeWayBranchConfigs(fts, dense, hybrid); err != nil {
		return ThreeWayComparison{}, err
	}
	categories, err := threeWayCategories(fts, dense, hybrid)
	if err != nil {
		return ThreeWayComparison{}, err
	}

	return ThreeWayComparison{
		SchemaVersion: ThreeWayComparisonSchemaVersion,
		Status:        SmokeStatus,
		Dataset:       fts.Dataset,
		CorpusSHA256:  fts.CorpusSHA256,
		TopK:          fts.TopK,
		QueryCount:    len(fts.PerQuery),
		FTS:           comparisonSide(fts),
		Dense:         comparisonSide(dense),
		Hybrid:        comparisonSide(hybrid),
		Categories:    categories,
		Deltas: ThreeWayComparisonDelta{
			DenseMinusFTS:    comparisonDelta("dense-minus-fts", dense, fts),
			HybridMinusFTS:   comparisonDelta("hybrid-minus-fts", hybrid, fts),
			HybridMinusDense: comparisonDelta("hybrid-minus-dense", hybrid, dense),
		},
	}, nil
}

func threeWayCategories(fts, dense, hybrid Manifest) ([]ThreeWayCategory, error) {
	categoryNames := make([]string, 0)
	metricsByCategory := make(map[string][3][]RankingMetrics)
	for index := range fts.PerQuery {
		category := strings.TrimSpace(fts.PerQuery[index].Category)
		if category == "" {
			return nil, fmt.Errorf("three-way comparison: per_query[%d] category is required", index)
		}
		metrics, exists := metricsByCategory[category]
		if !exists {
			categoryNames = append(categoryNames, category)
		}
		metrics[0] = append(metrics[0], fts.PerQuery[index].Metrics)
		metrics[1] = append(metrics[1], dense.PerQuery[index].Metrics)
		metrics[2] = append(metrics[2], hybrid.PerQuery[index].Metrics)
		metricsByCategory[category] = metrics
	}
	slices.Sort(categoryNames)
	categories := make([]ThreeWayCategory, 0, len(categoryNames))
	for _, category := range categoryNames {
		metrics := metricsByCategory[category]
		categories = append(categories, ThreeWayCategory{
			Category:   category,
			QueryCount: len(metrics[0]),
			FTS:        meanMetrics(metrics[0]),
			Dense:      meanMetrics(metrics[1]),
			Hybrid:     meanMetrics(metrics[2]),
		})
	}
	return categories, nil
}

func validateThreeWaySide(label string, manifest Manifest, wantName, wantMode string) error {
	if manifest.Retriever.Name != wantName {
		return fmt.Errorf("validate %s manifest: retriever name must be %q", label, wantName)
	}
	value, ok := manifest.Retriever.Config["mode"]
	if !ok {
		return fmt.Errorf("validate %s manifest: retriever config mode is required", label)
	}
	mode, ok := value.(string)
	if !ok || mode != wantMode {
		return fmt.Errorf("validate %s manifest: retriever config mode must be %q", label, wantMode)
	}
	if mode == "hybrid" {
		if err := validateThreeWayHybridConfig(manifest.Retriever.Config); err != nil {
			return fmt.Errorf("validate %s manifest: %w", label, err)
		}
	}
	for index, query := range manifest.PerQuery {
		if err := validateQueryEvidence(mode, query, manifest.Retriever.Config, manifest.TopK); err != nil {
			return fmt.Errorf("validate %s manifest: per_query[%d]: %w", label, index, err)
		}
	}
	return nil
}

func validateThreeWayBranchConfigs(fts, dense, hybrid Manifest) error {
	for _, branch := range []struct {
		name       string
		standalone map[string]any
	}{
		{name: "fts", standalone: fts.Retriever.Config},
		{name: "dense", standalone: dense.Retriever.Config},
	} {
		nested, ok := hybrid.Retriever.Config[branch.name].(map[string]any)
		if !ok {
			return fmt.Errorf("validate Hybrid manifest: retriever config %s branch is required", branch.name)
		}
		normalizedNested, _, err := normalizeAndHashConfig(nested)
		if err != nil {
			return fmt.Errorf("validate Hybrid manifest: normalize nested %s config: %w", branch.name, err)
		}
		normalizedStandalone, _, err := normalizeAndHashConfig(branch.standalone)
		if err != nil {
			return fmt.Errorf("validate %s manifest: normalize retriever config: %w", strings.ToUpper(branch.name), err)
		}
		if !reflect.DeepEqual(normalizedNested, normalizedStandalone) {
			return fmt.Errorf("three-way comparison: Hybrid nested %s config differs from standalone %s config", branch.name, branch.name)
		}
	}
	return nil
}

func validateThreeWayHybridConfig(config map[string]any) error {
	for key, want := range map[string]string{
		"fusion":         "reciprocal_rank_fusion",
		"branch_failure": "fail_closed",
	} {
		if value, ok := config[key].(string); !ok || value != want {
			return fmt.Errorf("retriever config %s must be %q", key, want)
		}
	}
	for key, want := range map[string]int{
		"rrf_k":             threeWayHybridRRFK,
		"fts_candidate_k":   threeWayFTSCandidateK,
		"dense_candidate_k": threeWayDenseCandidateK,
	} {
		if value, ok := configInteger(config[key]); !ok || value != want {
			return fmt.Errorf("retriever config %s must be %d", key, want)
		}
	}
	for key, wantMode := range map[string]string{"fts": "fts", "dense": "dense"} {
		branch, ok := config[key].(map[string]any)
		if !ok {
			return fmt.Errorf("retriever config %s branch is required", key)
		}
		if value, ok := branch["mode"].(string); !ok || value != wantMode {
			return fmt.Errorf("retriever config %s branch mode must be %q", key, wantMode)
		}
	}
	return nil
}

func configInteger(value any) (int, bool) {
	switch value := value.(type) {
	case int:
		return value, true
	case int64:
		return int(value), int64(int(value)) == value
	case float64:
		converted := int(value)
		return converted, float64(converted) == value
	case json.Number:
		converted, err := value.Int64()
		if err != nil {
			return 0, false
		}
		return int(converted), int64(int(converted)) == converted
	default:
		return 0, false
	}
}

func validateQueryEvidence(mode string, query QueryResult, config map[string]any, topK int) error {
	wantTraces := map[string][]string{
		"fts":    {"fts"},
		"dense":  {"query_embedding", "dense"},
		"hybrid": {"fts", "query_embedding", "dense", "rrf_fusion"},
	}[mode]
	if len(query.Traces) != len(wantTraces) {
		return fmt.Errorf("%s trace stages must be %v", mode, wantTraces)
	}
	for index, trace := range query.Traces {
		if trace.Stage != wantTraces[index] {
			return fmt.Errorf("%s trace stages must be %v", mode, wantTraces)
		}
		if math.IsNaN(trace.DurationMS) || math.IsInf(trace.DurationMS, 0) || trace.DurationMS < 0 {
			return fmt.Errorf("%s trace %q duration must be finite and non-negative", mode, trace.Stage)
		}
	}
	for hitIndex, hit := range query.Hits {
		if math.IsNaN(hit.Score) || math.IsInf(hit.Score, 0) {
			return fmt.Errorf("%s hit %d final score must be finite", mode, hitIndex)
		}
		switch mode {
		case "fts", "dense":
			if len(hit.StageScores) != 1 {
				return fmt.Errorf("%s hit %d must contain exactly one %s stage score", mode, hitIndex, mode)
			}
			score := hit.StageScores[0]
			if score.Stage != mode || score.Rank != hit.Rank || !nearlyEqual(score.Score, hit.Score) ||
				math.IsNaN(score.Score) || math.IsInf(score.Score, 0) {
				return fmt.Errorf("%s hit %d stage score must match final rank and score", mode, hitIndex)
			}
		case "hybrid":
			// Hybrid evidence is validated as a query after all hits are read so
			// source-rank uniqueness and final ordering can be checked together.
		default:
			return fmt.Errorf("unsupported evidence mode %q", mode)
		}
	}
	if mode == "hybrid" {
		if err := validateHybridQueryEvidence(query, config, topK); err != nil {
			return err
		}
	}
	if err := validateCandidateSetEvidence(mode, query, config, topK); err != nil {
		return err
	}
	return nil
}

func validateCandidateSetEvidence(mode string, query QueryResult, config map[string]any, topK int) error {
	wantStages := map[string][]string{
		"fts":    {"fts"},
		"dense":  {"dense"},
		"hybrid": {"fts", "dense"},
	}[mode]
	if len(query.CandidateSets) != len(wantStages) {
		return fmt.Errorf("%s candidate sets must be %v", mode, wantStages)
	}
	sets := make(map[string]map[int]string, len(query.CandidateSets))
	for setIndex, candidateSet := range query.CandidateSets {
		if candidateSet.Stage != wantStages[setIndex] {
			return fmt.Errorf("%s candidate sets must be ordered %v", mode, wantStages)
		}
		limit := topK
		if mode == "hybrid" {
			floorKey := candidateSet.Stage + "_candidate_k"
			floor, _ := configInteger(config[floorKey])
			limit = max(topK, floor)
		}
		if len(candidateSet.Hits) > limit {
			return fmt.Errorf("%s candidate set %q has %d hits, greater than effective depth %d", mode, candidateSet.Stage, len(candidateSet.Hits), limit)
		}
		seen := make(map[string]struct{}, len(candidateSet.Hits))
		byRank := make(map[int]string, len(candidateSet.Hits))
		for index, candidate := range candidateSet.Hits {
			if candidate.Rank != index+1 || strings.TrimSpace(candidate.ChunkID) == "" {
				return fmt.Errorf("%s candidate set %q must have non-empty chunks ranked continuously from 1", mode, candidateSet.Stage)
			}
			if _, duplicate := seen[candidate.ChunkID]; duplicate {
				return fmt.Errorf("%s candidate set %q contains duplicate chunk %q", mode, candidateSet.Stage, candidate.ChunkID)
			}
			seen[candidate.ChunkID] = struct{}{}
			byRank[candidate.Rank] = candidate.ChunkID
		}
		sets[candidateSet.Stage] = byRank
	}

	if mode == "fts" || mode == "dense" {
		candidates := query.CandidateSets[0].Hits
		if len(candidates) != len(query.Hits) {
			return fmt.Errorf("%s final hits must equal its candidate set", mode)
		}
		for index, hit := range query.Hits {
			if candidates[index].ChunkID != hit.ChunkID || candidates[index].Rank != hit.Rank {
				return fmt.Errorf("%s final hits must equal its candidate set", mode)
			}
		}
		return nil
	}

	for hitIndex, hit := range query.Hits {
		foundSource := false
		for _, score := range hit.StageScores {
			if score.Stage != "fts" && score.Stage != "dense" {
				continue
			}
			foundSource = true
			if sets[score.Stage][score.Rank] != hit.ChunkID {
				return fmt.Errorf("hybrid hit %d %s source rank does not match the recorded candidate set", hitIndex, score.Stage)
			}
		}
		if !foundSource {
			return fmt.Errorf("hybrid hit %d has no recorded source candidate", hitIndex)
		}
	}
	return nil
}

func validateHybridQueryEvidence(query QueryResult, config map[string]any, topK int) error {
	rrfK, _ := configInteger(config["rrf_k"])
	ftsCandidateK, _ := configInteger(config["fts_candidate_k"])
	denseCandidateK, _ := configInteger(config["dense_candidate_k"])
	branchLimits := map[string]int{
		"fts":   max(topK, ftsCandidateK),
		"dense": max(topK, denseCandidateK),
	}
	seenRanks := map[string]map[int]struct{}{
		"fts":   {},
		"dense": {},
	}
	for hitIndex, hit := range query.Hits {
		if err := validateHybridHitEvidence(hitIndex, hit, rrfK, branchLimits, seenRanks); err != nil {
			return err
		}
		if hitIndex == 0 {
			continue
		}
		previous := query.Hits[hitIndex-1]
		if previous.Score < hit.Score || (previous.Score == hit.Score && previous.ChunkID > hit.ChunkID) {
			return fmt.Errorf("hybrid final hits must be ordered by score descending then chunk_id ascending")
		}
	}
	return nil
}

func validateHybridHitEvidence(
	hitIndex int,
	hit HitRecord,
	rrfK int,
	branchLimits map[string]int,
	seenRanks map[string]map[int]struct{},
) error {
	if len(hit.StageScores) < 2 || len(hit.StageScores) > 3 {
		return fmt.Errorf("hybrid hit %d must contain one or two source scores followed by rrf", hitIndex)
	}
	sourceScores := hit.StageScores[:len(hit.StageScores)-1]
	if len(sourceScores) == 1 && sourceScores[0].Stage != "fts" && sourceScores[0].Stage != "dense" {
		return fmt.Errorf("hybrid hit %d source stage must be fts or dense", hitIndex)
	}
	if len(sourceScores) == 2 && (sourceScores[0].Stage != "fts" || sourceScores[1].Stage != "dense") {
		return fmt.Errorf("hybrid hit %d source stages must be ordered fts then dense", hitIndex)
	}
	expectedRRF := 0.0
	for _, score := range sourceScores {
		if score.Rank <= 0 || math.IsNaN(score.Score) || math.IsInf(score.Score, 0) {
			return fmt.Errorf("hybrid hit %d source rank and score must be positive-rank finite evidence", hitIndex)
		}
		if score.Rank > branchLimits[score.Stage] {
			return fmt.Errorf("hybrid hit %d %s source rank %d exceeds candidate depth %d", hitIndex, score.Stage, score.Rank, branchLimits[score.Stage])
		}
		if _, duplicate := seenRanks[score.Stage][score.Rank]; duplicate {
			return fmt.Errorf("hybrid %s source rank %d appears more than once", score.Stage, score.Rank)
		}
		seenRanks[score.Stage][score.Rank] = struct{}{}
		expectedRRF += 1 / float64(rrfK+score.Rank)
	}
	rrf := hit.StageScores[len(hit.StageScores)-1]
	if rrf.Stage != "rrf" || rrf.Rank != hit.Rank || !nearlyEqual(rrf.Score, hit.Score) ||
		math.IsNaN(rrf.Score) || math.IsInf(rrf.Score, 0) {
		return fmt.Errorf("hybrid hit %d rrf stage must match final rank and score", hitIndex)
	}
	if !nearlyEqual(hit.Score, expectedRRF) {
		return fmt.Errorf("hybrid hit %d rrf score does not match source ranks and rrf_k", hitIndex)
	}
	return nil
}

func comparisonDelta(direction string, candidate, baseline Manifest) ComparisonDelta {
	return ComparisonDelta{
		Direction: direction,
		Metrics:   subtractMetrics(candidate.Summary.Metrics, baseline.Summary.Metrics),
		Latency:   subtractLatency(candidate.Summary.Latency, baseline.Summary.Latency),
	}
}

func validateComparableManifest(side string, manifest Manifest) error {
	prefix := "validate " + side + " manifest"
	if manifest.SchemaVersion != ReportSchemaVersion {
		return fmt.Errorf("%s: schema_version must be %q", prefix, ReportSchemaVersion)
	}
	if manifest.Status != SmokeStatus {
		return fmt.Errorf("%s: status must be %q", prefix, SmokeStatus)
	}
	if !manifest.Summary.Gates.Pass {
		return fmt.Errorf("%s: gates.pass must be true", prefix)
	}
	if !manifest.Summary.Gates.CorpusReferencesValid || !manifest.Summary.Gates.CorpusIsolated ||
		!manifest.Summary.Gates.SearchCompleted || !manifest.Summary.Gates.NoForbiddenHits {
		return fmt.Errorf("%s: all deterministic gates must be true when gates.pass is true", prefix)
	}
	if manifest.Summary.SearchErrorCount != 0 || manifest.Summary.ForbiddenHitCount != 0 || manifest.Error != "" {
		return fmt.Errorf("%s: passing smoke manifest contains errors or forbidden hits", prefix)
	}
	if strings.TrimSpace(manifest.Dataset.Name) == "" || strings.TrimSpace(manifest.Dataset.Version) == "" || strings.TrimSpace(manifest.Dataset.SHA256) == "" {
		return fmt.Errorf("%s: dataset name/version/sha256 are required", prefix)
	}
	if strings.TrimSpace(manifest.CorpusSHA256) == "" {
		return fmt.Errorf("%s: corpus_sha256 is required", prefix)
	}
	if manifest.TopK <= 0 {
		return fmt.Errorf("%s: top_k must be positive", prefix)
	}
	if manifest.Summary.QueryCount != len(manifest.PerQuery) {
		return fmt.Errorf("%s: summary query_count=%d differs from per_query length=%d", prefix, manifest.Summary.QueryCount, len(manifest.PerQuery))
	}
	if manifest.Summary.QueryCount == 0 {
		return fmt.Errorf("%s: at least one per_query result is required", prefix)
	}
	if err := validateMetricRange(manifest.Summary.Metrics); err != nil {
		return fmt.Errorf("%s: %w", prefix, err)
	}
	if err := validateLatency(manifest.Summary.Latency); err != nil {
		return fmt.Errorf("%s: %w", prefix, err)
	}
	if strings.TrimSpace(manifest.Retriever.Name) == "" || strings.TrimSpace(manifest.Retriever.ConfigSHA256) == "" {
		return fmt.Errorf("%s: retriever name and config_sha256 are required", prefix)
	}
	_, configSHA256, err := normalizeAndHashConfig(manifest.Retriever.Config)
	if err != nil {
		return fmt.Errorf("%s: hash retriever config: %w", prefix, err)
	}
	if configSHA256 != manifest.Retriever.ConfigSHA256 {
		return fmt.Errorf("%s: retriever config_sha256 does not match config", prefix)
	}
	if strings.TrimSpace(manifest.Runtime.GoVersion) == "" || strings.TrimSpace(manifest.Runtime.DatabaseVersion) == "" ||
		strings.TrimSpace(manifest.Runtime.VectorExtensionVersion) == "" || strings.TrimSpace(manifest.Runtime.CodeRevision) == "" {
		return fmt.Errorf("%s: runtime go/database/pgvector/code revision are required", prefix)
	}
	revision := strings.TrimSpace(manifest.Runtime.CodeRevision)
	if revision == "uncommitted" || strings.Contains(revision, "+dirty") {
		return fmt.Errorf("%s: code_revision must identify a clean committed revision", prefix)
	}

	seenQueryIDs := make(map[string]struct{}, len(manifest.PerQuery))
	queryMetrics := make([]RankingMetrics, 0, len(manifest.PerQuery))
	latencies := make([]float64, 0, len(manifest.PerQuery))
	for index, query := range manifest.PerQuery {
		if strings.TrimSpace(query.ID) == "" {
			return fmt.Errorf("%s: per_query[%d].id is required", prefix, index)
		}
		if _, duplicate := seenQueryIDs[query.ID]; duplicate {
			return fmt.Errorf("%s: duplicate per_query id %q", prefix, query.ID)
		}
		seenQueryIDs[query.ID] = struct{}{}
		if query.Error != "" || len(query.ForbiddenHits) != 0 {
			return fmt.Errorf("%s: per_query[%d] contains an error or forbidden hit", prefix, index)
		}
		if len(query.GoldChunkIDs) == 0 || hasBlankOrDuplicate(query.GoldChunkIDs) || hasBlankOrDuplicate(query.ForbiddenChunkIDs) {
			return fmt.Errorf("%s: per_query[%d] has invalid gold/forbidden chunk IDs", prefix, index)
		}
		gold := make(map[string]struct{}, len(query.GoldChunkIDs))
		for _, chunkID := range query.GoldChunkIDs {
			gold[chunkID] = struct{}{}
		}
		for _, chunkID := range query.ForbiddenChunkIDs {
			if _, overlap := gold[chunkID]; overlap {
				return fmt.Errorf("%s: per_query[%d] chunk %q is both gold and forbidden", prefix, index, chunkID)
			}
		}
		if err := validateMetricRange(query.Metrics); err != nil {
			return fmt.Errorf("%s: per_query[%d]: %w", prefix, index, err)
		}
		if math.IsNaN(query.LatencyMS) || math.IsInf(query.LatencyMS, 0) || query.LatencyMS < 0 {
			return fmt.Errorf("%s: per_query[%d].latency_ms must be finite and non-negative", prefix, index)
		}
		if len(query.Hits) > manifest.TopK {
			return fmt.Errorf("%s: per_query[%d] has %d hits, greater than top_k=%d", prefix, index, len(query.Hits), manifest.TopK)
		}
		rankedIDs := make([]string, 0, len(query.Hits))
		seenHitIDs := make(map[string]struct{}, len(query.Hits))
		for hitIndex, hit := range query.Hits {
			if hit.Rank != hitIndex+1 {
				return fmt.Errorf("%s: per_query[%d] hit ranks must be continuous from 1", prefix, index)
			}
			if strings.TrimSpace(hit.ChunkID) == "" {
				return fmt.Errorf("%s: per_query[%d] hit %d has empty chunk_id", prefix, index, hitIndex)
			}
			if _, duplicate := seenHitIDs[hit.ChunkID]; duplicate {
				return fmt.Errorf("%s: per_query[%d] contains duplicate hit chunk_id %q", prefix, index, hit.ChunkID)
			}
			seenHitIDs[hit.ChunkID] = struct{}{}
			rankedIDs = append(rankedIDs, hit.ChunkID)
		}
		evidenceIDs := append([]string(nil), rankedIDs...)
		for _, candidateSet := range query.CandidateSets {
			for _, candidate := range candidateSet.Hits {
				evidenceIDs = append(evidenceIDs, candidate.ChunkID)
			}
		}
		if leaked := forbiddenHits(evidenceIDs, query.ForbiddenChunkIDs); len(leaked) != 0 {
			return fmt.Errorf("%s: per_query[%d] candidate evidence contains forbidden chunks %v", prefix, index, leaked)
		}
		recomputed := EvaluateRanking(rankedIDs, query.GoldChunkIDs, manifest.TopK)
		if !metricsEqual(recomputed, query.Metrics) {
			return fmt.Errorf("%s: per_query[%d] metrics do not match ranked hits", prefix, index)
		}
		queryMetrics = append(queryMetrics, recomputed)
		latencies = append(latencies, query.LatencyMS)
	}
	if recomputed := meanMetrics(queryMetrics); !metricsEqual(recomputed, manifest.Summary.Metrics) {
		return fmt.Errorf("%s: summary metrics do not match per_query metrics", prefix)
	}
	if recomputed := latencyPercentiles(latencies); !latencyEqual(recomputed, manifest.Summary.Latency) {
		return fmt.Errorf("%s: summary latency does not match per_query latency", prefix)
	}
	return nil
}

func compareQueryCaseIdentity(index int, baseline, candidate QueryResult) error {
	if baseline.ID != candidate.ID || baseline.Category != candidate.Category ||
		baseline.TenantID != candidate.TenantID || baseline.PrincipalID != candidate.PrincipalID ||
		baseline.Query != candidate.Query {
		return fmt.Errorf("pair manifests: per_query[%d] case identity differs", index)
	}
	if !slices.Equal(baseline.GoldChunkIDs, candidate.GoldChunkIDs) {
		return fmt.Errorf("pair manifests: per_query[%d] gold_chunk_ids differ", index)
	}
	if !slices.Equal(baseline.ForbiddenChunkIDs, candidate.ForbiddenChunkIDs) {
		return fmt.Errorf("pair manifests: per_query[%d] forbidden_chunk_ids differ", index)
	}
	return nil
}

func comparisonSide(manifest Manifest) ComparisonSide {
	return ComparisonSide{
		RunID:     manifest.RunID,
		Retriever: manifest.Retriever,
		Runtime:   manifest.Runtime,
		Metrics:   manifest.Summary.Metrics,
		Latency:   manifest.Summary.Latency,
	}
}

func subtractMetrics(candidate, baseline RankingMetrics) RankingMetrics {
	return RankingMetrics{
		RecallAtK:  candidate.RecallAtK - baseline.RecallAtK,
		HitRateAtK: candidate.HitRateAtK - baseline.HitRateAtK,
		MRR:        candidate.MRR - baseline.MRR,
		NDCGAtK:    candidate.NDCGAtK - baseline.NDCGAtK,
	}
}

func subtractLatency(candidate, baseline LatencyPercentiles) LatencyPercentiles {
	return LatencyPercentiles{
		P50MS: candidate.P50MS - baseline.P50MS,
		P95MS: candidate.P95MS - baseline.P95MS,
	}
}

func validateMetricRange(metrics RankingMetrics) error {
	values := []float64{metrics.RecallAtK, metrics.HitRateAtK, metrics.MRR, metrics.NDCGAtK}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
			return errors.New("ranking metrics must be finite values between 0 and 1")
		}
	}
	return nil
}

func validateLatency(latency LatencyPercentiles) error {
	if math.IsNaN(latency.P50MS) || math.IsInf(latency.P50MS, 0) || latency.P50MS < 0 ||
		math.IsNaN(latency.P95MS) || math.IsInf(latency.P95MS, 0) || latency.P95MS < 0 {
		return errors.New("latency percentiles must be finite and non-negative")
	}
	return nil
}

func metricsEqual(first, second RankingMetrics) bool {
	return nearlyEqual(first.RecallAtK, second.RecallAtK) &&
		nearlyEqual(first.HitRateAtK, second.HitRateAtK) &&
		nearlyEqual(first.MRR, second.MRR) &&
		nearlyEqual(first.NDCGAtK, second.NDCGAtK)
}

func latencyEqual(first, second LatencyPercentiles) bool {
	return nearlyEqual(first.P50MS, second.P50MS) && nearlyEqual(first.P95MS, second.P95MS)
}

func nearlyEqual(first, second float64) bool {
	return math.Abs(first-second) <= 1e-12
}

func MarshalComparison(comparison Comparison) ([]byte, error) {
	data, err := json.MarshalIndent(comparison, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal eval comparison: %w", err)
	}
	return append(data, '\n'), nil
}

func MarshalThreeWayComparison(comparison ThreeWayComparison) ([]byte, error) {
	data, err := json.MarshalIndent(comparison, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal three-way eval comparison: %w", err)
	}
	return append(data, '\n'), nil
}
