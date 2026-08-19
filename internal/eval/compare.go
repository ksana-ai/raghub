package eval

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"
)

const ComparisonSchemaVersion = "raghub.eval.comparison/v1"

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
