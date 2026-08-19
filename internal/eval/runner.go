package eval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"slices"
	"strings"
	"time"

	"raghub/internal/model"
)

const (
	ReportSchemaVersion = "raghub.eval.report/v1"
	SmokeStatus         = "smoke"
	IncompleteStatus    = "incomplete"
)

type Ingestor interface {
	Ingest(ctx context.Context, document model.DocumentInput) (model.IngestResult, error)
}

type Searcher interface {
	Search(ctx context.Context, request model.SearchRequest) (model.SearchResult, error)
}

type Options struct {
	TopK            int
	RetrieverName   string
	RetrieverConfig map[string]any
	DatabaseVersion string
	CodeRevision    string
	Command         string
	Clock           func() time.Time
}

type Manifest struct {
	SchemaVersion string            `json:"schema_version"`
	RunID         string            `json:"run_id"`
	Status        string            `json:"status"`
	Command       string            `json:"command"`
	CorpusSHA256  string            `json:"corpus_sha256"`
	Dataset       DatasetManifest   `json:"dataset"`
	Retriever     RetrieverManifest `json:"retriever"`
	Runtime       RuntimeManifest   `json:"runtime"`
	StartedAt     time.Time         `json:"started_at"`
	FinishedAt    time.Time         `json:"finished_at"`
	TopK          int               `json:"top_k"`
	Ingestions    []IngestionRecord `json:"ingestions"`
	Summary       Summary           `json:"summary"`
	PerQuery      []QueryResult     `json:"per_query"`
	Error         string            `json:"error,omitempty"`
}

type DatasetManifest struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
}

type RetrieverManifest struct {
	Name         string         `json:"name"`
	Config       map[string]any `json:"config"`
	ConfigSHA256 string         `json:"config_sha256"`
}

type RuntimeManifest struct {
	GoVersion       string `json:"go_version"`
	DatabaseVersion string `json:"database_version,omitempty"`
	CodeRevision    string `json:"code_revision,omitempty"`
}

type IngestionRecord struct {
	TenantID   string   `json:"tenant_id"`
	DocumentID string   `json:"document_id"`
	Version    int      `json:"version,omitempty"`
	ChunkIDs   []string `json:"chunk_ids,omitempty"`
	Unchanged  bool     `json:"unchanged"`
	Error      string   `json:"error,omitempty"`
}

type HitRecord struct {
	Rank int `json:"rank"`
	model.SearchHit
}

type QueryResult struct {
	ID                string             `json:"id"`
	Category          string             `json:"category"`
	TenantID          string             `json:"tenant_id"`
	PrincipalID       string             `json:"principal_id,omitempty"`
	Query             string             `json:"query"`
	GoldChunkIDs      []string           `json:"gold_chunk_ids"`
	ForbiddenChunkIDs []string           `json:"forbidden_chunk_ids,omitempty"`
	ForbiddenHits     []string           `json:"forbidden_hits,omitempty"`
	Hits              []HitRecord        `json:"hits"`
	Traces            []model.StageTrace `json:"traces,omitempty"`
	Metrics           RankingMetrics     `json:"metrics"`
	LatencyMS         float64            `json:"latency_ms"`
	Error             string             `json:"error,omitempty"`
}

type Summary struct {
	QueryCount        int                `json:"query_count"`
	SearchErrorCount  int                `json:"search_error_count"`
	ForbiddenHitCount int                `json:"forbidden_hit_count"`
	Gates             GateSummary        `json:"gates"`
	Metrics           RankingMetrics     `json:"metrics"`
	Latency           LatencyPercentiles `json:"latency"`
}

// GateSummary separates deterministic correctness and security gates from
// retrieval-quality averages. Pass is true only when all three gates pass.
type GateSummary struct {
	Pass                  bool `json:"pass"`
	CorpusReferencesValid bool `json:"corpus_references_valid"`
	SearchCompleted       bool `json:"search_completed"`
	NoForbiddenHits       bool `json:"no_forbidden_hits"`
}

type Runner struct {
	ingestor Ingestor
	searcher Searcher
}

func NewRunner(ingestor Ingestor, searcher Searcher) *Runner {
	return &Runner{ingestor: ingestor, searcher: searcher}
}

func (r *Runner) Run(ctx context.Context, loaded LoadedDataset, options Options) (Manifest, error) {
	clock := options.Clock
	if clock == nil {
		clock = time.Now
	}
	manifest := newManifest(loaded, options, clock())
	if r == nil || r.ingestor == nil || r.searcher == nil {
		return finishIncomplete(manifest, clock, errors.New("eval runner requires ingestor and searcher"))
	}
	if options.TopK <= 0 {
		return finishIncomplete(manifest, clock, errors.New("eval top_k must be positive"))
	}
	if strings.TrimSpace(options.RetrieverName) == "" {
		return finishIncomplete(manifest, clock, errors.New("eval retriever name is required"))
	}
	if strings.TrimSpace(options.Command) == "" {
		return finishIncomplete(manifest, clock, errors.New("eval command is required"))
	}
	normalizedConfig, configSHA256, err := normalizeAndHashConfig(options.RetrieverConfig)
	if err != nil {
		return finishIncomplete(manifest, clock, fmt.Errorf("hash retriever config: %w", err))
	}
	manifest.Retriever.Config = normalizedConfig
	manifest.Retriever.ConfigSHA256 = configSHA256

	activeChunks := make(map[string]activeChunkOwner)
	for _, document := range loaded.Dataset.Documents {
		record := IngestionRecord{TenantID: document.TenantID, DocumentID: document.DocumentID}
		result, err := r.ingestor.Ingest(ctx, document.documentInput())
		if err != nil {
			record.Error = err.Error()
			manifest.Ingestions = append(manifest.Ingestions, record)
			return finishIncomplete(manifest, clock, fmt.Errorf("ingest %s/%s: %w", document.TenantID, document.DocumentID, err))
		}
		record.Version = result.Version
		record.ChunkIDs = append([]string(nil), result.ChunkIDs...)
		record.Unchanged = result.Unchanged
		manifest.Ingestions = append(manifest.Ingestions, record)
		if result.TenantID != document.TenantID || result.DocumentID != document.DocumentID {
			return finishIncomplete(manifest, clock, fmt.Errorf(
				"ingest %s/%s returned identity %s/%s",
				document.TenantID, document.DocumentID, result.TenantID, result.DocumentID,
			))
		}
		if result.Version <= 0 || len(result.ChunkIDs) == 0 {
			return finishIncomplete(manifest, clock, fmt.Errorf(
				"ingest %s/%s returned invalid active version/chunks: version=%d chunks=%d",
				document.TenantID, document.DocumentID, result.Version, len(result.ChunkIDs),
			))
		}
		for _, chunkID := range result.ChunkIDs {
			if strings.TrimSpace(chunkID) == "" {
				return finishIncomplete(manifest, clock, fmt.Errorf("ingest %s/%s returned an empty active chunk id", document.TenantID, document.DocumentID))
			}
			if owner, duplicate := activeChunks[chunkID]; duplicate {
				return finishIncomplete(manifest, clock, fmt.Errorf(
					"active chunk id %q returned more than once (owners %s/%s and %s/%s)",
					chunkID, owner.TenantID, owner.DocumentID, document.TenantID, document.DocumentID,
				))
			}
			activeChunks[chunkID] = activeChunkOwner{TenantID: document.TenantID, DocumentID: document.DocumentID}
		}
	}
	if err := validateActiveChunkReferences(loaded.Dataset.Queries, activeChunks); err != nil {
		return finishIncomplete(manifest, clock, err)
	}
	manifest.Summary.Gates.CorpusReferencesValid = true

	queryMetrics := make([]RankingMetrics, 0, len(loaded.Dataset.Queries))
	latencies := make([]float64, 0, len(loaded.Dataset.Queries))
	for _, query := range loaded.Dataset.Queries {
		started := clock()
		searchResult, searchErr := r.searcher.Search(ctx, model.SearchRequest{
			TenantID:    query.TenantID,
			PrincipalID: query.PrincipalID,
			Query:       query.Query,
			TopK:        options.TopK,
		})
		latencyMS := max(0, clock().Sub(started).Seconds()*1000)
		latencies = append(latencies, latencyMS)

		result := QueryResult{
			ID:                query.ID,
			Category:          query.Category,
			TenantID:          query.TenantID,
			PrincipalID:       query.PrincipalID,
			Query:             query.Query,
			GoldChunkIDs:      append([]string(nil), query.GoldChunkIDs...),
			ForbiddenChunkIDs: append([]string(nil), query.ForbiddenChunkIDs...),
			Hits:              []HitRecord{},
			LatencyMS:         latencyMS,
		}
		if searchErr != nil {
			result.Error = searchErr.Error()
			manifest.Summary.SearchErrorCount++
		} else {
			allRankedIDs := make([]string, 0, len(searchResult.Hits))
			for _, hit := range searchResult.Hits {
				allRankedIDs = append(allRankedIDs, hit.ChunkID)
			}
			hits := searchResult.Hits
			if len(hits) > options.TopK {
				hits = hits[:options.TopK]
			}
			rankedIDs := make([]string, 0, len(hits))
			for rank, hit := range hits {
				rankedIDs = append(rankedIDs, hit.ChunkID)
				result.Hits = append(result.Hits, HitRecord{Rank: rank + 1, SearchHit: hit})
			}
			result.Traces = append([]model.StageTrace(nil), searchResult.Traces...)
			result.Metrics = EvaluateRanking(rankedIDs, query.GoldChunkIDs, options.TopK)
			// Security checks inspect everything the backend returned, even if a
			// buggy implementation violates TopK and leaks only in the tail.
			result.ForbiddenHits = forbiddenHits(allRankedIDs, query.ForbiddenChunkIDs)
			manifest.Summary.ForbiddenHitCount += len(result.ForbiddenHits)
		}
		queryMetrics = append(queryMetrics, result.Metrics)
		manifest.PerQuery = append(manifest.PerQuery, result)
	}

	manifest.Summary.QueryCount = len(loaded.Dataset.Queries)
	manifest.Summary.Metrics = meanMetrics(queryMetrics)
	manifest.Summary.Latency = latencyPercentiles(latencies)
	manifest.Summary.Gates.SearchCompleted = manifest.Summary.SearchErrorCount == 0
	manifest.Summary.Gates.NoForbiddenHits = manifest.Summary.ForbiddenHitCount == 0
	manifest.Summary.Gates.Pass = manifest.Summary.Gates.CorpusReferencesValid &&
		manifest.Summary.Gates.SearchCompleted && manifest.Summary.Gates.NoForbiddenHits
	manifest.FinishedAt = clock().UTC()
	if manifest.Summary.SearchErrorCount > 0 {
		err := fmt.Errorf("evaluation incomplete: %d of %d searches failed", manifest.Summary.SearchErrorCount, manifest.Summary.QueryCount)
		manifest.Status = IncompleteStatus
		manifest.Error = err.Error()
		return manifest, err
	}
	if manifest.Summary.ForbiddenHitCount > 0 {
		err := fmt.Errorf("evaluation safety gate failed: %d forbidden chunk hit(s)", manifest.Summary.ForbiddenHitCount)
		manifest.Error = err.Error()
		return manifest, err
	}
	return manifest, nil
}

func newManifest(loaded LoadedDataset, options Options, started time.Time) Manifest {
	return Manifest{
		SchemaVersion: ReportSchemaVersion,
		RunID:         evaluationRunID(started, loaded.SHA256),
		Status:        SmokeStatus,
		Command:       options.Command,
		// The v1 dataset is self-contained, so its exact-byte digest identifies
		// both the query specification and the corpus snapshot.
		CorpusSHA256: loaded.SHA256,
		Dataset: DatasetManifest{
			Name:    loaded.Dataset.Name,
			Version: loaded.Dataset.Version,
			SHA256:  loaded.SHA256,
		},
		// Config is populated only after it has been normalized and hashed. An
		// invalid caller value therefore still yields a serializable manifest.
		Retriever: RetrieverManifest{Name: options.RetrieverName, Config: map[string]any{}},
		Runtime: RuntimeManifest{
			GoVersion:       runtime.Version(),
			DatabaseVersion: options.DatabaseVersion,
			CodeRevision:    options.CodeRevision,
		},
		StartedAt:  started.UTC(),
		TopK:       options.TopK,
		Ingestions: []IngestionRecord{},
		PerQuery:   []QueryResult{},
	}
}

func evaluationRunID(started time.Time, datasetSHA256 string) string {
	digestPrefix := datasetSHA256
	if len(digestPrefix) > 12 {
		digestPrefix = digestPrefix[:12]
	}
	if digestPrefix == "" {
		digestPrefix = "unknown"
	}
	return fmt.Sprintf("run-%s-%s", started.UTC().Format("20060102T150405.000000000Z"), digestPrefix)
}

func finishIncomplete(manifest Manifest, clock func() time.Time, err error) (Manifest, error) {
	manifest.Status = IncompleteStatus
	manifest.FinishedAt = clock().UTC()
	manifest.Error = err.Error()
	return manifest, err
}

type activeChunkOwner struct {
	TenantID   string
	DocumentID string
}

func validateActiveChunkReferences(queries []QueryCase, activeChunks map[string]activeChunkOwner) error {
	var missing []string
	for _, query := range queries {
		for _, chunkID := range query.GoldChunkIDs {
			owner, active := activeChunks[chunkID]
			if !active {
				missing = append(missing, fmt.Sprintf("query=%s kind=gold chunk=%s", query.ID, chunkID))
			} else if owner.TenantID != query.TenantID {
				missing = append(missing, fmt.Sprintf(
					"query=%s kind=gold chunk=%s owner_tenant=%s query_tenant=%s",
					query.ID, chunkID, owner.TenantID, query.TenantID,
				))
			}
		}
		for _, chunkID := range query.ForbiddenChunkIDs {
			if _, active := activeChunks[chunkID]; !active {
				missing = append(missing, fmt.Sprintf("query=%s kind=forbidden chunk=%s", query.ID, chunkID))
			}
		}
	}
	if len(missing) == 0 {
		return nil
	}
	slices.Sort(missing)
	return fmt.Errorf("dataset references chunks outside this run's active corpus: %s", strings.Join(missing, "; "))
}

// normalizeAndHashConfig returns the same deep-copied configuration object that
// is written to the manifest and hashes its compact canonical JSON.
// encoding/json emits map keys in lexical order, making semantically identical
// string-keyed maps stable across insertion order.
func normalizeAndHashConfig(value map[string]any) (map[string]any, string, error) {
	if value == nil {
		value = map[string]any{}
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil, "", err
	}
	var normalized map[string]any
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.UseNumber()
	if err := decoder.Decode(&normalized); err != nil {
		return nil, "", err
	}
	canonical, err := json.Marshal(normalized)
	if err != nil {
		return nil, "", err
	}
	digest := sha256.Sum256(canonical)
	return normalized, hex.EncodeToString(digest[:]), nil
}

func forbiddenHits(rankedIDs, forbiddenIDs []string) []string {
	forbidden := make(map[string]struct{}, len(forbiddenIDs))
	for _, id := range forbiddenIDs {
		forbidden[id] = struct{}{}
	}
	var result []string
	seen := make(map[string]struct{})
	for _, id := range rankedIDs {
		if _, blocked := forbidden[id]; !blocked {
			continue
		}
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		seen[id] = struct{}{}
		result = append(result, id)
	}
	slices.Sort(result)
	return result
}

// MarshalManifest produces indented JSON and verifies that retriever config is
// serializable before a CLI writes the evidence artifact.
func MarshalManifest(manifest Manifest) ([]byte, error) {
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal eval manifest: %w", err)
	}
	return append(data, '\n'), nil
}
