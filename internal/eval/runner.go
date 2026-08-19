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
	ReportSchemaVersion = "raghub.eval.report/v2"
	SmokeStatus         = "smoke"
	IncompleteStatus    = "incomplete"
)

type Ingestor interface {
	Ingest(ctx context.Context, document model.DocumentInput) (model.IngestResult, error)
}

type Searcher interface {
	Search(ctx context.Context, request model.SearchRequest) (model.SearchResult, error)
}

type CorpusInspector interface {
	ActiveChunkInventory(ctx context.Context, tenantIDs []string) ([]model.ActiveChunkInventoryEntry, error)
}

type Options struct {
	TopK                   int
	SearchMode             model.SearchMode
	RetrieverName          string
	RetrieverConfig        map[string]any
	DatabaseVersion        string
	VectorExtensionVersion string
	CodeRevision           string
	Command                string
	Clock                  func() time.Time
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
	GoVersion              string `json:"go_version"`
	DatabaseVersion        string `json:"database_version,omitempty"`
	VectorExtensionVersion string `json:"vector_extension_version,omitempty"`
	CodeRevision           string `json:"code_revision,omitempty"`
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
// retrieval-quality averages. Pass is true only when all four gates pass.
type GateSummary struct {
	Pass                  bool `json:"pass"`
	CorpusReferencesValid bool `json:"corpus_references_valid"`
	CorpusIsolated        bool `json:"corpus_isolated"`
	SearchCompleted       bool `json:"search_completed"`
	NoForbiddenHits       bool `json:"no_forbidden_hits"`
}

type Runner struct {
	ingestor        Ingestor
	searcher        Searcher
	corpusInspector CorpusInspector
}

func NewRunner(ingestor Ingestor, searcher Searcher, corpusInspector CorpusInspector) *Runner {
	return &Runner{ingestor: ingestor, searcher: searcher, corpusInspector: corpusInspector}
}

func (r *Runner) Run(ctx context.Context, loaded LoadedDataset, options Options) (Manifest, error) {
	clock := options.Clock
	if clock == nil {
		clock = time.Now
	}
	manifest := newManifest(loaded, options, clock())
	if r == nil || r.ingestor == nil || r.searcher == nil || r.corpusInspector == nil {
		return finishIncomplete(manifest, clock, errors.New("eval runner requires ingestor, searcher, and corpus inspector"))
	}
	if options.TopK <= 0 {
		return finishIncomplete(manifest, clock, errors.New("eval top_k must be positive"))
	}
	if options.SearchMode != model.SearchModeFTS && options.SearchMode != model.SearchModeDense {
		return finishIncomplete(manifest, clock, fmt.Errorf("eval search mode must be %q or %q", model.SearchModeFTS, model.SearchModeDense))
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
	expectedInventory := make(map[string]activeChunkOwner)
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
			owner := activeChunkOwner{TenantID: document.TenantID, DocumentID: document.DocumentID, DocumentVersion: result.Version}
			activeChunks[chunkID] = owner
			expectedInventory[tenantChunkKey(document.TenantID, chunkID)] = owner
		}
	}
	if err := validateActiveChunkReferences(loaded.Dataset.Queries, activeChunks); err != nil {
		return finishIncomplete(manifest, clock, err)
	}
	manifest.Summary.Gates.CorpusReferencesValid = true
	tenantIDs := datasetTenantIDs(loaded.Dataset)
	inventory, err := r.corpusInspector.ActiveChunkInventory(ctx, tenantIDs)
	if err != nil {
		return finishIncomplete(manifest, clock, fmt.Errorf("inspect active corpus: %w", err))
	}
	if err := validateCorpusIsolation(expectedInventory, inventory); err != nil {
		return finishIncomplete(manifest, clock, err)
	}
	corpusSHA256, err := activeCorpusSHA256(inventory)
	if err != nil {
		return finishIncomplete(manifest, clock, fmt.Errorf("hash active corpus inventory: %w", err))
	}
	manifest.CorpusSHA256 = corpusSHA256
	manifest.Summary.Gates.CorpusIsolated = true

	queryMetrics := make([]RankingMetrics, 0, len(loaded.Dataset.Queries))
	latencies := make([]float64, 0, len(loaded.Dataset.Queries))
	for _, query := range loaded.Dataset.Queries {
		started := clock()
		searchResult, searchErr := r.searcher.Search(ctx, model.SearchRequest{
			TenantID:    query.TenantID,
			PrincipalID: query.PrincipalID,
			Query:       query.Query,
			TopK:        options.TopK,
			Mode:        options.SearchMode,
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
		manifest.Summary.Gates.CorpusIsolated && manifest.Summary.Gates.SearchCompleted && manifest.Summary.Gates.NoForbiddenHits
	postInventory, err := r.corpusInspector.ActiveChunkInventory(ctx, tenantIDs)
	if err != nil {
		manifest.Summary.Gates.CorpusIsolated = false
		manifest.Summary.Gates.Pass = false
		return finishIncomplete(manifest, clock, fmt.Errorf("reinspect active corpus after search: %w", err))
	}
	if err := validateCorpusIsolation(expectedInventory, postInventory); err != nil {
		manifest.Summary.Gates.CorpusIsolated = false
		manifest.Summary.Gates.Pass = false
		return finishIncomplete(manifest, clock, fmt.Errorf("active corpus changed during evaluation: %w", err))
	}
	postCorpusSHA256, err := activeCorpusSHA256(postInventory)
	if err != nil {
		manifest.Summary.Gates.CorpusIsolated = false
		manifest.Summary.Gates.Pass = false
		return finishIncomplete(manifest, clock, fmt.Errorf("hash post-search active corpus inventory: %w", err))
	}
	if postCorpusSHA256 != corpusSHA256 {
		manifest.Summary.Gates.CorpusIsolated = false
		manifest.Summary.Gates.Pass = false
		return finishIncomplete(manifest, clock, errors.New("active corpus changed during evaluation: inventory hash differs"))
	}
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
		// CorpusSHA256 is populated only after the complete tenant inventory is
		// proven isolated from extra or missing active chunks.
		CorpusSHA256: "",
		Dataset: DatasetManifest{
			Name:    loaded.Dataset.Name,
			Version: loaded.Dataset.Version,
			SHA256:  loaded.SHA256,
		},
		// Config is populated only after it has been normalized and hashed. An
		// invalid caller value therefore still yields a serializable manifest.
		Retriever: RetrieverManifest{Name: options.RetrieverName, Config: map[string]any{}},
		Runtime: RuntimeManifest{
			GoVersion:              runtime.Version(),
			DatabaseVersion:        options.DatabaseVersion,
			VectorExtensionVersion: options.VectorExtensionVersion,
			CodeRevision:           options.CodeRevision,
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
	TenantID        string
	DocumentID      string
	DocumentVersion int
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

func datasetTenantIDs(dataset Dataset) []string {
	tenants := make(map[string]struct{}, len(dataset.Documents))
	for _, document := range dataset.Documents {
		tenants[document.TenantID] = struct{}{}
	}
	for _, query := range dataset.Queries {
		tenants[query.TenantID] = struct{}{}
	}
	result := make([]string, 0, len(tenants))
	for tenantID := range tenants {
		result = append(result, tenantID)
	}
	slices.Sort(result)
	return result
}

func tenantChunkKey(tenantID, chunkID string) string {
	return tenantID + "\x00" + chunkID
}

func validateCorpusIsolation(expected map[string]activeChunkOwner, inventory []model.ActiveChunkInventoryEntry) error {
	remaining := make(map[string]activeChunkOwner, len(expected))
	for key, owner := range expected {
		remaining[key] = owner
	}
	seen := make(map[string]struct{}, len(inventory))
	var issues []string
	for _, entry := range inventory {
		key := tenantChunkKey(entry.TenantID, entry.ChunkID)
		if _, duplicate := seen[key]; duplicate {
			issues = append(issues, fmt.Sprintf("duplicate inventory tenant=%s chunk=%s", entry.TenantID, entry.ChunkID))
			continue
		}
		seen[key] = struct{}{}
		owner, exists := expected[key]
		if !exists {
			issues = append(issues, fmt.Sprintf(
				"extra active chunk tenant=%s document=%s version=%d chunk=%s",
				entry.TenantID, entry.DocumentID, entry.DocumentVersion, entry.ChunkID,
			))
			continue
		}
		delete(remaining, key)
		if entry.DocumentID != owner.DocumentID || entry.DocumentVersion != owner.DocumentVersion {
			issues = append(issues, fmt.Sprintf(
				"owner/version mismatch tenant=%s chunk=%s expected=%s/v%d actual=%s/v%d",
				entry.TenantID, entry.ChunkID, owner.DocumentID, owner.DocumentVersion, entry.DocumentID, entry.DocumentVersion,
			))
		}
		if entry.Ordinal < 0 || strings.TrimSpace(entry.DocumentFingerprint) == "" ||
			strings.TrimSpace(entry.RawTextSHA256) == "" || strings.TrimSpace(entry.IndexedTextSHA256) == "" {
			issues = append(issues, fmt.Sprintf("incomplete inventory metadata tenant=%s chunk=%s", entry.TenantID, entry.ChunkID))
		}
	}
	for key, owner := range remaining {
		separator := strings.IndexByte(key, 0)
		tenantID, chunkID := key, ""
		if separator >= 0 {
			tenantID, chunkID = key[:separator], key[separator+1:]
		}
		issues = append(issues, fmt.Sprintf(
			"missing active chunk tenant=%s document=%s version=%d chunk=%s",
			tenantID, owner.DocumentID, owner.DocumentVersion, chunkID,
		))
	}
	if len(issues) == 0 {
		return nil
	}
	slices.Sort(issues)
	return fmt.Errorf("active corpus is not isolated: %s", strings.Join(issues, "; "))
}

func activeCorpusSHA256(inventory []model.ActiveChunkInventoryEntry) (string, error) {
	canonical := append([]model.ActiveChunkInventoryEntry(nil), inventory...)
	slices.SortFunc(canonical, compareInventoryEntries)
	data, err := json.Marshal(canonical)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:]), nil
}

func compareInventoryEntries(first, second model.ActiveChunkInventoryEntry) int {
	for _, values := range [][2]string{
		{first.TenantID, second.TenantID},
		{first.DocumentID, second.DocumentID},
		{first.ChunkID, second.ChunkID},
		{first.DocumentFingerprint, second.DocumentFingerprint},
		{first.RawTextSHA256, second.RawTextSHA256},
		{first.IndexedTextSHA256, second.IndexedTextSHA256},
	} {
		if result := strings.Compare(values[0], values[1]); result != 0 {
			return result
		}
	}
	if first.DocumentVersion != second.DocumentVersion {
		if first.DocumentVersion < second.DocumentVersion {
			return -1
		}
		return 1
	}
	if first.Ordinal < second.Ordinal {
		return -1
	}
	if first.Ordinal > second.Ordinal {
		return 1
	}
	return 0
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
