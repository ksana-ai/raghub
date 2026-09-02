// Package postgres implements raghub's versioned ingestion store and
// authorization-scoped full-text and exact pgvector retrievers on PostgreSQL.
package postgres

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/ksana-ai/raghub/internal/model"
)

const (
	defaultSearchTopK         = 5
	maxSearchTopK             = 50
	storedEmbeddingDimensions = 1024
)

// Store persists immutable document versions and retrieves their current
// chunks. A Store is safe for concurrent use.
type Store struct {
	pool *pgxpool.Pool
}

// New constructs a PostgreSQL ingestion store and FTS retriever. Call
// ApplyMigrations before serving traffic.
func New(pool *pgxpool.Pool) *Store {
	return &Store{pool: pool}
}

// SaveDocumentVersion atomically appends a document version and advances the
// active pointer. Requests matching the active fingerprint return that exact
// version without rewriting it.
func (s *Store) SaveDocumentVersion(
	ctx context.Context,
	document model.DocumentInput,
	fingerprint string,
	chunks []model.ChunkDraft,
) (result model.IngestResult, err error) {
	if err := validateSaveInput(s, document, fingerprint, chunks); err != nil {
		return model.IngestResult{}, err
	}

	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return model.IngestResult{}, fmt.Errorf("begin save document transaction: %w", err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()

	if _, err = tx.Exec(ctx, `
INSERT INTO documents (tenant_id, document_id)
VALUES ($1, $2)
ON CONFLICT (tenant_id, document_id) DO NOTHING`,
		document.TenantID,
		document.ID,
	); err != nil {
		return model.IngestResult{}, fmt.Errorf("ensure document row: %w", err)
	}

	var currentVersion int
	if err = tx.QueryRow(ctx, `
SELECT current_version
FROM documents
WHERE tenant_id = $1 AND document_id = $2
FOR UPDATE`,
		document.TenantID,
		document.ID,
	).Scan(&currentVersion); err != nil {
		return model.IngestResult{}, fmt.Errorf("lock document row: %w", err)
	}

	if currentVersion > 0 {
		unchanged, unchangedErr := loadUnchangedResult(
			ctx,
			tx,
			document.TenantID,
			document.ID,
			currentVersion,
			fingerprint,
		)
		if unchangedErr != nil {
			return model.IngestResult{}, unchangedErr
		}
		if unchanged != nil {
			if err = persistChunkEmbeddings(ctx, tx, document.TenantID, unchanged.ChunkIDs, chunks, true); err != nil {
				return model.IngestResult{}, fmt.Errorf("backfill active document embeddings: %w", err)
			}
			if err = tx.Commit(ctx); err != nil {
				return model.IngestResult{}, fmt.Errorf("commit idempotent document save: %w", err)
			}
			return *unchanged, nil
		}
	}

	version := currentVersion + 1
	allowedPrincipals := document.AllowedPrincipals
	if allowedPrincipals == nil {
		allowedPrincipals = []string{}
	}
	var createdAt time.Time
	if err = tx.QueryRow(ctx, `
INSERT INTO document_versions (
    tenant_id,
    document_id,
    version,
    fingerprint,
    title,
    source_uri,
    raw_content,
    allowed_principals,
    metadata
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb)
RETURNING created_at`,
		document.TenantID,
		document.ID,
		version,
		fingerprint,
		document.Title,
		document.SourceURI,
		document.Content,
		allowedPrincipals,
		[]byte(document.Metadata),
	).Scan(&createdAt); err != nil {
		return model.IngestResult{}, fmt.Errorf("insert document version %d: %w", version, err)
	}

	chunkIDs := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		chunkID := versionedChunkID(document.ID, version, chunk.Ordinal)
		headingPath := chunk.HeadingPath
		if headingPath == nil {
			headingPath = []string{}
		}
		if _, err = tx.Exec(ctx, `
INSERT INTO chunks (
    tenant_id,
    chunk_id,
    document_id,
    document_version,
    ordinal,
    title,
    heading_path,
    heading_text,
    raw_text,
    indexed_text,
    token_count
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
			document.TenantID,
			chunkID,
			document.ID,
			version,
			chunk.Ordinal,
			document.Title,
			headingPath,
			strings.Join(headingPath, "\n"),
			chunk.RawText,
			chunk.IndexedText,
			chunk.TokenCount,
		); err != nil {
			return model.IngestResult{}, fmt.Errorf("insert chunk ordinal %d: %w", chunk.Ordinal, err)
		}
		chunkIDs = append(chunkIDs, chunkID)
	}
	if err = persistChunkEmbeddings(ctx, tx, document.TenantID, chunkIDs, chunks, false); err != nil {
		return model.IngestResult{}, fmt.Errorf("persist document version embeddings: %w", err)
	}

	commandTag, err := tx.Exec(ctx, `
UPDATE documents
SET current_version = $3, updated_at = clock_timestamp()
WHERE tenant_id = $1 AND document_id = $2 AND current_version = $4`,
		document.TenantID,
		document.ID,
		version,
		currentVersion,
	)
	if err != nil {
		return model.IngestResult{}, fmt.Errorf("activate document version %d: %w", version, err)
	}
	if commandTag.RowsAffected() != 1 {
		return model.IngestResult{}, errors.New("activate document version: current version changed while row was locked")
	}

	if err = tx.Commit(ctx); err != nil {
		return model.IngestResult{}, fmt.Errorf("commit document version %d: %w", version, err)
	}

	return model.IngestResult{
		TenantID:   document.TenantID,
		DocumentID: document.ID,
		Version:    version,
		ChunkIDs:   chunkIDs,
		CreatedAt:  createdAt,
		Unchanged:  false,
	}, nil
}

func loadUnchangedResult(
	ctx context.Context,
	tx pgx.Tx,
	tenantID string,
	documentID string,
	version int,
	fingerprint string,
) (*model.IngestResult, error) {
	var (
		storedFingerprint string
		createdAt         time.Time
	)
	if err := tx.QueryRow(ctx, `
SELECT fingerprint, created_at
FROM document_versions
WHERE tenant_id = $1 AND document_id = $2 AND version = $3`,
		tenantID,
		documentID,
		version,
	).Scan(&storedFingerprint, &createdAt); err != nil {
		return nil, fmt.Errorf("load active document version %d: %w", version, err)
	}
	if storedFingerprint != fingerprint {
		return nil, nil
	}

	rows, err := tx.Query(ctx, `
SELECT chunk_id
FROM chunks
WHERE tenant_id = $1 AND document_id = $2 AND document_version = $3
ORDER BY ordinal`,
		tenantID,
		documentID,
		version,
	)
	if err != nil {
		return nil, fmt.Errorf("load active chunk IDs: %w", err)
	}
	chunkIDs, err := pgx.CollectRows(rows, pgx.RowTo[string])
	if err != nil {
		return nil, fmt.Errorf("collect active chunk IDs: %w", err)
	}

	return &model.IngestResult{
		TenantID:   tenantID,
		DocumentID: documentID,
		Version:    version,
		ChunkIDs:   chunkIDs,
		CreatedAt:  createdAt,
		Unchanged:  true,
	}, nil
}

func persistChunkEmbeddings(
	ctx context.Context,
	tx pgx.Tx,
	tenantID string,
	chunkIDs []string,
	chunks []model.ChunkDraft,
	validateActiveChunks bool,
) error {
	if len(chunks) == 0 || chunks[0].Embedding == nil {
		return nil
	}
	if len(chunkIDs) != len(chunks) {
		return fmt.Errorf("received %d chunk IDs for %d embedding drafts", len(chunkIDs), len(chunks))
	}
	if validateActiveChunks {
		rows, err := tx.Query(ctx, `
SELECT chunk_id, ordinal, indexed_text
FROM chunks
WHERE tenant_id = $1 AND chunk_id = ANY($2::text[])
ORDER BY ordinal`, tenantID, chunkIDs)
		if err != nil {
			return fmt.Errorf("load active chunks for embedding backfill: %w", err)
		}
		type activeChunk struct {
			ID          string
			Ordinal     int
			IndexedText string
		}
		active, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (activeChunk, error) {
			var chunk activeChunk
			err := row.Scan(&chunk.ID, &chunk.Ordinal, &chunk.IndexedText)
			return chunk, err
		})
		if err != nil {
			return fmt.Errorf("collect active chunks for embedding backfill: %w", err)
		}
		if len(active) != len(chunks) {
			return fmt.Errorf("active version contains %d chunks, embedding request contains %d", len(active), len(chunks))
		}
		for index := range chunks {
			if active[index].ID != chunkIDs[index] || active[index].Ordinal != chunks[index].Ordinal || active[index].IndexedText != chunks[index].IndexedText {
				return fmt.Errorf("active chunk %d does not match embedding draft; fingerprint or chunker contract was violated", index)
			}
		}
	}

	profile := chunks[0].Embedding.Profile
	if err := ensureEmbeddingProfile(ctx, tx, profile); err != nil {
		return err
	}
	for index, chunk := range chunks {
		literal, err := vectorLiteral(chunk.Embedding.Values, storedEmbeddingDimensions)
		if err != nil {
			return fmt.Errorf("encode chunk %d embedding: %w", index, err)
		}
		inputDigest := sha256.Sum256([]byte(chunk.IndexedText))
		commandTag, err := tx.Exec(ctx, `
INSERT INTO chunk_embeddings (
    tenant_id, chunk_id, profile_id, input_sha256, embedding
)
VALUES ($1, $2, $3, $4, $5::vector(1024))
ON CONFLICT (tenant_id, chunk_id, profile_id) DO NOTHING`,
			tenantID,
			chunkIDs[index],
			profile.ProfileID,
			fmt.Sprintf("%x", inputDigest),
			literal,
		)
		if err != nil {
			return fmt.Errorf("insert chunk %d embedding: %w", index, err)
		}
		if commandTag.RowsAffected() == 0 {
			var matches bool
			if err := tx.QueryRow(ctx, `
SELECT input_sha256 = $4
   AND embedding = $5::vector(1024)
FROM chunk_embeddings
WHERE tenant_id = $1 AND chunk_id = $2 AND profile_id = $3`,
				tenantID,
				chunkIDs[index],
				profile.ProfileID,
				fmt.Sprintf("%x", inputDigest),
				literal,
			).Scan(&matches); err != nil {
				return fmt.Errorf("verify existing chunk %d embedding: %w", index, err)
			}
			if !matches {
				return fmt.Errorf("chunk %d already has a different vector or input under immutable profile %q; use a new profile ID", index, profile.ProfileID)
			}
		}
	}
	return nil
}

func ensureEmbeddingProfile(ctx context.Context, tx pgx.Tx, profile model.EmbeddingProfile) error {
	if _, err := tx.Exec(ctx, `
INSERT INTO embedding_profiles (
    profile_id, provider, model, dimensions, document_recipe, query_recipe
)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (profile_id) DO NOTHING`,
		profile.ProfileID,
		profile.Provider,
		profile.Model,
		profile.Dimensions,
		profile.DocumentRecipe,
		profile.QueryRecipe,
	); err != nil {
		return fmt.Errorf("insert embedding profile %q: %w", profile.ProfileID, err)
	}
	stored, err := loadEmbeddingProfile(ctx, tx, profile.ProfileID)
	if err != nil {
		return fmt.Errorf("load embedding profile %q: %w", profile.ProfileID, err)
	}
	if stored != profile {
		return fmt.Errorf("embedding profile %q is immutable: stored configuration differs from request", profile.ProfileID)
	}
	return nil
}

type rowQuerier interface {
	QueryRow(context.Context, string, ...any) pgx.Row
}

func loadEmbeddingProfile(ctx context.Context, querier rowQuerier, profileID string) (model.EmbeddingProfile, error) {
	var profile model.EmbeddingProfile
	err := querier.QueryRow(ctx, `
SELECT profile_id, provider, model, dimensions, document_recipe, query_recipe
FROM embedding_profiles
WHERE profile_id = $1`, profileID).Scan(
		&profile.ProfileID,
		&profile.Provider,
		&profile.Model,
		&profile.Dimensions,
		&profile.DocumentRecipe,
		&profile.QueryRecipe,
	)
	return profile, err
}

// ActiveChunkInventory returns every active source chunk for the supplied
// tenants. It intentionally ignores embedding profiles so FTS and Dense can be
// paired against the same source corpus snapshot.
func (s *Store) ActiveChunkInventory(ctx context.Context, tenantIDs []string) ([]model.ActiveChunkInventoryEntry, error) {
	if s == nil || s.pool == nil {
		return nil, errors.New("active chunk inventory: nil postgres store")
	}
	tenants := append([]string(nil), tenantIDs...)
	for index := range tenants {
		tenants[index] = strings.TrimSpace(tenants[index])
		if tenants[index] == "" {
			return nil, errors.New("active chunk inventory: tenant IDs must not be empty")
		}
	}
	slices.Sort(tenants)
	tenants = slices.Compact(tenants)
	if len(tenants) == 0 {
		return []model.ActiveChunkInventoryEntry{}, nil
	}
	rows, err := s.pool.Query(ctx, `
SELECT
    c.tenant_id,
    c.document_id,
    c.document_version,
    v.fingerprint,
    c.chunk_id,
    c.ordinal,
    c.raw_text,
    c.indexed_text
FROM chunks AS c
JOIN documents AS d
  ON d.tenant_id = c.tenant_id
 AND d.document_id = c.document_id
 AND d.current_version = c.document_version
JOIN document_versions AS v
  ON v.tenant_id = c.tenant_id
 AND v.document_id = c.document_id
 AND v.version = c.document_version
WHERE c.tenant_id = ANY($1::text[])
ORDER BY c.tenant_id, c.document_id, c.document_version, c.ordinal, c.chunk_id`, tenants)
	if err != nil {
		return nil, fmt.Errorf("active chunk inventory query: %w", err)
	}
	defer rows.Close()
	inventory := make([]model.ActiveChunkInventoryEntry, 0)
	for rows.Next() {
		var (
			entry       model.ActiveChunkInventoryEntry
			rawText     string
			indexedText string
		)
		if err := rows.Scan(
			&entry.TenantID,
			&entry.DocumentID,
			&entry.DocumentVersion,
			&entry.DocumentFingerprint,
			&entry.ChunkID,
			&entry.Ordinal,
			&rawText,
			&indexedText,
		); err != nil {
			return nil, fmt.Errorf("scan active chunk inventory: %w", err)
		}
		rawDigest := sha256.Sum256([]byte(rawText))
		indexedDigest := sha256.Sum256([]byte(indexedText))
		entry.RawTextSHA256 = fmt.Sprintf("%x", rawDigest)
		entry.IndexedTextSHA256 = fmt.Sprintf("%x", indexedDigest)
		inventory = append(inventory, entry)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate active chunk inventory: %w", err)
	}
	return inventory, nil
}

// Search runs a simple-config PostgreSQL full-text query over active document
// versions. Tenant, current-version, and ACL predicates are applied before the
// candidate limit, so unauthorized chunks can never occupy a result slot.
func (s *Store) Search(ctx context.Context, request model.SearchRequest) (model.SearchResult, error) {
	if s == nil || s.pool == nil {
		return model.SearchResult{}, errors.New("search: nil postgres store")
	}
	request.TenantID = strings.TrimSpace(request.TenantID)
	request.PrincipalID = strings.TrimSpace(request.PrincipalID)
	request.Query = strings.TrimSpace(request.Query)
	if request.TenantID == "" {
		return model.SearchResult{}, errors.New("search: tenant ID is required")
	}
	if request.Query == "" {
		return model.SearchResult{}, errors.New("search: query is required")
	}
	if request.TopK == 0 {
		request.TopK = defaultSearchTopK
	}
	if request.TopK < 1 || request.TopK > maxSearchTopK {
		return model.SearchResult{}, fmt.Errorf("search: top_k must be between 1 and %d", maxSearchTopK)
	}

	startedAt := time.Now()
	rows, err := s.pool.Query(ctx, `
WITH query AS (
    SELECT plainto_tsquery('simple'::regconfig, $2) AS value
)
SELECT
    c.chunk_id,
    c.document_id,
    c.document_version,
    v.title,
    v.source_uri,
    c.heading_path,
    c.raw_text,
    ts_rank_cd(c.search_vector, query.value)::double precision AS score,
    v.metadata
FROM chunks AS c
JOIN documents AS d
  ON d.tenant_id = c.tenant_id
 AND d.document_id = c.document_id
 AND d.current_version = c.document_version
JOIN document_versions AS v
  ON v.tenant_id = c.tenant_id
 AND v.document_id = c.document_id
 AND v.version = c.document_version
CROSS JOIN query
WHERE c.tenant_id = $1
  AND c.search_vector @@ query.value
  AND (
      cardinality(v.allowed_principals) = 0
      OR ($3 <> '' AND $3 = ANY(v.allowed_principals))
  )
ORDER BY score DESC, c.document_id, c.ordinal
LIMIT $4`,
		request.TenantID,
		request.Query,
		request.PrincipalID,
		request.TopK,
	)
	if err != nil {
		return model.SearchResult{}, fmt.Errorf("run postgres FTS search: %w", err)
	}
	defer rows.Close()

	hits := make([]model.SearchHit, 0, request.TopK)
	for rows.Next() {
		var (
			hit      model.SearchHit
			metadata []byte
		)
		if err := rows.Scan(
			&hit.ChunkID,
			&hit.DocumentID,
			&hit.DocumentVersion,
			&hit.Title,
			&hit.SourceURI,
			&hit.HeadingPath,
			&hit.Content,
			&hit.Score,
			&metadata,
		); err != nil {
			return model.SearchResult{}, fmt.Errorf("scan postgres FTS hit: %w", err)
		}
		hit.Metadata = json.RawMessage(append([]byte(nil), metadata...))
		hit.StageScores = []model.StageScore{{
			Stage: "fts",
			Rank:  len(hits) + 1,
			Score: hit.Score,
		}}
		hits = append(hits, hit)
	}
	if err := rows.Err(); err != nil {
		return model.SearchResult{}, fmt.Errorf("iterate postgres FTS hits: %w", err)
	}

	return model.SearchResult{
		Hits: hits,
		Traces: []model.StageTrace{{
			Stage:      "fts",
			DurationMS: float64(time.Since(startedAt).Microseconds()) / 1000,
		}},
	}, nil
}

// SearchDense performs deterministic exact cosine search over a materialized
// authorization-scoped set. Tenant, active-version, profile, and ACL predicates
// are resolved before distance ordering and LIMIT.
func (s *Store) SearchDense(
	ctx context.Context,
	request model.SearchRequest,
	profile model.EmbeddingProfile,
	queryVector []float32,
) (model.SearchResult, error) {
	if s == nil || s.pool == nil {
		return model.SearchResult{}, errors.New("dense search: nil postgres store")
	}
	request.TenantID = strings.TrimSpace(request.TenantID)
	request.PrincipalID = strings.TrimSpace(request.PrincipalID)
	if request.TenantID == "" {
		return model.SearchResult{}, errors.New("dense search: tenant ID is required")
	}
	if request.TopK == 0 {
		request.TopK = defaultSearchTopK
	}
	if request.TopK < 1 || request.TopK > maxSearchTopK {
		return model.SearchResult{}, fmt.Errorf("dense search: top_k must be between 1 and %d", maxSearchTopK)
	}
	if err := validateEmbeddingProfile(profile); err != nil {
		return model.SearchResult{}, fmt.Errorf("dense search: %w", err)
	}
	queryLiteral, err := vectorLiteral(queryVector, profile.Dimensions)
	if err != nil {
		return model.SearchResult{}, fmt.Errorf("dense search: encode query vector: %w", err)
	}
	startedAt := time.Now()
	storedProfile, err := loadEmbeddingProfile(ctx, s.pool, profile.ProfileID)
	if errors.Is(err, pgx.ErrNoRows) {
		return model.SearchResult{
			Hits: []model.SearchHit{},
			Traces: []model.StageTrace{{
				Stage: "dense", DurationMS: float64(time.Since(startedAt).Microseconds()) / 1000,
			}},
		}, nil
	}
	if err != nil {
		return model.SearchResult{}, fmt.Errorf("dense search: load profile: %w", err)
	}
	if storedProfile != profile {
		return model.SearchResult{}, fmt.Errorf("dense search: profile %q configuration differs from stored immutable profile", profile.ProfileID)
	}

	rows, err := s.pool.Query(ctx, `
WITH authorized AS MATERIALIZED (
    SELECT
        c.chunk_id,
        c.document_id,
        c.document_version,
        c.ordinal,
        v.title,
        v.source_uri,
        c.heading_path,
        c.raw_text,
        v.metadata,
        e.embedding
    FROM chunk_embeddings AS e
    JOIN chunks AS c
      ON c.tenant_id = e.tenant_id
     AND c.chunk_id = e.chunk_id
    JOIN documents AS d
      ON d.tenant_id = c.tenant_id
     AND d.document_id = c.document_id
     AND d.current_version = c.document_version
    JOIN document_versions AS v
      ON v.tenant_id = c.tenant_id
     AND v.document_id = c.document_id
     AND v.version = c.document_version
    WHERE e.tenant_id = $1
      AND e.profile_id = $3
      AND (
          cardinality(v.allowed_principals) = 0
          OR ($2 <> '' AND $2 = ANY(v.allowed_principals))
      )
)
SELECT
    chunk_id,
    document_id,
    document_version,
    title,
    source_uri,
    heading_path,
    raw_text,
    (1 - (embedding <=> $4::vector(1024)))::double precision AS score,
    metadata
FROM authorized
ORDER BY embedding <=> $4::vector(1024), document_id, ordinal
LIMIT $5`,
		request.TenantID,
		request.PrincipalID,
		profile.ProfileID,
		queryLiteral,
		request.TopK,
	)
	if err != nil {
		return model.SearchResult{}, fmt.Errorf("run postgres exact dense search: %w", err)
	}
	defer rows.Close()

	hits := make([]model.SearchHit, 0, request.TopK)
	for rows.Next() {
		var (
			hit      model.SearchHit
			metadata []byte
		)
		if err := rows.Scan(
			&hit.ChunkID,
			&hit.DocumentID,
			&hit.DocumentVersion,
			&hit.Title,
			&hit.SourceURI,
			&hit.HeadingPath,
			&hit.Content,
			&hit.Score,
			&metadata,
		); err != nil {
			return model.SearchResult{}, fmt.Errorf("scan postgres dense hit: %w", err)
		}
		hit.Metadata = json.RawMessage(append([]byte(nil), metadata...))
		hit.StageScores = []model.StageScore{{Stage: "dense", Rank: len(hits) + 1, Score: hit.Score}}
		hits = append(hits, hit)
	}
	if err := rows.Err(); err != nil {
		return model.SearchResult{}, fmt.Errorf("iterate postgres dense hits: %w", err)
	}
	return model.SearchResult{
		Hits: hits,
		Traces: []model.StageTrace{{
			Stage: "dense", DurationMS: float64(time.Since(startedAt).Microseconds()) / 1000,
		}},
	}, nil
}

func validateEmbeddingProfile(profile model.EmbeddingProfile) error {
	if strings.TrimSpace(profile.ProfileID) == "" || len(profile.ProfileID) > 128 {
		return errors.New("embedding profile ID must be between 1 and 128 bytes")
	}
	if strings.TrimSpace(profile.Provider) == "" || len(profile.Provider) > 128 {
		return errors.New("embedding provider must be between 1 and 128 bytes")
	}
	if strings.TrimSpace(profile.Model) == "" || len(profile.Model) > 256 {
		return errors.New("embedding model must be between 1 and 256 bytes")
	}
	if profile.Dimensions != storedEmbeddingDimensions {
		return fmt.Errorf("embedding dimensions must equal stored vector size %d", storedEmbeddingDimensions)
	}
	if strings.TrimSpace(profile.DocumentRecipe) == "" || len(profile.DocumentRecipe) > 256 {
		return errors.New("embedding document recipe must be between 1 and 256 bytes")
	}
	if strings.TrimSpace(profile.QueryRecipe) == "" || len(profile.QueryRecipe) > 256 {
		return errors.New("embedding query recipe must be between 1 and 256 bytes")
	}
	return nil
}

func vectorLiteral(values []float32, dimensions int) (string, error) {
	if dimensions != storedEmbeddingDimensions || len(values) != dimensions {
		return "", fmt.Errorf("vector has %d dimensions, want %d", len(values), storedEmbeddingDimensions)
	}
	var normSquared float64
	buffer := make([]byte, 0, len(values)*12)
	buffer = append(buffer, '[')
	for index, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return "", fmt.Errorf("dimension %d is not finite", index)
		}
		if index > 0 {
			buffer = append(buffer, ',')
		}
		buffer = strconv.AppendFloat(buffer, float64(value), 'g', -1, 32)
		normSquared += float64(value) * float64(value)
	}
	if normSquared == 0 {
		return "", errors.New("cosine vector must not be zero")
	}
	buffer = append(buffer, ']')
	return string(buffer), nil
}

func validateSaveInput(s *Store, document model.DocumentInput, fingerprint string, chunks []model.ChunkDraft) error {
	if s == nil || s.pool == nil {
		return errors.New("save document version: nil postgres store")
	}
	if strings.TrimSpace(document.TenantID) == "" {
		return errors.New("save document version: tenant ID is required")
	}
	if strings.TrimSpace(document.ID) == "" {
		return errors.New("save document version: document ID is required")
	}
	if strings.TrimSpace(document.Content) == "" {
		return errors.New("save document version: raw document content is required")
	}
	if strings.TrimSpace(fingerprint) == "" {
		return errors.New("save document version: fingerprint is required")
	}
	if len(document.Metadata) == 0 || !json.Valid(document.Metadata) {
		return errors.New("save document version: metadata must be valid JSON")
	}
	if len(chunks) == 0 {
		return errors.New("save document version: at least one chunk is required")
	}
	var embeddingProfile *model.EmbeddingProfile
	for index, chunk := range chunks {
		if chunk.Ordinal != index {
			return fmt.Errorf("save document version: chunk ordinal %d at index %d is not contiguous", chunk.Ordinal, index)
		}
		if strings.TrimSpace(chunk.RawText) == "" {
			return fmt.Errorf("save document version: chunk ordinal %d has empty raw text", chunk.Ordinal)
		}
		if strings.TrimSpace(chunk.IndexedText) == "" {
			return fmt.Errorf("save document version: chunk ordinal %d has empty indexed text", chunk.Ordinal)
		}
		if chunk.TokenCount < 0 {
			return fmt.Errorf("save document version: chunk ordinal %d has negative token count", chunk.Ordinal)
		}
		if chunk.Embedding == nil {
			if embeddingProfile != nil {
				return errors.New("save document version: embedding drafts must be present for every chunk or none")
			}
			continue
		}
		if index > 0 && chunks[0].Embedding == nil {
			return errors.New("save document version: embedding drafts must be present for every chunk or none")
		}
		if err := validateEmbeddingProfile(chunk.Embedding.Profile); err != nil {
			return fmt.Errorf("save document version: chunk ordinal %d: %w", chunk.Ordinal, err)
		}
		if embeddingProfile == nil {
			profile := chunk.Embedding.Profile
			embeddingProfile = &profile
		} else if *embeddingProfile != chunk.Embedding.Profile {
			return errors.New("save document version: every chunk embedding must use the same immutable profile")
		}
		if _, err := vectorLiteral(chunk.Embedding.Values, chunk.Embedding.Profile.Dimensions); err != nil {
			return fmt.Errorf("save document version: chunk ordinal %d embedding: %w", chunk.Ordinal, err)
		}
	}
	return nil
}

func versionedChunkID(documentID string, version, ordinal int) string {
	return fmt.Sprintf("%s:v%06d:c%04d", documentID, version, ordinal)
}
