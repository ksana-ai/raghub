// Package postgres implements raghub's versioned ingestion store and
// authorization-scoped full-text retriever on PostgreSQL.
package postgres

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"raghub/internal/model"
)

const (
	defaultSearchTopK = 5
	maxSearchTopK     = 50
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
	}
	return nil
}

func versionedChunkID(documentID string, version, ordinal int) string {
	return fmt.Sprintf("%s:v%06d:c%04d", documentID, version, ordinal)
}
