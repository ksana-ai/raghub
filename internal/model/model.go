package model

import (
	"encoding/json"
	"time"
)

// DocumentInput is the versioned source material accepted by the ingestion
// service. TenantID is supplied by the trusted application boundary, not by a
// document payload.
type DocumentInput struct {
	TenantID          string
	ID                string
	Title             string
	SourceURI         string
	Content           string
	AllowedPrincipals []string
	Metadata          json.RawMessage
}

// ChunkDraft is a deterministic, storage-independent chunk produced from one
// document version. The store assigns its immutable versioned ID.
type ChunkDraft struct {
	Ordinal     int
	HeadingPath []string
	RawText     string
	IndexedText string
	TokenCount  int
}

// IngestResult identifies the immutable version and chunks made active by an
// ingestion request.
type IngestResult struct {
	TenantID   string    `json:"tenant_id"`
	DocumentID string    `json:"document_id"`
	Version    int       `json:"version"`
	ChunkIDs   []string  `json:"chunk_ids"`
	CreatedAt  time.Time `json:"created_at"`
	Unchanged  bool      `json:"unchanged"`
}

// SearchRequest contains the authorization scope and retrieval parameters.
// PrincipalID may be empty; in that case only tenant-wide chunks are visible.
type SearchRequest struct {
	TenantID    string
	PrincipalID string
	Query       string
	TopK        int
}

// SearchHit is a ranked chunk plus the information required to cite the exact
// document version from which it came.
type SearchHit struct {
	ChunkID         string          `json:"chunk_id"`
	DocumentID      string          `json:"document_id"`
	DocumentVersion int             `json:"document_version"`
	Title           string          `json:"title"`
	SourceURI       string          `json:"source_uri"`
	HeadingPath     []string        `json:"heading_path"`
	Content         string          `json:"content"`
	Score           float64         `json:"score"`
	StageScores     []StageScore    `json:"stage_scores"`
	Metadata        json.RawMessage `json:"metadata"`
}

// StageScore preserves the rank and score assigned by each retrieval stage.
// It makes later FTS/dense/RRF/rerank ablations inspectable without changing
// the public hit contract.
type StageScore struct {
	Stage string  `json:"stage"`
	Rank  int     `json:"rank"`
	Score float64 `json:"score"`
}

type StageTrace struct {
	Stage      string  `json:"stage"`
	DurationMS float64 `json:"duration_ms"`
}

type SearchResult struct {
	Hits   []SearchHit  `json:"hits"`
	Traces []StageTrace `json:"traces"`
}
