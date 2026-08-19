package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"raghub/internal/model"
)

func TestPostgresVersioningAndAuthorization(t *testing.T) {
	store, tenantA, tenantB := integrationStore(t)
	ctx := context.Background()

	document := model.DocumentInput{
		TenantID:          tenantA,
		ID:                "versioned-document",
		Title:             "Versioned guide",
		SourceURI:         "https://example.test/versioned",
		Content:           "# Versioned guide\n\nOriginal source content retained for audit.",
		AllowedPrincipals: []string{},
		Metadata:          []byte(`{"source":"integration"}`),
	}
	v1, err := store.SaveDocumentVersion(ctx, document, "fingerprint-v1", []model.ChunkDraft{{
		Ordinal:     0,
		HeadingPath: []string{"Old heading"},
		RawText:     "The obsoleteword belongs only to the first version.",
		IndexedText: "The obsoleteword belongs only to the first version.",
		TokenCount:  8,
	}})
	if err != nil {
		t.Fatalf("save v1: %v", err)
	}
	if v1.Version != 1 || v1.Unchanged || len(v1.ChunkIDs) != 1 || v1.ChunkIDs[0] != "versioned-document:v000001:c0000" {
		t.Fatalf("unexpected v1 result: %+v", v1)
	}
	var storedRawContent string
	if err := store.pool.QueryRow(ctx, `
SELECT raw_content
FROM document_versions
WHERE tenant_id = $1 AND document_id = $2 AND version = $3`,
		tenantA, document.ID, v1.Version,
	).Scan(&storedRawContent); err != nil {
		t.Fatalf("load stored raw document: %v", err)
	}
	if storedRawContent != document.Content {
		t.Fatalf("stored raw document = %q, want %q", storedRawContent, document.Content)
	}

	unchanged, err := store.SaveDocumentVersion(ctx, document, "fingerprint-v1", []model.ChunkDraft{{
		Ordinal:     0,
		HeadingPath: []string{"Ignored on idempotent retry"},
		RawText:     "Different drafts do not matter when the current fingerprint is identical.",
		IndexedText: "Different drafts do not matter when the current fingerprint is identical.",
		TokenCount:  10,
	}})
	if err != nil {
		t.Fatalf("save unchanged v1: %v", err)
	}
	if !unchanged.Unchanged || unchanged.Version != 1 || unchanged.CreatedAt != v1.CreatedAt || unchanged.ChunkIDs[0] != v1.ChunkIDs[0] {
		t.Fatalf("idempotent result did not preserve v1: first=%+v retry=%+v", v1, unchanged)
	}

	document.Title = "Current guide"
	v2, err := store.SaveDocumentVersion(ctx, document, "fingerprint-v2", []model.ChunkDraft{{
		Ordinal:     0,
		HeadingPath: []string{"Current heading"},
		RawText:     "This exact raw citation is returned for the current version.",
		IndexedText: "This exact raw citation is returned for the current version contextualtoken.",
		TokenCount:  10,
	}})
	if err != nil {
		t.Fatalf("save v2: %v", err)
	}
	if v2.Version != 2 || v2.Unchanged || v2.ChunkIDs[0] == v1.ChunkIDs[0] || v2.ChunkIDs[0] != "versioned-document:v000002:c0000" {
		t.Fatalf("unexpected v2 result: %+v", v2)
	}

	oldResult, err := store.Search(ctx, model.SearchRequest{TenantID: tenantA, Query: "obsoleteword", TopK: 10})
	if err != nil {
		t.Fatalf("search old version: %v", err)
	}
	if len(oldResult.Hits) != 0 {
		t.Fatalf("old version leaked into current search: %+v", oldResult.Hits)
	}

	currentResult, err := store.Search(ctx, model.SearchRequest{TenantID: tenantA, Query: "contextualtoken", TopK: 10})
	if err != nil {
		t.Fatalf("search current version: %v", err)
	}
	if len(currentResult.Hits) != 1 {
		t.Fatalf("want one current-version hit, got %+v", currentResult.Hits)
	}
	hit := currentResult.Hits[0]
	if hit.ChunkID != v2.ChunkIDs[0] || hit.DocumentVersion != 2 || hit.Content != "This exact raw citation is returned for the current version." {
		t.Fatalf("search did not return the exact current raw citation: %+v", hit)
	}
	if hit.Title != "Current guide" || hit.SourceURI != document.SourceURI || len(hit.HeadingPath) != 1 || hit.HeadingPath[0] != "Current heading" {
		t.Fatalf("search citation metadata is incomplete: %+v", hit)
	}
	var metadata map[string]string
	if err := json.Unmarshal(hit.Metadata, &metadata); err != nil || metadata["source"] != "integration" {
		t.Fatalf("search citation JSON metadata = %s, error = %v", hit.Metadata, err)
	}
	if len(hit.StageScores) != 1 || hit.StageScores[0].Stage != "fts" || hit.StageScores[0].Rank != 1 {
		t.Fatalf("missing FTS stage scores: %+v", hit.StageScores)
	}
	if len(currentResult.Traces) != 1 || currentResult.Traces[0].Stage != "fts" {
		t.Fatalf("missing FTS trace: %+v", currentResult.Traces)
	}

	_, err = store.SaveDocumentVersion(ctx, model.DocumentInput{
		TenantID:          tenantB,
		ID:                "other-tenant-document",
		Title:             "Other tenant",
		SourceURI:         "https://example.test/other-tenant",
		Content:           "# Isolation\n\ntenantisolated is visible only inside tenant B.",
		AllowedPrincipals: []string{},
		Metadata:          []byte(`{}`),
	}, "tenant-b-fingerprint", []model.ChunkDraft{{
		Ordinal:     0,
		HeadingPath: []string{"Isolation"},
		RawText:     "tenantisolated is visible only inside tenant B.",
		IndexedText: "tenantisolated is visible only inside tenant B.",
		TokenCount:  7,
	}})
	if err != nil {
		t.Fatalf("save tenant B document: %v", err)
	}

	crossTenant, err := store.Search(ctx, model.SearchRequest{TenantID: tenantA, Query: "tenantisolated", TopK: 10})
	if err != nil {
		t.Fatalf("search tenant A for tenant B term: %v", err)
	}
	if len(crossTenant.Hits) != 0 {
		t.Fatalf("cross-tenant search leaked hits: %+v", crossTenant.Hits)
	}

	_, err = store.SaveDocumentVersion(ctx, model.DocumentInput{
		TenantID:          tenantA,
		ID:                "private-document",
		Title:             "Private handbook",
		SourceURI:         "https://example.test/private",
		Content:           "# Confidential\n\naclsecret is available to Alice.",
		AllowedPrincipals: []string{"alice"},
		Metadata:          []byte(`{"visibility":"private"}`),
	}, "private-fingerprint", []model.ChunkDraft{{
		Ordinal:     0,
		HeadingPath: []string{"Confidential"},
		RawText:     "aclsecret is available to Alice.",
		IndexedText: "aclsecret is available to Alice.",
		TokenCount:  6,
	}})
	if err != nil {
		t.Fatalf("save ACL document: %v", err)
	}

	for _, principal := range []string{"", "bob"} {
		unauthorized, searchErr := store.Search(ctx, model.SearchRequest{
			TenantID: tenantA, PrincipalID: principal, Query: "aclsecret", TopK: 10,
		})
		if searchErr != nil {
			t.Fatalf("search ACL document as %q: %v", principal, searchErr)
		}
		if len(unauthorized.Hits) != 0 {
			t.Fatalf("principal %q received unauthorized hits: %+v", principal, unauthorized.Hits)
		}
	}

	authorized, err := store.Search(ctx, model.SearchRequest{
		TenantID: tenantA, PrincipalID: "alice", Query: "aclsecret", TopK: 10,
	})
	if err != nil {
		t.Fatalf("search ACL document as alice: %v", err)
	}
	if len(authorized.Hits) != 1 || authorized.Hits[0].DocumentID != "private-document" {
		t.Fatalf("authorized principal did not receive private hit: %+v", authorized.Hits)
	}
}

func TestPostgresConcurrentIdempotentSave(t *testing.T) {
	store, tenantID, _ := integrationStore(t)
	ctx := context.Background()
	document := model.DocumentInput{
		TenantID:          tenantID,
		ID:                "concurrent-document",
		Title:             "Concurrent",
		SourceURI:         "https://example.test/concurrent",
		Content:           "# Concurrent\n\nconcurrencysafe body",
		AllowedPrincipals: []string{},
		Metadata:          []byte(`{}`),
	}
	chunks := []model.ChunkDraft{{
		Ordinal: 0, RawText: "concurrencysafe body", IndexedText: "concurrencysafe body", TokenCount: 2,
	}}

	const workers = 8
	results := make(chan model.IngestResult, workers)
	errors := make(chan error, workers)
	var group sync.WaitGroup
	for range workers {
		group.Add(1)
		go func() {
			defer group.Done()
			result, err := store.SaveDocumentVersion(ctx, document, "same-concurrent-fingerprint", chunks)
			if err != nil {
				errors <- err
				return
			}
			results <- result
		}()
	}
	group.Wait()
	close(results)
	close(errors)
	for err := range errors {
		t.Errorf("concurrent save: %v", err)
	}

	created := 0
	seen := 0
	for result := range results {
		seen++
		if result.Version != 1 {
			t.Errorf("concurrent idempotent save created version %d", result.Version)
		}
		if !result.Unchanged {
			created++
		}
	}
	if seen != workers || created != 1 {
		t.Fatalf("want %d results and exactly one creator, got results=%d creators=%d", workers, seen, created)
	}

	// Distinct concurrent fingerprints must serialize into distinct immutable
	// versions instead of racing on the active version pointer.
	distinctResults := make(chan model.IngestResult, 2)
	distinctErrors := make(chan error, 2)
	group = sync.WaitGroup{}
	for index := range 2 {
		group.Add(1)
		go func() {
			defer group.Done()
			word := fmt.Sprintf("distinctversion%d", index)
			result, err := store.SaveDocumentVersion(ctx, document, "distinct-fingerprint-"+word, []model.ChunkDraft{{
				Ordinal: 0, RawText: word, IndexedText: word, TokenCount: 1,
			}})
			if err != nil {
				distinctErrors <- err
				return
			}
			distinctResults <- result
		}()
	}
	group.Wait()
	close(distinctResults)
	close(distinctErrors)
	for err := range distinctErrors {
		t.Errorf("concurrent distinct save: %v", err)
	}
	versions := map[int]bool{}
	for result := range distinctResults {
		if result.Unchanged {
			t.Errorf("distinct fingerprint unexpectedly reported unchanged: %+v", result)
		}
		versions[result.Version] = true
	}
	if !versions[2] || !versions[3] || len(versions) != 2 {
		t.Fatalf("distinct concurrent fingerprints created versions %v, want versions 2 and 3", versions)
	}
}

func integrationStore(t *testing.T) (*Store, string, string) {
	t.Helper()
	databaseURL := strings.TrimSpace(os.Getenv("RAGHUB_TEST_DATABASE_URL"))
	if databaseURL == "" {
		t.Skip("RAGHUB_TEST_DATABASE_URL is not set; skipping PostgreSQL integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatalf("open PostgreSQL pool: %v", err)
	}
	t.Cleanup(pool.Close)
	if err := pool.Ping(ctx); err != nil {
		t.Fatalf("ping PostgreSQL: %v", err)
	}
	if err := ApplyMigrations(ctx, pool); err != nil {
		t.Fatalf("apply PostgreSQL migrations: %v", err)
	}

	suffix := fmt.Sprintf("%d", time.Now().UnixNano())
	tenantA := "postgres-test-a-" + suffix
	tenantB := "postgres-test-b-" + suffix
	t.Cleanup(func() {
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cleanupCancel()
		if _, cleanupErr := pool.Exec(cleanupCtx,
			`DELETE FROM documents WHERE tenant_id = ANY($1::text[])`,
			[]string{tenantA, tenantB},
		); cleanupErr != nil {
			t.Logf("cleanup PostgreSQL integration rows: %v", cleanupErr)
		}
	})
	return New(pool), tenantA, tenantB
}
