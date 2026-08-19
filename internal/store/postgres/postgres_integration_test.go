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
		Embedding: &model.EmbeddingDraft{
			Profile: integrationEmbeddingProfile("concurrent-" + tenantID),
			Values:  integrationAxisVector(0),
		},
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

func TestPostgresExactDenseBackfillVersioningAndAuthorization(t *testing.T) {
	store, tenantA, tenantB := integrationStore(t)
	ctx := context.Background()
	profile := integrationEmbeddingProfile("dense-" + tenantA)

	document := model.DocumentInput{
		TenantID: tenantA, ID: "dense-public-target", Title: "Dense target",
		SourceURI: "https://example.test/dense-target", Content: "Dense target source v1.",
		AllowedPrincipals: []string{}, Metadata: []byte(`{"kind":"dense-target"}`),
	}
	v1Chunk := plainIntegrationChunk("The first dense target version uses axis one.")
	v1, err := store.SaveDocumentVersion(ctx, document, "dense-target-v1", []model.ChunkDraft{v1Chunk})
	if err != nil {
		t.Fatalf("save dense target v1 without embedding: %v", err)
	}
	if v1.Version != 1 || v1.Unchanged {
		t.Fatalf("unexpected dense v1: %+v", v1)
	}

	// A Dense run must backfill a corpus previously ingested by FTS without
	// inventing a new source-document version.
	backfillChunk := v1Chunk
	backfillChunk.Embedding = &model.EmbeddingDraft{Profile: profile, Values: integrationAxisVector(0)}
	backfilled, err := store.SaveDocumentVersion(ctx, document, "dense-target-v1", []model.ChunkDraft{backfillChunk})
	if err != nil {
		t.Fatalf("backfill active dense target: %v", err)
	}
	if !backfilled.Unchanged || backfilled.Version != 1 || backfilled.ChunkIDs[0] != v1.ChunkIDs[0] {
		t.Fatalf("dense backfill changed source version: first=%+v backfill=%+v", v1, backfilled)
	}
	var backfillCount int
	if err := store.pool.QueryRow(ctx, `
SELECT count(*)
FROM chunk_embeddings
WHERE tenant_id = $1 AND chunk_id = $2 AND profile_id = $3`,
		tenantA, v1.ChunkIDs[0], profile.ProfileID,
	).Scan(&backfillCount); err != nil {
		t.Fatalf("count backfilled embeddings: %v", err)
	}
	if backfillCount != 1 {
		t.Fatalf("backfilled embedding count = %d, want 1", backfillCount)
	}
	repeatedBackfill, err := store.SaveDocumentVersion(ctx, document, "dense-target-v1", []model.ChunkDraft{backfillChunk})
	if err != nil {
		t.Fatalf("repeat unchanged dense backfill: %v", err)
	}
	if !repeatedBackfill.Unchanged || repeatedBackfill.Version != 1 {
		t.Fatalf("repeat dense backfill was not idempotent: %+v", repeatedBackfill)
	}
	if _, err := store.pool.Exec(ctx, `
UPDATE embedding_profiles SET model = model WHERE profile_id = $1`, profile.ProfileID); err == nil || !strings.Contains(err.Error(), "embedding profiles are immutable") {
		t.Fatalf("direct embedding profile update error = %v, want immutable trigger rejection", err)
	}
	if _, err := store.pool.Exec(ctx, `
UPDATE chunk_embeddings SET input_sha256 = input_sha256
WHERE tenant_id = $1 AND chunk_id = $2 AND profile_id = $3`, tenantA, v1.ChunkIDs[0], profile.ProfileID); err == nil || !strings.Contains(err.Error(), "chunk embeddings are immutable") {
		t.Fatalf("direct chunk embedding update error = %v, want immutable trigger rejection", err)
	}

	// A new source version writes its complete vector set in the same database
	// transaction and only then advances current_version.
	document.Content = "Dense target source v2."
	v2Chunk := plainIntegrationChunk("The current dense target version uses axis two.")
	v2Chunk.Embedding = &model.EmbeddingDraft{Profile: profile, Values: integrationAxisVector(1)}
	v2, err := store.SaveDocumentVersion(ctx, document, "dense-target-v2", []model.ChunkDraft{v2Chunk})
	if err != nil {
		t.Fatalf("save dense target v2: %v", err)
	}
	if v2.Version != 2 || v2.Unchanged {
		t.Fatalf("unexpected dense v2: %+v", v2)
	}
	oldAxisResult, err := store.SearchDense(ctx, model.SearchRequest{TenantID: tenantA, TopK: 1}, profile, integrationAxisVector(0))
	if err != nil {
		t.Fatalf("search old dense axis: %v", err)
	}
	if len(oldAxisResult.Hits) != 1 || oldAxisResult.Hits[0].ChunkID != v2.ChunkIDs[0] || oldAxisResult.Hits[0].DocumentVersion != 2 {
		t.Fatalf("old dense version was not filtered by active pointer: %+v", oldAxisResult.Hits)
	}
	if oldAxisResult.Hits[0].Content != v2Chunk.RawText || len(oldAxisResult.Hits[0].StageScores) != 1 || oldAxisResult.Hits[0].StageScores[0].Stage != "dense" {
		t.Fatalf("dense hit lacks raw citation or stage evidence: %+v", oldAxisResult.Hits[0])
	}

	// A changed configuration under the same profile ID is rejected after the
	// transaction begins. The attempted document v3 must roll back completely.
	driftedProfile := profile
	driftedProfile.Model = "different-model-under-same-profile"
	driftedChunk := plainIntegrationChunk("This transaction must roll back.")
	driftedChunk.Embedding = &model.EmbeddingDraft{Profile: driftedProfile, Values: integrationAxisVector(2)}
	document.Content = "Dense target source v3 that must not activate."
	if _, err := store.SaveDocumentVersion(ctx, document, "dense-target-v3", []model.ChunkDraft{driftedChunk}); err == nil {
		t.Fatal("profile drift unexpectedly succeeded")
	}
	var currentVersion, versionCount int
	if err := store.pool.QueryRow(ctx, `
SELECT d.current_version,
       (SELECT count(*) FROM document_versions v WHERE v.tenant_id = d.tenant_id AND v.document_id = d.document_id)
FROM documents d
WHERE d.tenant_id = $1 AND d.document_id = $2`, tenantA, document.ID).Scan(&currentVersion, &versionCount); err != nil {
		t.Fatalf("inspect rollback state: %v", err)
	}
	if currentVersion != 2 || versionCount != 2 {
		t.Fatalf("failed embedding transaction changed versions: current=%d count=%d", currentVersion, versionCount)
	}

	// Insert more-nearby vectors that the caller cannot see. Exact dense search
	// must still return the farther authorized result rather than underfilling.
	for index := range 6 {
		privateDocument := model.DocumentInput{
			TenantID: tenantA, ID: fmt.Sprintf("dense-private-%d", index), Title: "Private dense decoy",
			SourceURI: fmt.Sprintf("https://example.test/private/%d", index), Content: "Private dense source.",
			AllowedPrincipals: []string{"alice"}, Metadata: []byte(`{}`),
		}
		chunk := plainIntegrationChunk(fmt.Sprintf("Private exact-axis decoy %d.", index))
		chunk.Embedding = &model.EmbeddingDraft{Profile: profile, Values: integrationAxisVector(0)}
		if _, err := store.SaveDocumentVersion(ctx, privateDocument, fmt.Sprintf("dense-private-%d", index), []model.ChunkDraft{chunk}); err != nil {
			t.Fatalf("save private dense decoy %d: %v", index, err)
		}
	}
	otherTenantDocument := model.DocumentInput{
		TenantID: tenantB, ID: "dense-other-tenant", Title: "Other tenant dense decoy",
		SourceURI: "https://example.test/other-dense", Content: "Other tenant dense source.",
		AllowedPrincipals: []string{}, Metadata: []byte(`{}`),
	}
	otherChunk := plainIntegrationChunk("Other tenant exact-axis decoy.")
	otherChunk.Embedding = &model.EmbeddingDraft{Profile: profile, Values: integrationAxisVector(0)}
	otherResult, err := store.SaveDocumentVersion(ctx, otherTenantDocument, "dense-other-tenant", []model.ChunkDraft{otherChunk})
	if err != nil {
		t.Fatalf("save other tenant dense decoy: %v", err)
	}

	denied, err := store.SearchDense(ctx, model.SearchRequest{TenantID: tenantA, PrincipalID: "bob", TopK: 1}, profile, integrationAxisVector(0))
	if err != nil {
		t.Fatalf("dense search as denied principal: %v", err)
	}
	if len(denied.Hits) != 1 || denied.Hits[0].ChunkID != v2.ChunkIDs[0] {
		t.Fatalf("unauthorized closer vectors displaced authorized result: %+v", denied.Hits)
	}
	allowed, err := store.SearchDense(ctx, model.SearchRequest{TenantID: tenantA, PrincipalID: "alice", TopK: 1}, profile, integrationAxisVector(0))
	if err != nil {
		t.Fatalf("dense search as allowed principal: %v", err)
	}
	if len(allowed.Hits) != 1 || !strings.HasPrefix(allowed.Hits[0].DocumentID, "dense-private-") {
		t.Fatalf("allowed principal did not receive nearer private result: %+v", allowed.Hits)
	}
	otherTenant, err := store.SearchDense(ctx, model.SearchRequest{TenantID: tenantB, TopK: 1}, profile, integrationAxisVector(0))
	if err != nil {
		t.Fatalf("dense search other tenant: %v", err)
	}
	if len(otherTenant.Hits) != 1 || otherTenant.Hits[0].ChunkID != otherResult.ChunkIDs[0] {
		t.Fatalf("tenant isolation returned wrong result: %+v", otherTenant.Hits)
	}

	missingProfile := profile
	missingProfile.ProfileID += "-missing"
	missing, err := store.SearchDense(ctx, model.SearchRequest{TenantID: tenantA, TopK: 5}, missingProfile, integrationAxisVector(0))
	if err != nil {
		t.Fatalf("search missing profile: %v", err)
	}
	if len(missing.Hits) != 0 {
		t.Fatalf("missing profile returned hits: %+v", missing.Hits)
	}

	inventory, err := store.ActiveChunkInventory(ctx, []string{tenantA, tenantA})
	if err != nil {
		t.Fatalf("load active tenant inventory: %v", err)
	}
	if len(inventory) != 7 {
		t.Fatalf("active tenant inventory length = %d, want 7", len(inventory))
	}
	var foundCurrentTarget bool
	for _, entry := range inventory {
		if entry.TenantID != tenantA || len(entry.RawTextSHA256) != 64 || len(entry.IndexedTextSHA256) != 64 {
			t.Fatalf("invalid active inventory entry: %+v", entry)
		}
		if entry.DocumentID == document.ID {
			foundCurrentTarget = entry.DocumentVersion == 2 && entry.ChunkID == v2.ChunkIDs[0]
		}
	}
	if !foundCurrentTarget {
		t.Fatalf("active inventory did not retain only target v2: %+v", inventory)
	}
}

func plainIntegrationChunk(text string) model.ChunkDraft {
	return model.ChunkDraft{Ordinal: 0, RawText: text, IndexedText: text, TokenCount: len(strings.Fields(text))}
}

func integrationEmbeddingProfile(profileID string) model.EmbeddingProfile {
	return model.EmbeddingProfile{
		ProfileID: profileID, Provider: "integration", Model: "integration-model", Dimensions: 1024,
		DocumentRecipe: "indexed_text/v1", QueryRecipe: "raw_query/v1",
	}
}

func integrationAxisVector(axis int) []float32 {
	vector := make([]float32, 1024)
	vector[axis] = 1
	return vector
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
