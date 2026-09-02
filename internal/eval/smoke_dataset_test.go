package eval

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/ksana-ai/raghub/internal/ingest"
)

func TestSmokeDatasetContract(t *testing.T) {
	t.Parallel()

	loaded, err := LoadDataset(filepath.Join("..", "..", "datasets", "smoke", "v1.json"))
	if err != nil {
		t.Fatalf("LoadDataset() error = %v", err)
	}
	if loaded.Dataset.Name != "raghub-smoke" || loaded.Dataset.Version != "1.0.0" {
		t.Fatalf("unexpected dataset identity: %s@%s", loaded.Dataset.Name, loaded.Dataset.Version)
	}
	if len(loaded.Dataset.Queries) != 8 {
		t.Fatalf("query count = %d, want 8", len(loaded.Dataset.Queries))
	}

	var aclCases, tenantBoundaryCases int
	allowedTenants := map[string]bool{"eval-smoke-v1-acme": true, "eval-smoke-v1-globex": true}
	for _, document := range loaded.Dataset.Documents {
		if !allowedTenants[document.TenantID] {
			t.Fatalf("unexpected smoke document tenant %q", document.TenantID)
		}
	}
	for _, query := range loaded.Dataset.Queries {
		if !allowedTenants[query.TenantID] {
			t.Fatalf("unexpected smoke query tenant %q", query.TenantID)
		}
		switch query.Category {
		case "acl":
			aclCases++
		case "tenant-boundary":
			tenantBoundaryCases++
		}
	}
	if aclCases != 2 || tenantBoundaryCases != 2 {
		t.Fatalf("ACL/tenant cases = %d/%d, want 2/2", aclCases, tenantBoundaryCases)
	}
}

func TestPairedSmokeDatasetV2Contract(t *testing.T) {
	t.Parallel()

	loaded, err := LoadDataset(filepath.Join("..", "..", "datasets", "smoke", "v2.json"))
	if err != nil {
		t.Fatalf("LoadDataset(v2) error = %v", err)
	}
	if loaded.Dataset.Name != "raghub-smoke-paired" || loaded.Dataset.Version != "2.0.0" {
		t.Fatalf("unexpected v2 identity: %s@%s", loaded.Dataset.Name, loaded.Dataset.Version)
	}
	if len(loaded.Dataset.Queries) != 8 {
		t.Fatalf("v2 query count = %d, want 8", len(loaded.Dataset.Queries))
	}

	allowedTenants := map[string]bool{"eval-smoke-v2-acme": true, "eval-smoke-v2-globex": true}
	for _, document := range loaded.Dataset.Documents {
		if !allowedTenants[document.TenantID] || !strings.HasPrefix(document.DocumentID, "v2-") {
			t.Fatalf("v2 document is not isolated: tenant=%q id=%q", document.TenantID, document.DocumentID)
		}
	}
	wantCategories := map[string]int{
		"exact-lexical":       1,
		"semantic-paraphrase": 2,
		"cross-language":      1,
		"acl":                 2,
		"tenant-boundary":     2,
	}
	gotCategories := make(map[string]int)
	for _, query := range loaded.Dataset.Queries {
		if !allowedTenants[query.TenantID] {
			t.Fatalf("v2 query tenant %q is not isolated", query.TenantID)
		}
		gotCategories[query.Category]++
		for _, chunkID := range append(append([]string(nil), query.GoldChunkIDs...), query.ForbiddenChunkIDs...) {
			if !strings.HasPrefix(chunkID, "v2-") || !strings.Contains(chunkID, ":v000001:c") {
				t.Fatalf("v2 query %q has invalid chunk reference %q", query.ID, chunkID)
			}
		}
	}
	for category, want := range wantCategories {
		if gotCategories[category] != want {
			t.Fatalf("v2 category %q count = %d, want %d", category, gotCategories[category], want)
		}
	}
}

func TestHybridSmokeDatasetV3PreregisteredContract(t *testing.T) {
	t.Parallel()

	loaded, err := LoadDataset(filepath.Join("..", "..", "datasets", "smoke", "v3.json"))
	if err != nil {
		t.Fatalf("LoadDataset(v3) error = %v", err)
	}
	if loaded.Dataset.Name != "raghub-smoke-hybrid" || loaded.Dataset.Version != "3.0.0" {
		t.Fatalf("unexpected v3 identity: %s@%s", loaded.Dataset.Name, loaded.Dataset.Version)
	}
	const preregisteredSHA256 = "8fa1079bd1d0b1b895a6088087bcf9365df734ed95e8aa9c290109b1d437744e"
	if loaded.SHA256 != preregisteredSHA256 {
		t.Fatalf("v3 exact-byte SHA256 = %q, want frozen %q", loaded.SHA256, preregisteredSHA256)
	}
	if len(loaded.Dataset.Queries) != 20 {
		t.Fatalf("v3 query count = %d, want 20", len(loaded.Dataset.Queries))
	}

	allowedTenants := map[string]bool{"eval-smoke-v3-acme": true, "eval-smoke-v3-globex": true}
	chunker, err := ingest.NewMarkdownChunker(1200, 120)
	if err != nil {
		t.Fatalf("configure v3 contract chunker: %v", err)
	}
	expectedChunks := make(map[string]struct{}, len(loaded.Dataset.Documents))
	for _, document := range loaded.Dataset.Documents {
		if !allowedTenants[document.TenantID] || !strings.HasPrefix(document.DocumentID, "v3-") {
			t.Fatalf("v3 document is not isolated: tenant=%q id=%q", document.TenantID, document.DocumentID)
		}
		chunks, err := chunker.Chunk(document.Content)
		if err != nil || len(chunks) != 1 {
			t.Fatalf("v3 document %q must deterministically produce one chunk: count=%d error=%v", document.DocumentID, len(chunks), err)
		}
		expectedChunks[document.DocumentID+":v000001:c0000"] = struct{}{}
	}
	wantCategories := map[string]int{
		"exact-identifier":          4,
		"semantic-paraphrase":       4,
		"cross-language":            3,
		"near-duplicate-distractor": 3,
		"acl":                       3,
		"tenant-boundary":           3,
	}
	gotCategories := make(map[string]int)
	for _, query := range loaded.Dataset.Queries {
		if !allowedTenants[query.TenantID] || !strings.HasPrefix(query.ID, "v3-") {
			t.Fatalf("v3 query is not isolated: tenant=%q id=%q", query.TenantID, query.ID)
		}
		gotCategories[query.Category]++
		for _, chunkID := range append(append([]string(nil), query.GoldChunkIDs...), query.ForbiddenChunkIDs...) {
			if !strings.HasPrefix(chunkID, "v3-") || !strings.HasSuffix(chunkID, ":v000001:c0000") {
				t.Fatalf("v3 query %q has invalid preregistered chunk reference %q", query.ID, chunkID)
			}
			if _, exists := expectedChunks[chunkID]; !exists {
				t.Fatalf("v3 query %q references unknown preregistered chunk %q", query.ID, chunkID)
			}
		}
	}
	if len(gotCategories) != len(wantCategories) {
		t.Fatalf("v3 categories = %v, want exactly %v", gotCategories, wantCategories)
	}
	for category, want := range wantCategories {
		if gotCategories[category] != want {
			t.Fatalf("v3 category %q count = %d, want %d", category, gotCategories[category], want)
		}
	}
}
