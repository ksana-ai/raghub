package eval

import (
	"path/filepath"
	"strings"
	"testing"
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
