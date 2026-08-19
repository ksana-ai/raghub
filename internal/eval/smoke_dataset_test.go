package eval

import (
	"path/filepath"
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
