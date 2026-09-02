package eval

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ksana-ai/raghub/internal/ingest"
)

func TestHardBenchmarkV1PreregisteredContract(t *testing.T) {
	t.Parallel()

	loaded, err := LoadDataset(filepath.Join("..", "..", "datasets", "benchmark", "v1.json"))
	if err != nil {
		t.Fatalf("LoadDataset(benchmark v1) error = %v", err)
	}
	if loaded.Dataset.Name != "raghub-hard-benchmark" || loaded.Dataset.Version != "1.0.0" {
		t.Fatalf("unexpected benchmark identity: %s@%s", loaded.Dataset.Name, loaded.Dataset.Version)
	}
	const preregisteredSHA256 = "aa44175b9ae656d97473a8340ebac59bc1432d7cee90e51432c2b4f89e61f85f"
	if loaded.SHA256 != preregisteredSHA256 {
		t.Fatalf("benchmark exact-byte SHA256 = %q, want frozen %q", loaded.SHA256, preregisteredSHA256)
	}
	if len(loaded.Dataset.Documents) != 44 || len(loaded.Dataset.Queries) != 50 {
		t.Fatalf("benchmark documents/queries = %d/%d, want 44/50", len(loaded.Dataset.Documents), len(loaded.Dataset.Queries))
	}

	allowedTenants := map[string]bool{
		"eval-benchmark-v1-acme":   true,
		"eval-benchmark-v1-globex": true,
	}
	type chunkOwner struct {
		tenant            string
		allowedPrincipals []string
	}
	chunkOwners := make(map[string]chunkOwner, len(loaded.Dataset.Documents))
	chunker, err := ingest.NewMarkdownChunker(1200, 120)
	if err != nil {
		t.Fatal(err)
	}
	for _, document := range loaded.Dataset.Documents {
		if !allowedTenants[document.TenantID] || !strings.HasPrefix(document.DocumentID, "b1-") {
			t.Fatalf("benchmark document is not isolated: tenant=%q id=%q", document.TenantID, document.DocumentID)
		}
		chunks, err := chunker.Chunk(document.Content)
		if err != nil || len(chunks) != 1 {
			t.Fatalf("benchmark document %q must produce one deterministic chunk: count=%d error=%v", document.DocumentID, len(chunks), err)
		}
		chunkOwners[document.DocumentID+":v000001:c0000"] = chunkOwner{
			tenant: document.TenantID, allowedPrincipals: append([]string(nil), document.AllowedPrincipals...),
		}
	}

	wantCategories := map[string]int{
		"exact-disambiguation":      8,
		"semantic-paraphrase":       8,
		"cross-language":            6,
		"near-duplicate-distractor": 8,
		"multi-relevant":            8,
		"acl":                       6,
		"tenant-boundary":           6,
	}
	gotCategories := make(map[string]int)
	queryIdentities := make(map[string]struct{}, len(loaded.Dataset.Queries))
	aclForbiddenCases := 0
	for _, query := range loaded.Dataset.Queries {
		if !allowedTenants[query.TenantID] || !strings.HasPrefix(query.ID, "b1-") {
			t.Fatalf("benchmark query is not isolated: tenant=%q id=%q", query.TenantID, query.ID)
		}
		identity := query.TenantID + "\x00" + query.PrincipalID + "\x00" + query.Query
		if _, duplicate := queryIdentities[identity]; duplicate {
			t.Fatalf("benchmark duplicates tenant/principal/query identity for %q", query.ID)
		}
		queryIdentities[identity] = struct{}{}
		gotCategories[query.Category]++
		wantGoldCount := 1
		if query.Category == "multi-relevant" {
			wantGoldCount = 2
		}
		if len(query.GoldChunkIDs) != wantGoldCount {
			t.Fatalf("query %q gold count = %d, want %d", query.ID, len(query.GoldChunkIDs), wantGoldCount)
		}
		for _, chunkID := range query.GoldChunkIDs {
			owner, exists := chunkOwners[chunkID]
			if !exists || owner.tenant != query.TenantID {
				t.Fatalf("query %q has invalid gold owner for %q: %+v", query.ID, chunkID, owner)
			}
			if len(owner.allowedPrincipals) > 0 && !slices.Contains(owner.allowedPrincipals, query.PrincipalID) {
				t.Fatalf("query %q cannot access restricted gold %q", query.ID, chunkID)
			}
		}
		for _, chunkID := range query.ForbiddenChunkIDs {
			if _, exists := chunkOwners[chunkID]; !exists {
				t.Fatalf("query %q references unknown forbidden chunk %q", query.ID, chunkID)
			}
		}
		switch query.Category {
		case "acl":
			if query.PrincipalID == "" {
				t.Fatalf("ACL query %q requires a principal", query.ID)
			}
			if len(query.ForbiddenChunkIDs) > 0 {
				aclForbiddenCases++
			}
		case "tenant-boundary":
			if len(query.ForbiddenChunkIDs) != 1 || chunkOwners[query.ForbiddenChunkIDs[0]].tenant == query.TenantID {
				t.Fatalf("tenant query %q must forbid the paired other-tenant chunk", query.ID)
			}
		}
	}
	if aclForbiddenCases != 2 {
		t.Fatalf("ACL forbidden cases = %d, want 2", aclForbiddenCases)
	}
	if len(gotCategories) != len(wantCategories) {
		t.Fatalf("benchmark categories = %v, want exactly %v", gotCategories, wantCategories)
	}
	for category, want := range wantCategories {
		if gotCategories[category] != want {
			t.Fatalf("benchmark category %q count = %d, want %d", category, gotCategories[category], want)
		}
	}
}
