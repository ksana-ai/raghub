package eval

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

func TestParseDatasetPreservesInputDigest(t *testing.T) {
	t.Parallel()

	data := []byte(`{
  "schema_version": "raghub.eval.dataset/v1",
  "name": "test",
  "version": "1.0.0",
  "documents": [{
    "tenant_id": "tenant-a",
    "document_id": "doc-a",
    "title": "Document A",
    "source_uri": "fixture://doc-a",
    "content": "body",
    "metadata": {"kind": "test"}
  }],
  "queries": [{
    "id": "q1",
    "category": "exact",
    "tenant_id": "tenant-a",
    "query": "body",
    "gold_chunk_ids": ["doc-a:v000001:c0000"]
  }]
}`)

	loaded, err := ParseDataset(data)
	if err != nil {
		t.Fatalf("ParseDataset() error = %v", err)
	}
	digest := sha256.Sum256(data)
	wantDigest := hex.EncodeToString(digest[:])
	if loaded.SHA256 != wantDigest {
		t.Fatalf("SHA256 = %q, want %q", loaded.SHA256, wantDigest)
	}
	if loaded.Dataset.Name != "test" || len(loaded.Dataset.Queries) != 1 {
		t.Fatalf("unexpected dataset: %+v", loaded.Dataset)
	}
}

func TestParseDatasetRejectsUnknownFields(t *testing.T) {
	t.Parallel()

	_, err := ParseDataset([]byte(`{
  "schema_version": "raghub.eval.dataset/v1",
  "name": "test",
  "version": "1",
  "unexpected": true,
  "documents": [],
  "queries": []
}`))
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("ParseDataset() error = %v, want unknown field error", err)
	}
}

func TestParseDatasetRequiresGloballyUniqueDocumentIDs(t *testing.T) {
	t.Parallel()

	_, err := ParseDataset([]byte(`{
  "schema_version": "raghub.eval.dataset/v1",
  "name": "test",
  "version": "1",
  "documents": [
    {"tenant_id":"tenant-a","document_id":"same","title":"A","content":"alpha"},
    {"tenant_id":"tenant-b","document_id":"same","title":"B","content":"beta"}
  ],
  "queries": [{
    "id":"q1","category":"exact","tenant_id":"tenant-a","query":"alpha",
    "gold_chunk_ids":["same:v000001:c0000"]
  }]
}`))
	if err == nil || !strings.Contains(err.Error(), "duplicates global document_id") {
		t.Fatalf("ParseDataset() error = %v, want global document_id error", err)
	}
}
