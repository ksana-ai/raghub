package ingest

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"raghub/internal/model"
)

type captureStore struct {
	document    model.DocumentInput
	fingerprint string
	chunks      []model.ChunkDraft
}

func (s *captureStore) SaveDocumentVersion(_ context.Context, document model.DocumentInput, fingerprint string, chunks []model.ChunkDraft) (model.IngestResult, error) {
	s.document = document
	s.fingerprint = fingerprint
	s.chunks = chunks
	return model.IngestResult{TenantID: document.TenantID, DocumentID: document.ID}, nil
}

type staticChunker struct{}

func (staticChunker) Chunk(string) ([]model.ChunkDraft, error) {
	return []model.ChunkDraft{{Ordinal: 0, RawText: "content", IndexedText: "content"}}, nil
}

type customChunker struct{ text string }

func (c customChunker) Chunk(string) ([]model.ChunkDraft, error) {
	return []model.ChunkDraft{{Ordinal: 0, RawText: c.text, IndexedText: c.text}}, nil
}

func TestServiceCanonicalizesVersionInputs(t *testing.T) {
	store := &captureStore{}
	service := NewService(store, staticChunker{})

	_, err := service.Ingest(context.Background(), model.DocumentInput{
		TenantID:          " tenant-a ",
		ID:                " guide ",
		Title:             " Guide ",
		Content:           "body",
		AllowedPrincipals: []string{"user:bob", "user:alice", "user:bob"},
		Metadata:          json.RawMessage(`{"z":1,"a":2}`),
	})
	if err != nil {
		t.Fatal(err)
	}

	if got, want := store.document.AllowedPrincipals, []string{"user:alice", "user:bob"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("principals = %#v, want %#v", got, want)
	}
	if got, want := string(store.document.Metadata), `{"a":2,"z":1}`; got != want {
		t.Fatalf("metadata = %s, want %s", got, want)
	}
	if len(store.fingerprint) != 64 {
		t.Fatalf("fingerprint length = %d, want 64", len(store.fingerprint))
	}
}

func TestServiceFingerprintIgnoresACLAndMetadataOrdering(t *testing.T) {
	storeA := &captureStore{}
	storeB := &captureStore{}
	documentA := model.DocumentInput{
		TenantID: "tenant-a", ID: "guide", Title: "Guide", Content: "body",
		AllowedPrincipals: []string{"user:bob", "user:alice"},
		Metadata:          json.RawMessage(`{"z":1,"a":2}`),
	}
	documentB := model.DocumentInput{
		TenantID: "tenant-a", ID: "guide", Title: "Guide", Content: "body",
		AllowedPrincipals: []string{"user:alice", "user:bob"},
		Metadata:          json.RawMessage(`{"a":2,"z":1}`),
	}
	if _, err := NewService(storeA, staticChunker{}).Ingest(context.Background(), documentA); err != nil {
		t.Fatal(err)
	}
	if _, err := NewService(storeB, staticChunker{}).Ingest(context.Background(), documentB); err != nil {
		t.Fatal(err)
	}
	if storeA.fingerprint != storeB.fingerprint {
		t.Fatalf("fingerprints differ: %s != %s", storeA.fingerprint, storeB.fingerprint)
	}
}

func TestServiceFingerprintIncludesChunkOutput(t *testing.T) {
	document := model.DocumentInput{
		TenantID: "tenant-a", ID: "guide", Title: "Guide", Content: "body",
		Metadata: json.RawMessage(`{}`),
	}
	storeA := &captureStore{}
	storeB := &captureStore{}
	if _, err := NewService(storeA, customChunker{text: "first chunk"}).Ingest(context.Background(), document); err != nil {
		t.Fatal(err)
	}
	if _, err := NewService(storeB, customChunker{text: "different chunk"}).Ingest(context.Background(), document); err != nil {
		t.Fatal(err)
	}
	if storeA.fingerprint == storeB.fingerprint {
		t.Fatal("fingerprint did not change with chunk output")
	}
}

func TestServiceWrapsValidationError(t *testing.T) {
	_, err := NewService(&captureStore{}, staticChunker{}).Ingest(context.Background(), model.DocumentInput{
		TenantID: "bad tenant", Title: "Guide", Content: "body",
	})
	if !errors.Is(err, ErrInvalidInput) {
		t.Fatalf("error = %v, want ErrInvalidInput", err)
	}
}
