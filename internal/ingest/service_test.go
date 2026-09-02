package ingest

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"

	"github.com/ksana-ai/raghub/internal/model"
)

type captureStore struct {
	document    model.DocumentInput
	fingerprint string
	chunks      []model.ChunkDraft
	called      bool
}

func (s *captureStore) SaveDocumentVersion(_ context.Context, document model.DocumentInput, fingerprint string, chunks []model.ChunkDraft) (model.IngestResult, error) {
	s.called = true
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

type fakeEmbedder struct {
	profile model.EmbeddingProfile
	vectors [][]float32
	err     error
	inputs  []string
}

func (e *fakeEmbedder) Profile() model.EmbeddingProfile { return e.profile }
func (e *fakeEmbedder) Embed(_ context.Context, inputs []string) ([][]float32, error) {
	e.inputs = append([]string(nil), inputs...)
	return e.vectors, e.err
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

func TestServiceEmbedsIndexedTextBeforeSaving(t *testing.T) {
	store := &captureStore{}
	embedder := &fakeEmbedder{
		profile: testEmbeddingProfile("profile-a"),
		vectors: [][]float32{{0.25, 0.75}},
	}
	_, err := NewServiceWithEmbedder(store, customChunker{text: "indexed chunk"}, embedder).Ingest(context.Background(), model.DocumentInput{
		TenantID: "tenant-a", ID: "guide", Title: "Guide", Content: "body", Metadata: json.RawMessage(`{}`),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(embedder.inputs, []string{"indexed chunk"}) {
		t.Fatalf("embedding inputs = %#v", embedder.inputs)
	}
	if len(store.chunks) != 1 || store.chunks[0].Embedding == nil {
		t.Fatalf("stored chunks missing embedding: %+v", store.chunks)
	}
	if got := store.chunks[0].Embedding; got.Profile.ProfileID != "profile-a" || !reflect.DeepEqual(got.Values, []float32{0.25, 0.75}) {
		t.Fatalf("embedding draft = %+v", got)
	}
}

func TestServiceEmbeddingProfileDoesNotChangeDocumentFingerprint(t *testing.T) {
	document := model.DocumentInput{TenantID: "tenant-a", ID: "guide", Title: "Guide", Content: "body", Metadata: json.RawMessage(`{}`)}
	storeA := &captureStore{}
	storeB := &captureStore{}
	embedderA := &fakeEmbedder{profile: testEmbeddingProfile("profile-a"), vectors: [][]float32{{1, 0}}}
	embedderB := &fakeEmbedder{profile: testEmbeddingProfile("profile-b"), vectors: [][]float32{{0, 1}}}
	if _, err := NewServiceWithEmbedder(storeA, staticChunker{}, embedderA).Ingest(context.Background(), document); err != nil {
		t.Fatal(err)
	}
	if _, err := NewServiceWithEmbedder(storeB, staticChunker{}, embedderB).Ingest(context.Background(), document); err != nil {
		t.Fatal(err)
	}
	if storeA.fingerprint != storeB.fingerprint {
		t.Fatalf("embedding profile changed source fingerprint: %s != %s", storeA.fingerprint, storeB.fingerprint)
	}
}

func TestServiceDoesNotSaveWhenEmbeddingFailsValidation(t *testing.T) {
	tests := []struct {
		name    string
		vectors [][]float32
	}{
		{name: "wrong count", vectors: nil},
		{name: "wrong dimensions", vectors: [][]float32{{1}}},
		{name: "zero vector", vectors: [][]float32{{0, 0}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := &captureStore{}
			embedder := &fakeEmbedder{profile: testEmbeddingProfile("profile-a"), vectors: test.vectors}
			_, err := NewServiceWithEmbedder(store, staticChunker{}, embedder).Ingest(context.Background(), model.DocumentInput{
				TenantID: "tenant-a", ID: "guide", Title: "Guide", Content: "body", Metadata: json.RawMessage(`{}`),
			})
			if err == nil {
				t.Fatal("expected embedding error")
			}
			if store.called {
				t.Fatal("store was called after invalid embedding response")
			}
		})
	}
}

func testEmbeddingProfile(id string) model.EmbeddingProfile {
	return model.EmbeddingProfile{
		ProfileID: id, Provider: "test", Model: "test-model", Dimensions: 2,
		DocumentRecipe: "indexed_text/v1", QueryRecipe: "raw_query/v1",
	}
}
