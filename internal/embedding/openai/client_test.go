package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestClientEmbedsBatchesAndRestoresResponseOrder(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("Authorization = %q", got)
		}
		var request embeddingRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if request.Model != "test-model" {
			t.Errorf("model = %q", request.Model)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"test-model","data":[{"index":1,"embedding":[2,2,2]},{"index":0,"embedding":[1,1,1]}]}`))
	}))
	defer server.Close()

	client, err := NewClient(Config{
		Endpoint:  server.URL,
		ProfileID: "test-profile", Provider: "test", Model: "test-model",
		Dimensions: 3, DocumentRecipe: "indexed_text/v1", QueryRecipe: "raw_query/v1", BatchSize: 2,
		APIKey: "secret", Timeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	vectors, err := client.Embed(context.Background(), []string{"a", "b", "c", "d"})
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 || len(vectors) != 4 {
		t.Fatalf("calls=%d vectors=%d", calls.Load(), len(vectors))
	}
	for index, vector := range vectors {
		want := float32(1)
		if index%2 == 1 {
			want = 2
		}
		if len(vector) != 3 || vector[0] != want {
			t.Fatalf("vector[%d] = %v, want leading %v", index, vector, want)
		}
	}
}

func TestClientRejectsMalformedResponses(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "wrong dimensions", body: `{"model":"m","data":[{"index":0,"embedding":[1]}]}`, want: "1 dimensions, want 2"},
		{name: "duplicate index", body: `{"model":"m","data":[{"index":0,"embedding":[1,2]},{"index":0,"embedding":[3,4]}]}`, want: "duplicated"},
		{name: "wrong model", body: `{"model":"other","data":[{"index":0,"embedding":[1,2]}]}`, want: "does not match"},
		{name: "missing model", body: `{"data":[{"index":0,"embedding":[1,2]}]}`, want: "model is required"},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()
			client, err := NewClient(testConfig(server.URL))
			if err != nil {
				t.Fatal(err)
			}
			inputs := []string{"x"}
			if test.name == "duplicate index" {
				inputs = append(inputs, "y")
			}
			_, err = client.Embed(context.Background(), inputs)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want substring %q", err, test.want)
			}
		})
	}
}

func TestClientReturnsBoundedEndpointError(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"message":"model unavailable"}}`))
	}))
	defer server.Close()
	client, err := NewClient(testConfig(server.URL))
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "HTTP 503: model unavailable") {
		t.Fatalf("error = %v", err)
	}
}

func TestClientHonorsRequestTimeout(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		time.Sleep(100 * time.Millisecond)
		_, _ = w.Write([]byte(`{"model":"m","data":[{"index":0,"embedding":[1,2]}]}`))
	}))
	defer server.Close()
	config := testConfig(server.URL)
	config.Timeout = 20 * time.Millisecond
	client, err := NewClient(config)
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Embed(context.Background(), []string{"x"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("error = %v, want context deadline exceeded", err)
	}
}

func TestClientHonorsCallerCancellation(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()
	client, err := NewClient(testConfig(server.URL))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = client.Embed(ctx, []string{"x"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context canceled", err)
	}
}

func TestNewClientValidatesConfiguration(t *testing.T) {
	t.Parallel()
	tests := []Config{
		{},
		{Endpoint: "file:///tmp/model", ProfileID: "p", Provider: "x", Model: "m", Dimensions: 2, DocumentRecipe: "d", QueryRecipe: "q"},
		{Endpoint: "http://user:secret@example.test/v1/embeddings", ProfileID: "p", Provider: "x", Model: "m", Dimensions: 2, DocumentRecipe: "d", QueryRecipe: "q"},
		{Endpoint: "http://example.test/v1/embeddings", ProfileID: "p", Provider: "x", Dimensions: 2, DocumentRecipe: "d", QueryRecipe: "q"},
		{Endpoint: "http://example.test/v1/embeddings", ProfileID: "p", Provider: "x", Model: "m", DocumentRecipe: "d", QueryRecipe: "q"},
		{Endpoint: "http://example.test/v1/embeddings", ProfileID: "p", Provider: "x", Model: "m", Dimensions: 2, DocumentRecipe: "d", QueryRecipe: "q", BatchSize: 257},
		{Endpoint: "http://example.test/v1/embeddings", ProfileID: "p", Provider: "x", Model: "m", Dimensions: 2, DocumentRecipe: "d", QueryRecipe: "q", Timeout: -time.Second},
	}
	for index, config := range tests {
		if _, err := NewClient(config); err == nil {
			t.Errorf("config %d unexpectedly succeeded", index)
		}
	}
}

func testConfig(endpoint string) Config {
	return Config{
		Endpoint: endpoint, ProfileID: "p", Provider: "test", Model: "m",
		Dimensions: 2, DocumentRecipe: "indexed_text/v1", QueryRecipe: "raw_query/v1",
	}
}
