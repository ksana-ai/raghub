package main

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestLoadDenseSettingsDefaults(t *testing.T) {
	t.Parallel()

	settings, err := loadDenseSettings(func(string) string { return "" })
	if err != nil {
		t.Fatalf("loadDenseSettings() error = %v", err)
	}
	if settings.Endpoint != defaultEmbeddingEndpoint || settings.Model != defaultEmbeddingModel ||
		settings.ProfileID != defaultEmbeddingProfileID || settings.Dimensions != 1024 ||
		settings.Timeout != 30*time.Second || settings.BatchSize != 64 {
		t.Fatalf("unexpected defaults: %+v", settings)
	}
}

func TestDenseManifestConfigExcludesSecretAndRecordsRuntimeKnobs(t *testing.T) {
	t.Parallel()

	values := map[string]string{
		"RAGHUB_EMBEDDING_ENDPOINT":   "https://embeddings.example.test/v1/embeddings",
		"RAGHUB_EMBEDDING_API_KEY":    "do-not-record",
		"RAGHUB_EMBEDDING_PROFILE_ID": "custom-profile-v1",
		"RAGHUB_EMBEDDING_MODEL":      "example-multilingual",
		"RAGHUB_EMBEDDING_DIMENSIONS": "1024",
		"RAGHUB_EMBEDDING_TIMEOUT":    "45s",
		"RAGHUB_EMBEDDING_BATCH_SIZE": "16",
	}
	settings, err := loadDenseSettings(func(key string) string { return values[key] })
	if err != nil {
		t.Fatalf("loadDenseSettings() error = %v", err)
	}
	want := map[string]any{
		"endpoint":           "https://embeddings.example.test/v1/embeddings",
		"profile_id":         "custom-profile-v1",
		"provider":           "lmstudio-openai-compatible",
		"model":              "example-multilingual",
		"model_revision":     "not_reported_by_provider",
		"dimensions":         1024,
		"document_recipe":    "indexed_text/v1",
		"query_recipe":       "raw_query/v1",
		"search":             "exact",
		"distance":           "cosine",
		"request_timeout_ms": int64(45000),
		"batch_size":         16,
	}
	if got := settings.manifestConfig(); !reflect.DeepEqual(got, want) {
		t.Fatalf("manifestConfig() = %#v, want %#v", got, want)
	}
	clientConfig := settings.clientConfig()
	if clientConfig.APIKey != "do-not-record" || clientConfig.ProfileID != "custom-profile-v1" ||
		clientConfig.DocumentRecipe != embeddingDocumentRecipe || clientConfig.QueryRecipe != embeddingQueryRecipe {
		t.Fatal("client config does not preserve private/runtime settings")
	}
	for key, value := range settings.manifestConfig() {
		if strings.Contains(key, "key") || value == "do-not-record" {
			t.Fatalf("manifest leaked API key through %q=%v", key, value)
		}
	}
}

func TestLoadDenseSettingsRejectsInvalidValues(t *testing.T) {
	t.Parallel()

	tests := []struct {
		key   string
		value string
	}{
		{key: "RAGHUB_EMBEDDING_ENDPOINT", value: "localhost:1234"},
		{key: "RAGHUB_EMBEDDING_ENDPOINT", value: "http://example.test/v1/embeddings?token=secret"},
		{key: "RAGHUB_EMBEDDING_PROFILE_ID", value: "not safe!"},
		{key: "RAGHUB_EMBEDDING_MODEL", value: strings.Repeat("m", 257)},
		{key: "RAGHUB_EMBEDDING_DIMENSIONS", value: "0"},
		{key: "RAGHUB_EMBEDDING_DIMENSIONS", value: "768"},
		{key: "RAGHUB_EMBEDDING_DIMENSIONS", value: "many"},
		{key: "RAGHUB_EMBEDDING_TIMEOUT", value: "forever"},
		{key: "RAGHUB_EMBEDDING_BATCH_SIZE", value: "-1"},
		{key: "RAGHUB_EMBEDDING_BATCH_SIZE", value: "257"},
	}
	for _, test := range tests {
		t.Run(test.key+"="+test.value, func(t *testing.T) {
			t.Parallel()
			_, err := loadDenseSettings(func(key string) string {
				if key == test.key {
					return test.value
				}
				return ""
			})
			if err == nil {
				t.Fatal("loadDenseSettings() error = nil")
			}
		})
	}
}
