package main

import (
	"fmt"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	embeddingopenai "github.com/ksana-ai/raghub/internal/embedding/openai"
)

var embeddingProfileIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$`)

const (
	defaultEmbeddingEndpoint   = "http://127.0.0.1:1234/v1/embeddings"
	defaultEmbeddingModel      = "text-embedding-bge-m3"
	defaultEmbeddingDimensions = 1024
	defaultEmbeddingTimeout    = 30 * time.Second
	defaultEmbeddingBatchSize  = 64
	defaultEmbeddingProfileID  = "lmstudio-bge-m3-1024-v1"
	embeddingProvider          = "lmstudio-openai-compatible"
	embeddingDocumentRecipe    = "indexed_text/v1"
	embeddingQueryRecipe       = "raw_query/v1"
	denseSearch                = "exact"
	denseDistance              = "cosine"
	embeddingModelRevision     = "not_reported_by_provider"
)

type denseSettings struct {
	Endpoint   string
	APIKey     string
	ProfileID  string
	Model      string
	Dimensions int
	Timeout    time.Duration
	BatchSize  int
}

func loadDenseSettings(getenv func(string) string) (denseSettings, error) {
	settings := denseSettings{
		Endpoint:   envOrDefault(getenv, "RAGHUB_EMBEDDING_ENDPOINT", defaultEmbeddingEndpoint),
		APIKey:     strings.TrimSpace(getenv("RAGHUB_EMBEDDING_API_KEY")),
		ProfileID:  envOrDefault(getenv, "RAGHUB_EMBEDDING_PROFILE_ID", defaultEmbeddingProfileID),
		Model:      envOrDefault(getenv, "RAGHUB_EMBEDDING_MODEL", defaultEmbeddingModel),
		Dimensions: defaultEmbeddingDimensions,
		Timeout:    defaultEmbeddingTimeout,
		BatchSize:  defaultEmbeddingBatchSize,
	}

	parsedEndpoint, err := url.Parse(settings.Endpoint)
	if err != nil || parsedEndpoint.Host == "" || (parsedEndpoint.Scheme != "http" && parsedEndpoint.Scheme != "https") ||
		parsedEndpoint.User != nil || parsedEndpoint.RawQuery != "" || parsedEndpoint.Fragment != "" {
		return denseSettings{}, fmt.Errorf("RAGHUB_EMBEDDING_ENDPOINT must be an absolute http(s) URL")
	}
	if !embeddingProfileIDPattern.MatchString(settings.ProfileID) {
		return denseSettings{}, fmt.Errorf("RAGHUB_EMBEDDING_PROFILE_ID must be a 1-128 character safe identifier")
	}
	if len(settings.Model) > 256 {
		return denseSettings{}, fmt.Errorf("RAGHUB_EMBEDDING_MODEL must not exceed 256 bytes")
	}
	if value := strings.TrimSpace(getenv("RAGHUB_EMBEDDING_DIMENSIONS")); value != "" {
		settings.Dimensions, err = strconv.Atoi(value)
		if err != nil || settings.Dimensions <= 0 {
			return denseSettings{}, fmt.Errorf("RAGHUB_EMBEDDING_DIMENSIONS must be a positive integer")
		}
	}
	if settings.Dimensions != defaultEmbeddingDimensions {
		return denseSettings{}, fmt.Errorf("RAGHUB_EMBEDDING_DIMENSIONS must be %d for the current pgvector schema", defaultEmbeddingDimensions)
	}
	if value := strings.TrimSpace(getenv("RAGHUB_EMBEDDING_TIMEOUT")); value != "" {
		settings.Timeout, err = time.ParseDuration(value)
		if err != nil || settings.Timeout <= 0 {
			return denseSettings{}, fmt.Errorf("RAGHUB_EMBEDDING_TIMEOUT must be a positive Go duration")
		}
	}
	if value := strings.TrimSpace(getenv("RAGHUB_EMBEDDING_BATCH_SIZE")); value != "" {
		settings.BatchSize, err = strconv.Atoi(value)
		if err != nil || settings.BatchSize < 1 || settings.BatchSize > 256 {
			return denseSettings{}, fmt.Errorf("RAGHUB_EMBEDDING_BATCH_SIZE must be between 1 and 256")
		}
	}
	return settings, nil
}

func envOrDefault(getenv func(string) string, key, fallback string) string {
	if value := strings.TrimSpace(getenv(key)); value != "" {
		return value
	}
	return fallback
}

// manifestConfig deliberately omits the API key. Provider runtime knobs are
// retained so failed or slow runs can be compared without leaking credentials.
func (settings denseSettings) manifestConfig() map[string]any {
	return map[string]any{
		"endpoint":           settings.Endpoint,
		"profile_id":         settings.ProfileID,
		"provider":           embeddingProvider,
		"model":              settings.Model,
		"model_revision":     embeddingModelRevision,
		"dimensions":         settings.Dimensions,
		"document_recipe":    embeddingDocumentRecipe,
		"query_recipe":       embeddingQueryRecipe,
		"search":             denseSearch,
		"distance":           denseDistance,
		"request_timeout_ms": settings.Timeout.Milliseconds(),
		"batch_size":         settings.BatchSize,
	}
}

func (settings denseSettings) clientConfig() embeddingopenai.Config {
	return embeddingopenai.Config{
		Endpoint:       settings.Endpoint,
		ProfileID:      settings.ProfileID,
		Provider:       embeddingProvider,
		Model:          settings.Model,
		Dimensions:     settings.Dimensions,
		DocumentRecipe: embeddingDocumentRecipe,
		QueryRecipe:    embeddingQueryRecipe,
		BatchSize:      settings.BatchSize,
		Timeout:        settings.Timeout,
		APIKey:         settings.APIKey,
	}
}
