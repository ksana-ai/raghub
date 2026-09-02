// Package openai implements the OpenAI-compatible embeddings protocol used by
// LM Studio. It has no dependency on an OpenAI-hosted service.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ksana-ai/raghub/internal/model"
)

const (
	defaultBatchSize      = 64
	defaultRequestTimeout = 30 * time.Second
	maxBatchSize          = 256
	maxResponseBytes      = 64 << 20
)

type Config struct {
	Endpoint       string
	ProfileID      string
	Provider       string
	Model          string
	Dimensions     int
	DocumentRecipe string
	QueryRecipe    string
	BatchSize      int
	Timeout        time.Duration
	APIKey         string
	HTTPClient     *http.Client
}

// Client embeds text through an OpenAI-compatible /v1/embeddings endpoint.
// It is safe for concurrent use.
type Client struct {
	endpoint   string
	profile    model.EmbeddingProfile
	batchSize  int
	timeout    time.Duration
	apiKey     string
	httpClient *http.Client
}

func NewClient(config Config) (*Client, error) {
	config.Endpoint = strings.TrimSpace(config.Endpoint)
	config.ProfileID = strings.TrimSpace(config.ProfileID)
	config.Provider = strings.TrimSpace(config.Provider)
	config.Model = strings.TrimSpace(config.Model)
	config.DocumentRecipe = strings.TrimSpace(config.DocumentRecipe)
	config.QueryRecipe = strings.TrimSpace(config.QueryRecipe)
	config.APIKey = strings.TrimSpace(config.APIKey)
	if config.Endpoint == "" {
		return nil, errors.New("embedding endpoint is required")
	}
	parsed, err := url.Parse(config.Endpoint)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, errors.New("embedding endpoint must be an absolute http or https URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, errors.New("embedding endpoint must not contain credentials, a query, or a fragment")
	}
	if config.Model == "" {
		return nil, errors.New("embedding model is required")
	}
	if config.ProfileID == "" {
		return nil, errors.New("embedding profile ID is required")
	}
	if config.Provider == "" {
		return nil, errors.New("embedding provider is required")
	}
	if config.DocumentRecipe == "" || config.QueryRecipe == "" {
		return nil, errors.New("embedding document and query recipes are required")
	}
	if config.Dimensions <= 0 {
		return nil, errors.New("embedding dimensions must be positive")
	}
	if config.BatchSize == 0 {
		config.BatchSize = defaultBatchSize
	}
	if config.BatchSize < 1 || config.BatchSize > maxBatchSize {
		return nil, fmt.Errorf("embedding batch size must be between 1 and %d", maxBatchSize)
	}
	if config.Timeout == 0 {
		config.Timeout = defaultRequestTimeout
	}
	if config.Timeout < 0 {
		return nil, errors.New("embedding timeout must not be negative")
	}
	if config.HTTPClient == nil {
		config.HTTPClient = http.DefaultClient
	}

	return &Client{
		endpoint: strings.TrimRight(config.Endpoint, "/"),
		profile: model.EmbeddingProfile{
			ProfileID: config.ProfileID, Provider: config.Provider, Model: config.Model,
			Dimensions: config.Dimensions, DocumentRecipe: config.DocumentRecipe, QueryRecipe: config.QueryRecipe,
		},
		batchSize:  config.BatchSize,
		timeout:    config.Timeout,
		apiKey:     config.APIKey,
		httpClient: config.HTTPClient,
	}, nil
}

func (c *Client) Profile() model.EmbeddingProfile { return c.profile }
func (c *Client) Model() string                   { return c.profile.Model }
func (c *Client) Dimensions() int                 { return c.profile.Dimensions }
func (c *Client) Endpoint() string                { return c.endpoint }
func (c *Client) BatchSize() int                  { return c.batchSize }
func (c *Client) Timeout() time.Duration          { return c.timeout }

func (c *Client) Embed(ctx context.Context, inputs []string) ([][]float32, error) {
	if c == nil || c.httpClient == nil {
		return nil, errors.New("embed: nil embedding client")
	}
	if len(inputs) == 0 {
		return nil, errors.New("embed: at least one input is required")
	}
	for index, input := range inputs {
		if strings.TrimSpace(input) == "" {
			return nil, fmt.Errorf("embed: input %d is empty", index)
		}
	}

	vectors := make([][]float32, 0, len(inputs))
	for start := 0; start < len(inputs); start += c.batchSize {
		end := min(start+c.batchSize, len(inputs))
		batch, err := c.embedBatch(ctx, inputs[start:end])
		if err != nil {
			return nil, fmt.Errorf("embed inputs %d-%d: %w", start, end-1, err)
		}
		vectors = append(vectors, batch...)
	}
	return vectors, nil
}

type embeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embeddingResponse struct {
	Model string `json:"model"`
	Data  []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

func (c *Client) embedBatch(ctx context.Context, inputs []string) ([][]float32, error) {
	payload, err := json.Marshal(embeddingRequest{Model: c.profile.Model, Input: inputs})
	if err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}
	requestCtx := ctx
	cancel := func() {}
	if c.timeout > 0 {
		requestCtx, cancel = context.WithTimeout(ctx, c.timeout)
	}
	defer cancel()

	request, err := http.NewRequestWithContext(requestCtx, http.MethodPost, c.endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	if c.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	response, err := c.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if len(body) > maxResponseBytes {
		return nil, fmt.Errorf("response exceeds %d bytes", maxResponseBytes)
	}

	var decoded embeddingResponse
	decodeErr := json.Unmarshal(body, &decoded)
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		message := strings.TrimSpace(string(body))
		if decodeErr == nil && decoded.Error != nil && strings.TrimSpace(decoded.Error.Message) != "" {
			message = strings.TrimSpace(decoded.Error.Message)
		}
		if len(message) > 4096 {
			message = message[:4096]
		}
		return nil, fmt.Errorf("endpoint returned HTTP %d: %s", response.StatusCode, message)
	}
	if decodeErr != nil {
		return nil, fmt.Errorf("decode response: %w", decodeErr)
	}
	if decoded.Error != nil {
		return nil, fmt.Errorf("endpoint error: %s", strings.TrimSpace(decoded.Error.Message))
	}
	if strings.TrimSpace(decoded.Model) == "" {
		return nil, errors.New("response model is required for embedding provenance")
	}
	if decoded.Model != c.profile.Model {
		return nil, fmt.Errorf("response model %q does not match configured model %q", decoded.Model, c.profile.Model)
	}
	if len(decoded.Data) != len(inputs) {
		return nil, fmt.Errorf("response contains %d embeddings for %d inputs", len(decoded.Data), len(inputs))
	}

	vectors := make([][]float32, len(inputs))
	for _, item := range decoded.Data {
		if item.Index < 0 || item.Index >= len(inputs) {
			return nil, fmt.Errorf("response embedding index %d is out of range", item.Index)
		}
		if vectors[item.Index] != nil {
			return nil, fmt.Errorf("response embedding index %d is duplicated", item.Index)
		}
		if len(item.Embedding) != c.profile.Dimensions {
			return nil, fmt.Errorf("response embedding index %d has %d dimensions, want %d", item.Index, len(item.Embedding), c.profile.Dimensions)
		}
		for dimension, value := range item.Embedding {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("response embedding index %d dimension %d is not finite", item.Index, dimension)
			}
		}
		vectors[item.Index] = append([]float32(nil), item.Embedding...)
	}
	return vectors, nil
}
