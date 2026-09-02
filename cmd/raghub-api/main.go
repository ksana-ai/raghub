package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	openaiembedding "github.com/ksana-ai/raghub/internal/embedding/openai"
	"github.com/ksana-ai/raghub/internal/httpapi"
	"github.com/ksana-ai/raghub/internal/ingest"
	"github.com/ksana-ai/raghub/internal/readiness"
	"github.com/ksana-ai/raghub/internal/retrieval"
	"github.com/ksana-ai/raghub/internal/store/postgres"
)

const (
	defaultEmbeddingEndpoint  = "http://127.0.0.1:1234/v1/embeddings"
	defaultEmbeddingModel     = "text-embedding-bge-m3"
	defaultEmbeddingProfileID = "lmstudio-bge-m3-1024-v1"
	embeddingProvider         = "lmstudio-openai-compatible"
	embeddingDocumentRecipe   = "indexed_text/v1"
	embeddingQueryRecipe      = "raw_query/v1"
	storedEmbeddingDimensions = 1024
)

func main() {
	var (
		address             = flag.String("address", envOrDefault("RAGHUB_ADDRESS", ":8080"), "HTTP listen address")
		databaseURL         = flag.String("database-url", os.Getenv("RAGHUB_DATABASE_URL"), "PostgreSQL connection URL")
		migrate             = flag.Bool("migrate", envBool("RAGHUB_AUTO_MIGRATE"), "apply embedded database migrations before serving")
		embeddingEndpoint   = flag.String("embedding-endpoint", envOrDefault("RAGHUB_EMBEDDING_ENDPOINT", defaultEmbeddingEndpoint), "OpenAI-compatible embeddings endpoint")
		embeddingModel      = flag.String("embedding-model", envOrDefault("RAGHUB_EMBEDDING_MODEL", defaultEmbeddingModel), "embedding model name")
		embeddingProfileID  = flag.String("embedding-profile-id", envOrDefault("RAGHUB_EMBEDDING_PROFILE_ID", defaultEmbeddingProfileID), "immutable embedding profile ID")
		embeddingDimensions = flag.Int("embedding-dimensions", envIntOrDefault("RAGHUB_EMBEDDING_DIMENSIONS", storedEmbeddingDimensions), "embedding vector dimensions")
		embeddingBatchSize  = flag.Int("embedding-batch-size", envIntOrDefault("RAGHUB_EMBEDDING_BATCH_SIZE", 32), "maximum texts per embedding request")
		embeddingTimeout    = flag.Duration("embedding-timeout", envDurationOrDefault("RAGHUB_EMBEDDING_TIMEOUT", 30*time.Second), "timeout per embedding request")
	)
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if strings.TrimSpace(*databaseURL) == "" {
		logger.Error("database URL is required", "hint", "set RAGHUB_DATABASE_URL or -database-url")
		os.Exit(2)
	}
	if *embeddingDimensions != storedEmbeddingDimensions {
		logger.Error("embedding dimensions must match the database vector size", "got", *embeddingDimensions, "want", storedEmbeddingDimensions)
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pool, err := pgxpool.New(ctx, *databaseURL)
	if err != nil {
		logger.Error("create database pool", "error", err)
		os.Exit(1)
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		logger.Error("connect to database", "error", err)
		os.Exit(1)
	}
	if *migrate {
		if err := postgres.ApplyMigrations(ctx, pool); err != nil {
			logger.Error("apply database migrations", "error", err)
			os.Exit(1)
		}
	}

	store := postgres.New(pool)
	chunker, err := ingest.NewMarkdownChunker(0, 120)
	if err != nil {
		logger.Error("configure chunker", "error", err)
		os.Exit(1)
	}
	embedder, err := openaiembedding.NewClient(openaiembedding.Config{
		Endpoint:       *embeddingEndpoint,
		ProfileID:      *embeddingProfileID,
		Provider:       embeddingProvider,
		Model:          *embeddingModel,
		Dimensions:     *embeddingDimensions,
		DocumentRecipe: embeddingDocumentRecipe,
		QueryRecipe:    embeddingQueryRecipe,
		BatchSize:      *embeddingBatchSize,
		Timeout:        *embeddingTimeout,
		APIKey:         os.Getenv("RAGHUB_EMBEDDING_API_KEY"),
	})
	if err != nil {
		logger.Error("configure embedding client", "error", err)
		os.Exit(2)
	}
	ingestionService := ingest.NewServiceWithEmbedder(store, chunker, embedder)
	retrievalService, err := retrieval.NewServiceWithHybrid(store, store, embedder, retrieval.DefaultHybridConfig())
	if err != nil {
		logger.Error("configure hybrid retriever", "error", err)
		os.Exit(2)
	}
	readinessChecker, err := readiness.New(pool, embedder, 10*time.Second)
	if err != nil {
		logger.Error("configure readiness checker", "error", err)
		os.Exit(2)
	}

	server := &http.Server{
		Addr:              *address,
		Handler:           httpapi.New(ingestionService, retrievalService, readinessChecker, logger),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      2 * time.Minute,
		IdleTimeout:       60 * time.Second,
	}

	errChannel := make(chan error, 1)
	go func() {
		logger.Info("raghub API listening", "address", *address)
		errChannel <- server.ListenAndServe()
	}()

	select {
	case <-ctx.Done():
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			logger.Error("graceful shutdown failed", "error", err)
			os.Exit(1)
		}
	case err := <-errChannel:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("HTTP server stopped", "error", err)
			os.Exit(1)
		}
	}
}

func envOrDefault(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func envBool(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func envIntOrDefault(name string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return -1
	}
	return parsed
}

func envDurationOrDefault(name string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil {
		return -1
	}
	return parsed
}
