package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"raghub/internal/httpapi"
	"raghub/internal/ingest"
	"raghub/internal/retrieval"
	"raghub/internal/store/postgres"
)

func main() {
	var (
		address     = flag.String("address", envOrDefault("RAGHUB_ADDRESS", ":8080"), "HTTP listen address")
		databaseURL = flag.String("database-url", os.Getenv("RAGHUB_DATABASE_URL"), "PostgreSQL connection URL")
		migrate     = flag.Bool("migrate", envBool("RAGHUB_AUTO_MIGRATE"), "apply embedded database migrations before serving")
	)
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if strings.TrimSpace(*databaseURL) == "" {
		logger.Error("database URL is required", "hint", "set RAGHUB_DATABASE_URL or -database-url")
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
	ingestionService := ingest.NewService(store, chunker)
	retrievalService := retrieval.NewService(store)

	server := &http.Server{
		Addr:              *address,
		Handler:           httpapi.New(ingestionService, retrievalService, pool, logger),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
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
