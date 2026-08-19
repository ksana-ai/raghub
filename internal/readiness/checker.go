// Package readiness verifies the dependencies required by the API's advertised
// ingestion and retrieval paths.
package readiness

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

type Database interface {
	Ping(context.Context) error
}

type Embedder interface {
	Embed(context.Context, []string) ([][]float32, error)
}

type Checker struct {
	database Database
	embedder Embedder
	cacheTTL time.Duration

	mu        sync.Mutex
	checkedAt time.Time
	cachedErr error
}

func New(database Database, embedder Embedder, cacheTTL time.Duration) (*Checker, error) {
	if database == nil {
		return nil, errors.New("readiness database is required")
	}
	if embedder == nil {
		return nil, errors.New("readiness embedder is required")
	}
	if cacheTTL < 0 {
		return nil, errors.New("readiness cache TTL must not be negative")
	}
	return &Checker{database: database, embedder: embedder, cacheTTL: cacheTTL}, nil
}

func (c *Checker) Ping(ctx context.Context) error {
	if c == nil || c.database == nil || c.embedder == nil {
		return errors.New("readiness checker is not configured")
	}
	if err := c.database.Ping(ctx); err != nil {
		return fmt.Errorf("database readiness: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.checkedAt.IsZero() && time.Since(c.checkedAt) < c.cacheTTL {
		return c.cachedErr
	}
	_, err := c.embedder.Embed(ctx, []string{"raghub embedding readiness probe"})
	if ctx.Err() != nil {
		return fmt.Errorf("embedding readiness: %w", ctx.Err())
	}
	if err != nil {
		c.cachedErr = fmt.Errorf("embedding readiness: %w", err)
	} else {
		c.cachedErr = nil
	}
	c.checkedAt = time.Now()
	return c.cachedErr
}
