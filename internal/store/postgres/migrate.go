package postgres

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"

	"raghub/migrations"
)

const createMigrationsTableSQL = `
CREATE TABLE IF NOT EXISTS raghub_schema_migrations (
    version    text        PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
)`

// ApplyMigrations installs every embedded forward migration exactly once.
// The transaction-scoped advisory lock serializes concurrent process startup.
func ApplyMigrations(ctx context.Context, pool *pgxpool.Pool) (err error) {
	if pool == nil {
		return errors.New("apply migrations: nil postgres pool")
	}

	tx, err := pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin migration transaction: %w", err)
	}
	defer func() { _ = tx.Rollback(context.WithoutCancel(ctx)) }()

	if _, err = tx.Exec(ctx, `SELECT pg_advisory_xact_lock(hashtext('raghub:schema-migrations'))`); err != nil {
		return fmt.Errorf("lock schema migrations: %w", err)
	}
	if _, err = tx.Exec(ctx, createMigrationsTableSQL); err != nil {
		return fmt.Errorf("create schema migrations table: %w", err)
	}

	entries, err := migrations.Files.ReadDir(".")
	if err != nil {
		return fmt.Errorf("list embedded migrations: %w", err)
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".up.sql") {
			names = append(names, entry.Name())
		}
	}
	sort.Strings(names)

	for _, name := range names {
		var applied bool
		if err = tx.QueryRow(ctx,
			`SELECT EXISTS (SELECT 1 FROM raghub_schema_migrations WHERE version = $1)`,
			name,
		).Scan(&applied); err != nil {
			return fmt.Errorf("check migration %s: %w", name, err)
		}
		if applied {
			continue
		}

		script, readErr := migrations.Files.ReadFile(name)
		if readErr != nil {
			return fmt.Errorf("read migration %s: %w", name, readErr)
		}
		if _, err = tx.Exec(ctx, string(script)); err != nil {
			return fmt.Errorf("execute migration %s: %w", name, err)
		}
		if _, err = tx.Exec(ctx,
			`INSERT INTO raghub_schema_migrations (version) VALUES ($1)`,
			name,
		); err != nil {
			return fmt.Errorf("record migration %s: %w", name, err)
		}
	}

	if err = tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit schema migrations: %w", err)
	}
	return nil
}
