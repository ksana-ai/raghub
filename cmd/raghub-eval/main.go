package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"

	"github.com/jackc/pgx/v5/pgxpool"

	embeddingopenai "github.com/ksana-ai/raghub/internal/embedding/openai"
	evalrun "github.com/ksana-ai/raghub/internal/eval"
	"github.com/ksana-ai/raghub/internal/ingest"
	"github.com/ksana-ai/raghub/internal/model"
	"github.com/ksana-ai/raghub/internal/retrieval"
	postgresstore "github.com/ksana-ai/raghub/internal/store/postgres"
)

const (
	defaultDataset      = "datasets/smoke/v3.json"
	defaultChunkRunes   = 1200
	defaultOverlapRunes = 120
)

func main() {
	args := os.Args[1:]
	logicalArgv := append([]string{"raghub-eval"}, args...)
	os.Exit(run(context.Background(), args, os.Getenv, os.Stdout, os.Stderr, commandString(logicalArgv)))
}

func run(ctx context.Context, args []string, getenv func(string) string, stdout, stderr io.Writer, command string) int {
	flags := flag.NewFlagSet("raghub-eval", flag.ContinueOnError)
	flags.SetOutput(stderr)
	datasetPath := flags.String("dataset", defaultDataset, "path to a versioned JSON evaluation dataset")
	outputPath := flags.String("output", "-", "manifest output path, or - for stdout")
	topK := flags.Int("top-k", 5, "number of ranked chunks to evaluate")
	modeFlag := flags.String("mode", string(model.SearchModeFTS), "retrieval mode: fts, dense, or hybrid")
	migrate := flags.Bool("migrate", false, "apply database migrations before evaluation")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(stderr, "raghub-eval: unexpected positional arguments")
		return 2
	}
	if *topK < 1 || *topK > 50 {
		fmt.Fprintln(stderr, "raghub-eval: -top-k must be between 1 and 50")
		return 2
	}
	searchMode, err := parseSearchMode(*modeFlag)
	if err != nil {
		fmt.Fprintf(stderr, "raghub-eval: %v\n", err)
		return 2
	}
	var (
		denseConfig denseSettings
		denseClient *embeddingopenai.Client
	)
	if searchMode == model.SearchModeDense || searchMode == model.SearchModeHybrid {
		denseConfig, err = loadDenseSettings(getenv)
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval: %v\n", err)
			return 2
		}
		denseClient, err = embeddingopenai.NewClient(denseConfig.clientConfig())
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval: configure embedding client: %v\n", err)
			return 2
		}
	}

	databaseURL := strings.TrimSpace(getenv("RAGHUB_DATABASE_URL"))
	if databaseURL == "" {
		fmt.Fprintln(stderr, "raghub-eval: RAGHUB_DATABASE_URL is required")
		return 2
	}
	loaded, err := evalrun.LoadDataset(*datasetPath)
	if err != nil {
		fmt.Fprintf(stderr, "raghub-eval: %v\n", err)
		return 1
	}

	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		fmt.Fprintf(stderr, "raghub-eval: connect to PostgreSQL: %v\n", err)
		return 1
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		fmt.Fprintf(stderr, "raghub-eval: ping PostgreSQL: %v\n", err)
		return 1
	}
	if *migrate {
		if err := postgresstore.ApplyMigrations(ctx, pool); err != nil {
			fmt.Fprintf(stderr, "raghub-eval: apply migrations: %v\n", err)
			return 1
		}
	}

	chunker, err := ingest.NewMarkdownChunker(defaultChunkRunes, defaultOverlapRunes)
	if err != nil {
		fmt.Fprintf(stderr, "raghub-eval: configure chunker: %v\n", err)
		return 1
	}
	store := postgresstore.New(pool)
	var (
		ingestor        evalrun.Ingestor
		searcher        evalrun.Searcher
		retrieverName   string
		retrieverConfig map[string]any
	)
	switch searchMode {
	case model.SearchModeDense:
		ingestor = ingest.NewServiceWithEmbedder(store, chunker, denseClient)
		searcher = retrieval.NewServiceWithDense(store, store, denseClient)
		retrieverName = "postgres_dense"
		retrieverConfig = denseManifestConfig(denseConfig)
	case model.SearchModeHybrid:
		hybridConfig := retrieval.DefaultHybridConfig()
		ingestor = ingest.NewServiceWithEmbedder(store, chunker, denseClient)
		searcher, err = retrieval.NewServiceWithHybrid(store, store, denseClient, hybridConfig)
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval: configure hybrid retriever: %v\n", err)
			return 2
		}
		retrieverName = "postgres_hybrid_rrf"
		retrieverConfig = hybridManifestConfig(denseConfig, hybridConfig)
	case model.SearchModeFTS:
		ingestor = ingest.NewService(store, chunker)
		searcher = retrieval.NewService(store)
		retrieverName = "postgres_fts"
		retrieverConfig = ftsManifestConfig()
	}

	databaseVersion := ""
	if err := pool.QueryRow(ctx, "SHOW server_version").Scan(&databaseVersion); err != nil {
		fmt.Fprintf(stderr, "raghub-eval: read PostgreSQL version: %v\n", err)
		return 1
	}
	vectorExtensionVersion := ""
	if err := pool.QueryRow(ctx, `
SELECT COALESCE(
    (SELECT extversion FROM pg_extension WHERE extname = 'vector'),
    ''
)`).Scan(&vectorExtensionVersion); err != nil {
		fmt.Fprintf(stderr, "raghub-eval: read pgvector extension version: %v\n", err)
		return 1
	}
	manifest, runErr := evalrun.NewRunner(ingestor, searcher, store).Run(ctx, loaded, evalrun.Options{
		TopK:                   *topK,
		SearchMode:             searchMode,
		RetrieverName:          retrieverName,
		RetrieverConfig:        retrieverConfig,
		DatabaseVersion:        databaseVersion,
		VectorExtensionVersion: vectorExtensionVersion,
		CodeRevision:           codeRevision(),
		Command:                command,
	})
	data, marshalErr := evalrun.MarshalManifest(manifest)
	if marshalErr != nil {
		fmt.Fprintf(stderr, "raghub-eval: %v\n", marshalErr)
		return 1
	}
	if err := writeOutput(*outputPath, data, stdout); err != nil {
		fmt.Fprintf(stderr, "raghub-eval: write manifest: %v\n", err)
		return 1
	}
	if runErr != nil {
		fmt.Fprintf(stderr, "raghub-eval: evaluation failed: %v\n", runErr)
		return 1
	}
	return 0
}

func parseSearchMode(value string) (model.SearchMode, error) {
	mode := model.SearchMode(strings.TrimSpace(value))
	if mode != model.SearchModeFTS && mode != model.SearchModeDense && mode != model.SearchModeHybrid {
		return "", fmt.Errorf(
			"-mode must be %q, %q, or %q",
			model.SearchModeFTS,
			model.SearchModeDense,
			model.SearchModeHybrid,
		)
	}
	return mode, nil
}

func writeOutput(path string, data []byte, stdout io.Writer) (err error) {
	if path == "-" {
		_, err := stdout.Write(data)
		return err
	}

	temporary, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer func() {
		_ = temporary.Close()
		if err != nil {
			_ = os.Remove(temporaryPath)
		}
	}()
	if err = temporary.Chmod(0o644); err != nil {
		return err
	}
	if _, err = temporary.Write(data); err != nil {
		return err
	}
	if err = temporary.Sync(); err != nil {
		return err
	}
	if err = temporary.Close(); err != nil {
		return err
	}
	if err = os.Rename(temporaryPath, path); err != nil {
		return err
	}
	return nil
}

func codeRevision() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "uncommitted"
	}
	var revision string
	modified := false
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value == "true"
		}
	}
	return formatCodeRevision(revision, modified)
}

func formatCodeRevision(revision string, modified bool) string {
	if revision == "" {
		return "uncommitted"
	}
	if modified {
		return revision + "+dirty"
	}
	return revision
}

// commandString reconstructs the process argv as a shell-readable command.
// The original user quoting is not available after process startup.
func commandString(args []string) string {
	quoted := make([]string, 0, len(args))
	for _, arg := range args {
		quoted = append(quoted, quoteCommandArg(arg))
	}
	return strings.Join(quoted, " ")
}

func quoteCommandArg(arg string) string {
	const safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-"
	if arg != "" && strings.IndexFunc(arg, func(r rune) bool { return !strings.ContainsRune(safe, r) }) == -1 {
		return arg
	}
	return "'" + strings.ReplaceAll(arg, "'", `'"'"'`) + "'"
}
