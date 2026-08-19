package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestParseSearchMode(t *testing.T) {
	t.Parallel()

	for _, value := range []string{"fts", "dense", "hybrid"} {
		if mode, err := parseSearchMode(value); err != nil || string(mode) != value {
			t.Fatalf("parseSearchMode(%q) = %q, %v", value, mode, err)
		}
	}
	if _, err := parseSearchMode("rerank"); err == nil {
		t.Fatal("parseSearchMode(rerank) error = nil")
	}
}

func TestRunRejectsUnknownModeBeforeExternalDependencies(t *testing.T) {
	t.Parallel()

	var stdout, stderr bytes.Buffer
	code := run(context.Background(), []string{"-mode", "rerank"}, func(string) string { return "" }, &stdout, &stderr, "raghub-eval -mode rerank")
	if code != 2 || !bytes.Contains(stderr.Bytes(), []byte("-mode must be")) {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
}

func TestRunValidatesDenseEnvironmentBeforeDatabase(t *testing.T) {
	t.Parallel()

	for _, mode := range []string{"dense", "hybrid"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			var stdout, stderr bytes.Buffer
			code := run(context.Background(), []string{"-mode", mode}, func(key string) string {
				if key == "RAGHUB_EMBEDDING_DIMENSIONS" {
					return "768"
				}
				return ""
			}, &stdout, &stderr, "raghub-eval -mode "+mode)
			if code != 2 || !bytes.Contains(stderr.Bytes(), []byte("must be 1024")) {
				t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
			}
		})
	}
}

func TestRunFTSDoesNotRequireEmbeddingConfiguration(t *testing.T) {
	t.Parallel()

	var stdout, stderr bytes.Buffer
	code := run(context.Background(), []string{"-mode", "fts"}, func(key string) string {
		if key == "RAGHUB_EMBEDDING_ENDPOINT" {
			return "not-a-url"
		}
		return ""
	}, &stdout, &stderr, "raghub-eval -mode fts")
	if code != 2 || !bytes.Contains(stderr.Bytes(), []byte("RAGHUB_DATABASE_URL is required")) {
		t.Fatalf("run() = %d, stderr=%q", code, stderr.String())
	}
}

func TestDefaultDatasetIsPreregisteredHybridV3(t *testing.T) {
	t.Parallel()

	if defaultDataset != "datasets/smoke/v3.json" {
		t.Fatalf("defaultDataset = %q, want hybrid v3", defaultDataset)
	}
}

func TestCommandStringPreservesArgvBoundaries(t *testing.T) {
	t.Parallel()

	got := commandString([]string{"raghub-eval", "-dataset", "fixtures/smoke set.json", "it's-valid"})
	want := `raghub-eval -dataset 'fixtures/smoke set.json' 'it'"'"'s-valid'`
	if got != want {
		t.Fatalf("commandString() = %q, want %q", got, want)
	}
}

func TestFormatCodeRevision(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		revision string
		modified bool
		want     string
	}{
		{name: "uncommitted repository", want: "uncommitted"},
		{name: "clean revision", revision: "deadbeef", want: "deadbeef"},
		{name: "dirty revision", revision: "deadbeef", modified: true, want: "deadbeef+dirty"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := formatCodeRevision(test.revision, test.modified); got != test.want {
				t.Fatalf("formatCodeRevision() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestWriteOutputReplacesFileAtomically(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(path, []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := writeOutput(path, []byte("new\n"), &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "new\n" {
		t.Fatalf("file = %q, want new manifest", data)
	}
}
