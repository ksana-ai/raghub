package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

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
