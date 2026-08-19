package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	evalrun "raghub/internal/eval"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("raghub-eval-compare", flag.ContinueOnError)
	flags.SetOutput(stderr)
	baselinePath := flags.String("baseline", "", "baseline eval manifest path")
	candidatePath := flags.String("candidate", "", "candidate eval manifest path")
	outputPath := flags.String("output", "-", "comparison output path, or - for stdout")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(stderr, "raghub-eval-compare: unexpected positional arguments")
		return 2
	}
	if *baselinePath == "" || *candidatePath == "" {
		fmt.Fprintln(stderr, "raghub-eval-compare: -baseline and -candidate are required")
		return 2
	}
	if *outputPath != "-" && (samePath(*outputPath, *baselinePath) || samePath(*outputPath, *candidatePath)) {
		fmt.Fprintln(stderr, "raghub-eval-compare: -output must not overwrite an input manifest")
		return 2
	}

	baseline, err := evalrun.LoadManifest(*baselinePath)
	if err != nil {
		fmt.Fprintf(stderr, "raghub-eval-compare: baseline: %v\n", err)
		return 1
	}
	candidate, err := evalrun.LoadManifest(*candidatePath)
	if err != nil {
		fmt.Fprintf(stderr, "raghub-eval-compare: candidate: %v\n", err)
		return 1
	}
	comparison, err := evalrun.CompareManifests(baseline, candidate)
	if err != nil {
		fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
		return 1
	}
	data, err := evalrun.MarshalComparison(comparison)
	if err != nil {
		fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
		return 1
	}
	if err := writeOutput(*outputPath, data, stdout); err != nil {
		fmt.Fprintf(stderr, "raghub-eval-compare: write comparison: %v\n", err)
		return 1
	}
	return 0
}

func samePath(first, second string) bool {
	firstAbsolute, firstErr := filepath.Abs(first)
	secondAbsolute, secondErr := filepath.Abs(second)
	if firstErr != nil || secondErr != nil {
		return filepath.Clean(first) == filepath.Clean(second)
	}
	return filepath.Clean(firstAbsolute) == filepath.Clean(secondAbsolute)
}

func writeOutput(path string, data []byte, stdout io.Writer) (err error) {
	if path == "-" {
		_, err = stdout.Write(data)
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
