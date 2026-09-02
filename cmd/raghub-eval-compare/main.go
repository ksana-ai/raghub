package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	evalrun "github.com/ksana-ai/raghub/internal/eval"
)

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("raghub-eval-compare", flag.ContinueOnError)
	flags.SetOutput(stderr)
	baselinePath := flags.String("baseline", "", "baseline eval manifest path")
	candidatePath := flags.String("candidate", "", "candidate eval manifest path")
	ftsPath := flags.String("fts", "", "FTS eval manifest path for a three-way comparison")
	densePath := flags.String("dense", "", "Dense eval manifest path for a three-way comparison")
	hybridPath := flags.String("hybrid", "", "Hybrid eval manifest path for a three-way comparison")
	candidateDiagnosis := flags.Bool("candidate-diagnosis", false, "diagnose Hybrid final misses against FTS/Dense candidate-depth manifests")
	outputPath := flags.String("output", "-", "comparison output path, or - for stdout")
	if err := flags.Parse(args); err != nil {
		return 2
	}
	if flags.NArg() != 0 {
		fmt.Fprintln(stderr, "raghub-eval-compare: unexpected positional arguments")
		return 2
	}
	pairRequested := *baselinePath != "" || *candidatePath != ""
	threeWayRequested := *ftsPath != "" || *densePath != "" || *hybridPath != ""
	if *candidateDiagnosis && !threeWayRequested {
		fmt.Fprintln(stderr, "raghub-eval-compare: -candidate-diagnosis requires -fts, -dense, and -hybrid")
		return 2
	}
	if pairRequested && threeWayRequested {
		fmt.Fprintln(stderr, "raghub-eval-compare: pairwise and three-way flags cannot be combined")
		return 2
	}
	if pairRequested && (*baselinePath == "" || *candidatePath == "") {
		fmt.Fprintln(stderr, "raghub-eval-compare: -baseline and -candidate are both required for pairwise comparison")
		return 2
	}
	if threeWayRequested && (*ftsPath == "" || *densePath == "" || *hybridPath == "") {
		fmt.Fprintln(stderr, "raghub-eval-compare: -fts, -dense, and -hybrid are all required for three-way comparison")
		return 2
	}
	if !pairRequested && !threeWayRequested {
		fmt.Fprintln(stderr, "raghub-eval-compare: provide either -baseline/-candidate or -fts/-dense/-hybrid")
		return 2
	}
	inputPaths := []string{*baselinePath, *candidatePath, *ftsPath, *densePath, *hybridPath}
	if *outputPath != "-" {
		for _, inputPath := range inputPaths {
			if inputPath != "" && samePath(*outputPath, inputPath) {
				fmt.Fprintln(stderr, "raghub-eval-compare: -output must not overwrite an input manifest")
				return 2
			}
		}
	}

	var data []byte
	if pairRequested {
		baseline, err := loadManifest("baseline", *baselinePath)
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
			return 1
		}
		candidate, err := loadManifest("candidate", *candidatePath)
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
			return 1
		}
		comparison, err := evalrun.CompareManifests(baseline, candidate)
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
			return 1
		}
		data, err = evalrun.MarshalComparison(comparison)
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
			return 1
		}
	} else {
		fts, err := loadManifest("FTS", *ftsPath)
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
			return 1
		}
		dense, err := loadManifest("Dense", *densePath)
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
			return 1
		}
		hybrid, err := loadManifest("Hybrid", *hybridPath)
		if err != nil {
			fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
			return 1
		}
		if *candidateDiagnosis {
			diagnosis, err := evalrun.DiagnoseCandidateCoverage(fts, dense, hybrid)
			if err != nil {
				fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
				return 1
			}
			data, err = evalrun.MarshalCandidateDiagnosis(diagnosis)
			if err != nil {
				fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
				return 1
			}
		} else {
			comparison, err := evalrun.CompareThreeManifests(fts, dense, hybrid)
			if err != nil {
				fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
				return 1
			}
			data, err = evalrun.MarshalThreeWayComparison(comparison)
			if err != nil {
				fmt.Fprintf(stderr, "raghub-eval-compare: %v\n", err)
				return 1
			}
		}
	}

	if err := writeOutput(*outputPath, data, stdout); err != nil {
		fmt.Fprintf(stderr, "raghub-eval-compare: write comparison: %v\n", err)
		return 1
	}
	return 0
}

func loadManifest(label, path string) (evalrun.Manifest, error) {
	manifest, err := evalrun.LoadManifest(path)
	if err != nil {
		return evalrun.Manifest{}, fmt.Errorf("%s: %w", label, err)
	}
	return manifest, nil
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
