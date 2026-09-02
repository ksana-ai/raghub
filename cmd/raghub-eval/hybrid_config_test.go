package main

import (
	"testing"

	"github.com/ksana-ai/raghub/internal/retrieval"
)

func TestHybridManifestConfigPreregistersFusionProtocol(t *testing.T) {
	t.Parallel()

	config := hybridManifestConfig(denseSettings{}, retrieval.DefaultHybridConfig())
	if config["mode"] != "hybrid" || config["fusion"] != "reciprocal_rank_fusion" || config["branch_failure"] != "fail_closed" {
		t.Fatalf("hybrid protocol identity = %#v", config)
	}
	if config["rrf_k"] != 60 || config["fts_candidate_k"] != 20 || config["dense_candidate_k"] != 20 {
		t.Fatalf("hybrid preregistered parameters = %#v", config)
	}
	fts, ok := config["fts"].(map[string]any)
	if !ok || fts["mode"] != "fts" {
		t.Fatalf("hybrid FTS branch config = %#v", config["fts"])
	}
	dense, ok := config["dense"].(map[string]any)
	if !ok || dense["mode"] != "dense" {
		t.Fatalf("hybrid dense branch config = %#v", config["dense"])
	}
}
