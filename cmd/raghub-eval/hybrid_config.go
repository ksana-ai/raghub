package main

import (
	"github.com/ksana-ai/raghub/internal/model"
	"github.com/ksana-ai/raghub/internal/retrieval"
)

const hybridBranchFailure = "fail_closed"

func ftsManifestConfig() map[string]any {
	return map[string]any{
		"mode":                string(model.SearchModeFTS),
		"backend":             "postgresql",
		"fts_config":          "simple",
		"query_parser":        "plainto_tsquery",
		"chunker":             "markdown",
		"chunk_max_runes":     defaultChunkRunes,
		"chunk_overlap_runes": defaultOverlapRunes,
	}
}

func denseManifestConfig(settings denseSettings) map[string]any {
	config := settings.manifestConfig()
	config["mode"] = string(model.SearchModeDense)
	config["chunker"] = "markdown"
	config["chunk_max_runes"] = defaultChunkRunes
	config["chunk_overlap_runes"] = defaultOverlapRunes
	return config
}

func hybridManifestConfig(settings denseSettings, config retrieval.HybridConfig) map[string]any {
	return map[string]any{
		"mode":              string(model.SearchModeHybrid),
		"backend":           "postgresql",
		"fusion":            "reciprocal_rank_fusion",
		"rrf_k":             config.RRFK,
		"fts_candidate_k":   config.FTSCandidateDepth,
		"dense_candidate_k": config.DenseCandidateDepth,
		"branch_failure":    hybridBranchFailure,
		"fts":               ftsManifestConfig(),
		"dense":             denseManifestConfig(settings),
	}
}
