package eval

import (
	"math"
	"slices"
)

type RankingMetrics struct {
	RecallAtK  float64 `json:"recall_at_k"`
	HitRateAtK float64 `json:"hit_rate_at_k"`
	MRR        float64 `json:"mrr"`
	NDCGAtK    float64 `json:"ndcg_at_k"`
}

// EvaluateRanking calculates standard recall (not the success/hit-rate variant
// sometimes also called recall), hit rate, reciprocal rank, and binary nDCG.
// Duplicate result IDs are ignored after their first occurrence.
func EvaluateRanking(rankedIDs, goldIDs []string, k int) RankingMetrics {
	if k <= 0 || len(goldIDs) == 0 {
		return RankingMetrics{}
	}

	gold := make(map[string]struct{}, len(goldIDs))
	for _, id := range goldIDs {
		gold[id] = struct{}{}
	}

	seen := make(map[string]struct{}, min(k, len(rankedIDs)))
	relevantHits := 0
	dcg := 0.0
	reciprocalRank := 0.0
	uniqueRank := 0
	for _, id := range rankedIDs {
		if _, duplicate := seen[id]; duplicate {
			continue
		}
		if uniqueRank == k {
			break
		}
		seen[id] = struct{}{}
		uniqueRank++
		if _, relevant := gold[id]; !relevant {
			continue
		}
		relevantHits++
		if reciprocalRank == 0 {
			reciprocalRank = 1 / float64(uniqueRank)
		}
		dcg += 1 / math.Log2(float64(uniqueRank)+1)
	}

	idealHits := min(len(gold), k)
	idcg := 0.0
	for rank := 1; rank <= idealHits; rank++ {
		idcg += 1 / math.Log2(float64(rank)+1)
	}

	hitRate := 0.0
	if relevantHits > 0 {
		hitRate = 1
	}
	ndcg := 0.0
	if idcg > 0 {
		ndcg = dcg / idcg
	}
	return RankingMetrics{
		RecallAtK:  float64(relevantHits) / float64(len(gold)),
		HitRateAtK: hitRate,
		MRR:        reciprocalRank,
		NDCGAtK:    ndcg,
	}
}

func meanMetrics(values []RankingMetrics) RankingMetrics {
	if len(values) == 0 {
		return RankingMetrics{}
	}
	var result RankingMetrics
	for _, value := range values {
		result.RecallAtK += value.RecallAtK
		result.HitRateAtK += value.HitRateAtK
		result.MRR += value.MRR
		result.NDCGAtK += value.NDCGAtK
	}
	count := float64(len(values))
	result.RecallAtK /= count
	result.HitRateAtK /= count
	result.MRR /= count
	result.NDCGAtK /= count
	return result
}

type LatencyPercentiles struct {
	P50MS float64 `json:"p50_ms"`
	P95MS float64 `json:"p95_ms"`
}

func latencyPercentiles(values []float64) LatencyPercentiles {
	if len(values) == 0 {
		return LatencyPercentiles{}
	}
	sorted := append([]float64(nil), values...)
	slices.Sort(sorted)
	return LatencyPercentiles{
		P50MS: nearestRank(sorted, 0.50),
		P95MS: nearestRank(sorted, 0.95),
	}
}

// nearestRank uses the nearest-rank percentile definition: ceil(p*N), with
// ranks starting at one.
func nearestRank(sorted []float64, percentile float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	rank := int(math.Ceil(percentile*float64(len(sorted)))) - 1
	rank = max(0, min(rank, len(sorted)-1))
	return sorted[rank]
}
