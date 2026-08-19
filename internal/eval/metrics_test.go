package eval

import (
	"math"
	"testing"
)

func TestEvaluateRanking(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		ranked []string
		gold   []string
		k      int
		want   RankingMetrics
	}{
		{
			name:   "multiple gold documents use standard recall",
			ranked: []string{"a", "x", "b"},
			gold:   []string{"a", "b", "c"},
			k:      3,
			want: RankingMetrics{
				RecallAtK:  2.0 / 3.0,
				HitRateAtK: 1,
				MRR:        1,
				NDCGAtK:    1.5 / (1 + 1/math.Log2(3) + 0.5),
			},
		},
		{
			name:   "no relevant result",
			ranked: []string{"x", "y"},
			gold:   []string{"a"},
			k:      2,
			want:   RankingMetrics{},
		},
		{
			name:   "duplicates do not consume rank positions",
			ranked: []string{"x", "x", "a"},
			gold:   []string{"a"},
			k:      2,
			want: RankingMetrics{
				RecallAtK:  1,
				HitRateAtK: 1,
				MRR:        0.5,
				NDCGAtK:    1 / math.Log2(3),
			},
		},
		{
			name:   "invalid k returns zero values",
			ranked: []string{"a"},
			gold:   []string{"a"},
			k:      0,
			want:   RankingMetrics{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got := EvaluateRanking(test.ranked, test.gold, test.k)
			assertClose(t, "recall", got.RecallAtK, test.want.RecallAtK)
			assertClose(t, "hit rate", got.HitRateAtK, test.want.HitRateAtK)
			assertClose(t, "mrr", got.MRR, test.want.MRR)
			assertClose(t, "ndcg", got.NDCGAtK, test.want.NDCGAtK)
		})
	}
}

func TestLatencyPercentilesUseNearestRank(t *testing.T) {
	t.Parallel()

	got := latencyPercentiles([]float64{100, 10, 50, 20, 30})
	if got.P50MS != 30 {
		t.Fatalf("p50 = %v, want 30", got.P50MS)
	}
	if got.P95MS != 100 {
		t.Fatalf("p95 = %v, want 100", got.P95MS)
	}
}

func assertClose(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("%s = %.12f, want %.12f", name, got, want)
	}
}
