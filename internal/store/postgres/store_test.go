package postgres

import (
	"math"
	"strings"
	"testing"
)

func TestVectorLiteralValidatesStoredCosineVector(t *testing.T) {
	t.Parallel()
	valid := make([]float32, storedEmbeddingDimensions)
	valid[0] = 0.5
	literal, err := vectorLiteral(valid, storedEmbeddingDimensions)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(literal, "[0.5,") || !strings.HasSuffix(literal, "]") {
		t.Fatalf("unexpected vector literal prefix/suffix: %.24s...", literal)
	}

	wrongSize := valid[:len(valid)-1]
	zero := make([]float32, storedEmbeddingDimensions)
	notFinite := append([]float32(nil), valid...)
	notFinite[1] = float32(math.NaN())
	for _, vector := range [][]float32{wrongSize, zero, notFinite} {
		if _, err := vectorLiteral(vector, storedEmbeddingDimensions); err == nil {
			t.Fatalf("invalid vector unexpectedly encoded (length=%d)", len(vector))
		}
	}
}
