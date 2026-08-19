package readiness

import (
	"context"
	"errors"
	"testing"
	"time"
)

type fakeDatabase struct {
	calls int
	err   error
}

func (f *fakeDatabase) Ping(context.Context) error {
	f.calls++
	return f.err
}

type fakeEmbedder struct {
	calls int
	err   error
}

func (f *fakeEmbedder) Embed(context.Context, []string) ([][]float32, error) {
	f.calls++
	return [][]float32{{1}}, f.err
}

func TestCheckerRequiresDatabaseAndEmbeddingReadiness(t *testing.T) {
	t.Parallel()
	databaseFailure := errors.New("database down")
	embeddingFailure := errors.New("model down")

	database := &fakeDatabase{err: databaseFailure}
	embedder := &fakeEmbedder{}
	checker, err := New(database, embedder, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if err := checker.Ping(context.Background()); !errors.Is(err, databaseFailure) {
		t.Fatalf("database error = %v", err)
	}
	if embedder.calls != 0 {
		t.Fatalf("embedder called %d times after database failure", embedder.calls)
	}

	database.err = nil
	embedder.err = embeddingFailure
	if err := checker.Ping(context.Background()); !errors.Is(err, embeddingFailure) {
		t.Fatalf("embedding error = %v", err)
	}
}

func TestCheckerCachesEmbeddingProbeButAlwaysPingsDatabase(t *testing.T) {
	t.Parallel()
	database := &fakeDatabase{}
	embedder := &fakeEmbedder{}
	checker, err := New(database, embedder, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if err := checker.Ping(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	if database.calls != 2 || embedder.calls != 1 {
		t.Fatalf("database calls=%d embedding calls=%d, want 2/1", database.calls, embedder.calls)
	}
}

func TestCheckerDoesNotCacheCallerCancellation(t *testing.T) {
	t.Parallel()
	database := &fakeDatabase{}
	embedder := &fakeEmbedder{err: context.Canceled}
	checker, err := New(database, embedder, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := checker.Ping(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled probe error = %v", err)
	}
	embedder.err = nil
	if err := checker.Ping(context.Background()); err != nil {
		t.Fatalf("fresh probe after cancellation = %v", err)
	}
	if embedder.calls != 2 {
		t.Fatalf("embedding calls=%d, want cancellation not cached", embedder.calls)
	}
}

func TestNewCheckerValidatesDependencies(t *testing.T) {
	t.Parallel()
	if _, err := New(nil, &fakeEmbedder{}, time.Second); err == nil {
		t.Fatal("nil database unexpectedly accepted")
	}
	if _, err := New(&fakeDatabase{}, nil, time.Second); err == nil {
		t.Fatal("nil embedder unexpectedly accepted")
	}
	if _, err := New(&fakeDatabase{}, &fakeEmbedder{}, -time.Second); err == nil {
		t.Fatal("negative cache TTL unexpectedly accepted")
	}
}
