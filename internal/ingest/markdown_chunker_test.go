package ingest

import (
	"reflect"
	"strings"
	"testing"
)

func TestMarkdownChunkerPreservesHeadingPathAndRawText(t *testing.T) {
	chunker, err := NewMarkdownChunker(200, 20)
	if err != nil {
		t.Fatal(err)
	}

	chunks, err := chunker.Chunk("# Operations\n\nIntro.\n\n## Deploy\n\nUse blue green deployment.")
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunk count = %d, want 2", len(chunks))
	}
	if got, want := chunks[1].HeadingPath, []string{"Operations", "Deploy"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("heading path = %#v, want %#v", got, want)
	}
	if chunks[1].RawText != "Use blue green deployment." || chunks[1].IndexedText != chunks[1].RawText {
		t.Fatalf("unexpected raw/indexed text: %#v", chunks[1])
	}
}

func TestMarkdownChunkerIsDeterministicAndBounded(t *testing.T) {
	chunker, err := NewMarkdownChunker(40, 5)
	if err != nil {
		t.Fatal(err)
	}
	input := "# Long\n\n" + strings.Repeat("abcdef", 20)

	first, err := chunker.Chunk(input)
	if err != nil {
		t.Fatal(err)
	}
	second, err := chunker.Chunk(input)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("chunking is not deterministic:\nfirst=%#v\nsecond=%#v", first, second)
	}
	if len(first) < 2 {
		t.Fatalf("chunk count = %d, want at least 2", len(first))
	}
	for _, chunk := range first {
		if got := len([]rune(chunk.RawText)); got > 40 {
			t.Fatalf("chunk %d has %d runes, want <= 40", chunk.Ordinal, got)
		}
	}
}

func TestMarkdownChunkerTrimsOverlapToKeepParagraphChunkBounded(t *testing.T) {
	chunker, err := NewMarkdownChunker(40, 10)
	if err != nil {
		t.Fatal(err)
	}
	input := strings.Repeat("a", 35) + "\n\n" + strings.Repeat("b", 35)

	chunks, err := chunker.Chunk(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunk count = %d, want 2", len(chunks))
	}
	if got := len([]rune(chunks[1].RawText)); got > 40 {
		t.Fatalf("second chunk has %d runes, want <= 40: %q", got, chunks[1].RawText)
	}
}

func TestNewMarkdownChunkerRejectsInvalidOverlap(t *testing.T) {
	if _, err := NewMarkdownChunker(10, 10); err == nil {
		t.Fatal("expected invalid overlap error")
	}
}
