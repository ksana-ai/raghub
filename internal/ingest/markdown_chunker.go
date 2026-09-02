package ingest

import (
	"errors"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ksana-ai/raghub/internal/model"
)

const (
	defaultMaxRunes     = 1200
	defaultOverlapRunes = 120
)

var markdownHeadingPattern = regexp.MustCompile(`^(#{1,6})\s+(.+?)\s*#*\s*$`)

// MarkdownChunker keeps heading ancestry in every chunk and splits oversized
// sections at paragraph boundaries. The same input and configuration always
// produce the same ordered chunks.
type MarkdownChunker struct {
	MaxRunes     int
	OverlapRunes int
}

func NewMarkdownChunker(maxRunes, overlapRunes int) (*MarkdownChunker, error) {
	if maxRunes <= 0 {
		maxRunes = defaultMaxRunes
	}
	if overlapRunes < 0 || overlapRunes >= maxRunes {
		return nil, errors.New("overlap runes must be non-negative and smaller than max runes")
	}
	return &MarkdownChunker{MaxRunes: maxRunes, OverlapRunes: overlapRunes}, nil
}

func (c *MarkdownChunker) Chunk(content string) ([]model.ChunkDraft, error) {
	if c == nil || c.MaxRunes <= 0 || c.OverlapRunes < 0 || c.OverlapRunes >= c.MaxRunes {
		return nil, errors.New("invalid markdown chunker configuration")
	}

	sections := parseSections(strings.ReplaceAll(content, "\r\n", "\n"))
	chunks := make([]model.ChunkDraft, 0, len(sections))
	for _, section := range sections {
		for _, text := range splitSection(section.text, c.MaxRunes, c.OverlapRunes) {
			text = strings.TrimSpace(text)
			if text == "" {
				continue
			}
			chunks = append(chunks, model.ChunkDraft{
				Ordinal:     len(chunks),
				HeadingPath: append([]string(nil), section.headingPath...),
				RawText:     text,
				IndexedText: text,
				TokenCount:  estimateTokens(text),
			})
		}
	}
	return chunks, nil
}

type markdownSection struct {
	headingPath []string
	text        string
}

func parseSections(content string) []markdownSection {
	var (
		headings []string
		body     []string
		result   []markdownSection
	)
	flush := func() {
		text := strings.TrimSpace(strings.Join(body, "\n"))
		if text != "" {
			result = append(result, markdownSection{
				headingPath: append([]string(nil), headings...),
				text:        text,
			})
		}
		body = body[:0]
	}

	for _, line := range strings.Split(content, "\n") {
		match := markdownHeadingPattern.FindStringSubmatch(strings.TrimSpace(line))
		if match == nil {
			body = append(body, line)
			continue
		}
		flush()
		level := len(match[1])
		if len(headings) >= level {
			headings = headings[:level-1]
		}
		for len(headings) < level-1 {
			headings = append(headings, "")
		}
		headings = append(headings, strings.TrimSpace(match[2]))
	}
	flush()
	return result
}

func splitSection(text string, maxRunes, overlapRunes int) []string {
	if utf8.RuneCountInString(text) <= maxRunes {
		return []string{text}
	}

	paragraphs := splitParagraphs(text)
	var chunks []string
	var current string
	for _, paragraph := range paragraphs {
		if utf8.RuneCountInString(paragraph) > maxRunes {
			if strings.TrimSpace(current) != "" {
				chunks = append(chunks, strings.TrimSpace(current))
				current = overlapSuffix(current, overlapRunes)
			}
			for _, part := range splitRunes(paragraph, maxRunes, overlapRunes) {
				chunks = append(chunks, part)
			}
			current = ""
			continue
		}

		candidate := paragraph
		if current != "" {
			candidate = current + "\n\n" + paragraph
		}
		if utf8.RuneCountInString(candidate) <= maxRunes {
			current = candidate
			continue
		}

		chunks = append(chunks, strings.TrimSpace(current))
		prefix := overlapSuffix(current, overlapRunes)
		current = joinWithBoundedOverlap(prefix, paragraph, maxRunes)
	}
	if strings.TrimSpace(current) != "" {
		chunks = append(chunks, strings.TrimSpace(current))
	}
	return chunks
}

func joinWithBoundedOverlap(prefix, text string, maxRunes int) string {
	prefix = strings.TrimSpace(prefix)
	text = strings.TrimSpace(text)
	if prefix == "" {
		return text
	}

	textRunes := []rune(text)
	prefixRunes := []rune(prefix)
	// Two runes are reserved for the paragraph separator. When the next
	// paragraph already fills the chunk, preserving the bound takes precedence
	// over overlap.
	available := maxRunes - len(textRunes) - 2
	if available <= 0 {
		return text
	}
	if len(prefixRunes) > available {
		prefixRunes = prefixRunes[len(prefixRunes)-available:]
	}
	return strings.TrimSpace(string(prefixRunes)) + "\n\n" + text
}

func splitParagraphs(text string) []string {
	parts := regexp.MustCompile(`\n\s*\n`).Split(text, -1)
	result := make([]string, 0, len(parts))
	for _, part := range parts {
		if part = strings.TrimSpace(part); part != "" {
			result = append(result, part)
		}
	}
	return result
}

func splitRunes(text string, maxRunes, overlapRunes int) []string {
	runes := []rune(text)
	step := maxRunes - overlapRunes
	result := make([]string, 0, (len(runes)+step-1)/step)
	for start := 0; start < len(runes); start += step {
		end := min(start+maxRunes, len(runes))
		result = append(result, strings.TrimSpace(string(runes[start:end])))
		if end == len(runes) {
			break
		}
	}
	return result
}

func overlapSuffix(text string, overlapRunes int) string {
	if overlapRunes == 0 {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= overlapRunes {
		return strings.TrimSpace(text)
	}
	return strings.TrimSpace(string(runes[len(runes)-overlapRunes:]))
}

func estimateTokens(text string) int {
	count := 0
	inWord := false
	for _, r := range text {
		if unicode.Is(unicode.Han, r) {
			count++
			inWord = false
			continue
		}
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			if !inWord {
				count++
				inWord = true
			}
			continue
		}
		inWord = false
	}
	return count
}
