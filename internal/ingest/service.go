package ingest

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"raghub/internal/model"
)

const (
	maxDocumentBytes = 5 << 20
	maxMetadataBytes = 64 << 10
)

var identifierPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$`)

var ErrInvalidInput = errors.New("invalid ingestion input")

// Store persists a complete document version atomically and makes it active.
type Store interface {
	SaveDocumentVersion(ctx context.Context, document model.DocumentInput, fingerprint string, chunks []model.ChunkDraft) (model.IngestResult, error)
}

// Chunker converts source text into deterministic, ordered chunks.
type Chunker interface {
	Chunk(content string) ([]model.ChunkDraft, error)
}

type Service struct {
	store   Store
	chunker Chunker
}

func NewService(store Store, chunker Chunker) *Service {
	return &Service{store: store, chunker: chunker}
}

func (s *Service) Ingest(ctx context.Context, document model.DocumentInput) (model.IngestResult, error) {
	document, err := normalizeAndValidate(document)
	if err != nil {
		return model.IngestResult{}, fmt.Errorf("%w: %v", ErrInvalidInput, err)
	}

	chunks, err := s.chunker.Chunk(document.Content)
	if err != nil {
		return model.IngestResult{}, fmt.Errorf("chunk document: %w", err)
	}
	if len(chunks) == 0 {
		return model.IngestResult{}, fmt.Errorf("%w: document produced no chunks", ErrInvalidInput)
	}

	return s.store.SaveDocumentVersion(ctx, document, fingerprint(document, chunks), chunks)
}

func normalizeAndValidate(document model.DocumentInput) (model.DocumentInput, error) {
	document.TenantID = strings.TrimSpace(document.TenantID)
	document.ID = strings.TrimSpace(document.ID)
	document.Title = strings.TrimSpace(document.Title)
	document.SourceURI = strings.TrimSpace(document.SourceURI)

	if !identifierPattern.MatchString(document.TenantID) {
		return model.DocumentInput{}, errors.New("tenant ID must be 1-128 safe identifier characters")
	}
	if document.ID == "" {
		id, err := randomID()
		if err != nil {
			return model.DocumentInput{}, fmt.Errorf("generate document ID: %w", err)
		}
		document.ID = id
	}
	if !identifierPattern.MatchString(document.ID) {
		return model.DocumentInput{}, errors.New("document ID must be 1-128 safe identifier characters")
	}
	if document.Title == "" || len(document.Title) > 512 {
		return model.DocumentInput{}, errors.New("title must be between 1 and 512 bytes")
	}
	if len(document.SourceURI) > 2048 {
		return model.DocumentInput{}, errors.New("source URI exceeds 2048 bytes")
	}
	if strings.TrimSpace(document.Content) == "" {
		return model.DocumentInput{}, errors.New("content is required")
	}
	if len(document.Content) > maxDocumentBytes {
		return model.DocumentInput{}, fmt.Errorf("content exceeds %d bytes", maxDocumentBytes)
	}

	principals := append([]string(nil), document.AllowedPrincipals...)
	for i := range principals {
		principals[i] = strings.TrimSpace(principals[i])
		if !identifierPattern.MatchString(principals[i]) {
			return model.DocumentInput{}, fmt.Errorf("allowed principal %q is invalid", principals[i])
		}
	}
	slices.Sort(principals)
	principals = slices.Compact(principals)
	document.AllowedPrincipals = principals

	if len(document.Metadata) == 0 {
		document.Metadata = json.RawMessage(`{}`)
	}
	if len(document.Metadata) > maxMetadataBytes || !json.Valid(document.Metadata) {
		return model.DocumentInput{}, errors.New("metadata must be valid JSON no larger than 64 KiB")
	}
	var metadata any
	if err := json.Unmarshal(document.Metadata, &metadata); err != nil {
		return model.DocumentInput{}, fmt.Errorf("decode metadata: %w", err)
	}
	canonicalMetadata, err := json.Marshal(metadata)
	if err != nil {
		return model.DocumentInput{}, fmt.Errorf("canonicalize metadata: %w", err)
	}
	document.Metadata = canonicalMetadata

	return document, nil
}

func fingerprint(document model.DocumentInput, chunks []model.ChunkDraft) string {
	h := sha256.New()
	for _, part := range []string{
		document.Title,
		document.SourceURI,
		document.Content,
		strings.Join(document.AllowedPrincipals, "\x00"),
		string(document.Metadata),
	} {
		writeFingerprintPart(h, []byte(part))
	}
	var count [8]byte
	binary.BigEndian.PutUint64(count[:], uint64(len(chunks)))
	writeFingerprintPart(h, count[:])
	for _, chunk := range chunks {
		var number [8]byte
		binary.BigEndian.PutUint64(number[:], uint64(chunk.Ordinal))
		writeFingerprintPart(h, number[:])
		binary.BigEndian.PutUint64(number[:], uint64(len(chunk.HeadingPath)))
		writeFingerprintPart(h, number[:])
		for _, heading := range chunk.HeadingPath {
			writeFingerprintPart(h, []byte(heading))
		}
		writeFingerprintPart(h, []byte(chunk.RawText))
		writeFingerprintPart(h, []byte(chunk.IndexedText))
		binary.BigEndian.PutUint64(number[:], uint64(chunk.TokenCount))
		writeFingerprintPart(h, number[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

type hashWriter interface {
	Write([]byte) (int, error)
}

func writeFingerprintPart(writer hashWriter, part []byte) {
	var length [8]byte
	binary.BigEndian.PutUint64(length[:], uint64(len(part)))
	_, _ = writer.Write(length[:])
	_, _ = writer.Write(part)
}

func randomID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	value[6] = (value[6] & 0x0f) | 0x40
	value[8] = (value[8] & 0x3f) | 0x80
	return fmt.Sprintf("%x-%x-%x-%x-%x", value[0:4], value[4:6], value[6:8], value[8:10], value[10:16]), nil
}
