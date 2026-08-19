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
	"math"
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

// Embedder converts the exact indexed_text values into one immutable embedding
// profile. Network calls happen before the store opens its transaction.
type Embedder interface {
	Profile() model.EmbeddingProfile
	Embed(ctx context.Context, inputs []string) ([][]float32, error)
}

type Service struct {
	store    Store
	chunker  Chunker
	embedder Embedder
}

func NewService(store Store, chunker Chunker) *Service {
	return &Service{store: store, chunker: chunker}
}

func NewServiceWithEmbedder(store Store, chunker Chunker, embedder Embedder) *Service {
	return &Service{store: store, chunker: chunker, embedder: embedder}
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
	if s.embedder != nil {
		if err := embedChunks(ctx, s.embedder, chunks); err != nil {
			return model.IngestResult{}, err
		}
	}

	return s.store.SaveDocumentVersion(ctx, document, fingerprint(document, chunks), chunks)
}

func embedChunks(ctx context.Context, embedder Embedder, chunks []model.ChunkDraft) error {
	profile := embedder.Profile()
	if err := validateEmbeddingProfile(profile); err != nil {
		return fmt.Errorf("embed chunks: invalid profile: %w", err)
	}
	inputs := make([]string, len(chunks))
	for index := range chunks {
		inputs[index] = chunks[index].IndexedText
	}
	vectors, err := embedder.Embed(ctx, inputs)
	if err != nil {
		return fmt.Errorf("embed chunks with profile %q: %w", profile.ProfileID, err)
	}
	if len(vectors) != len(chunks) {
		return fmt.Errorf("embed chunks with profile %q: received %d vectors for %d chunks", profile.ProfileID, len(vectors), len(chunks))
	}
	for index, vector := range vectors {
		if len(vector) != profile.Dimensions {
			return fmt.Errorf("embed chunk %d with profile %q: received %d dimensions, want %d", index, profile.ProfileID, len(vector), profile.Dimensions)
		}
		var normSquared float64
		for dimension, value := range vector {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("embed chunk %d with profile %q: dimension %d is not finite", index, profile.ProfileID, dimension)
			}
			normSquared += float64(value) * float64(value)
		}
		if normSquared == 0 {
			return fmt.Errorf("embed chunk %d with profile %q: cosine vector must not be zero", index, profile.ProfileID)
		}
		chunks[index].Embedding = &model.EmbeddingDraft{
			Profile: profile,
			Values:  append([]float32(nil), vector...),
		}
	}
	return nil
}

func validateEmbeddingProfile(profile model.EmbeddingProfile) error {
	if !identifierPattern.MatchString(strings.TrimSpace(profile.ProfileID)) {
		return errors.New("profile ID must be a 1-128 character safe identifier")
	}
	if strings.TrimSpace(profile.Provider) == "" || len(profile.Provider) > 128 {
		return errors.New("provider must be between 1 and 128 bytes")
	}
	if strings.TrimSpace(profile.Model) == "" || len(profile.Model) > 256 {
		return errors.New("model must be between 1 and 256 bytes")
	}
	if profile.Dimensions <= 0 {
		return errors.New("dimensions must be positive")
	}
	if strings.TrimSpace(profile.DocumentRecipe) == "" || len(profile.DocumentRecipe) > 256 {
		return errors.New("document recipe must be between 1 and 256 bytes")
	}
	if strings.TrimSpace(profile.QueryRecipe) == "" || len(profile.QueryRecipe) > 256 {
		return errors.New("query recipe must be between 1 and 256 bytes")
	}
	return nil
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
