package eval

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ksana-ai/raghub/internal/model"
)

const DatasetSchemaVersion = "raghub.eval.dataset/v1"

// Dataset is a versioned, self-contained retrieval benchmark. Documents and
// queries intentionally live in the same artifact so a report can identify the
// exact corpus snapshot by a single SHA-256 digest.
type Dataset struct {
	SchemaVersion string            `json:"schema_version"`
	Name          string            `json:"name"`
	Version       string            `json:"version"`
	Description   string            `json:"description,omitempty"`
	Documents     []DatasetDocument `json:"documents"`
	Queries       []QueryCase       `json:"queries"`
}

type DatasetDocument struct {
	TenantID          string          `json:"tenant_id"`
	DocumentID        string          `json:"document_id"`
	Title             string          `json:"title"`
	SourceURI         string          `json:"source_uri"`
	Content           string          `json:"content"`
	AllowedPrincipals []string        `json:"allowed_principals,omitempty"`
	Metadata          json.RawMessage `json:"metadata,omitempty"`
}

func (d DatasetDocument) documentInput() model.DocumentInput {
	metadata := d.Metadata
	if len(metadata) == 0 {
		metadata = json.RawMessage(`{}`)
	}
	return model.DocumentInput{
		TenantID:          d.TenantID,
		ID:                d.DocumentID,
		Title:             d.Title,
		SourceURI:         d.SourceURI,
		Content:           d.Content,
		AllowedPrincipals: append([]string(nil), d.AllowedPrincipals...),
		Metadata:          append(json.RawMessage(nil), metadata...),
	}
}

type QueryCase struct {
	ID                string   `json:"id"`
	Category          string   `json:"category"`
	TenantID          string   `json:"tenant_id"`
	PrincipalID       string   `json:"principal_id,omitempty"`
	Query             string   `json:"query"`
	GoldChunkIDs      []string `json:"gold_chunk_ids"`
	ForbiddenChunkIDs []string `json:"forbidden_chunk_ids,omitempty"`
}

// LoadedDataset preserves the digest of the exact input bytes rather than a
// re-marshaled representation of the dataset.
type LoadedDataset struct {
	Dataset Dataset
	SHA256  string
}

func LoadDataset(path string) (LoadedDataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return LoadedDataset{}, fmt.Errorf("read dataset: %w", err)
	}
	return ParseDataset(data)
}

func ParseDataset(data []byte) (LoadedDataset, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	var dataset Dataset
	if err := decoder.Decode(&dataset); err != nil {
		return LoadedDataset{}, fmt.Errorf("decode dataset: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return LoadedDataset{}, errors.New("decode dataset: input must contain exactly one JSON object")
		}
		return LoadedDataset{}, fmt.Errorf("decode dataset trailing data: %w", err)
	}
	if err := validateDataset(dataset); err != nil {
		return LoadedDataset{}, err
	}

	digest := sha256.Sum256(data)
	return LoadedDataset{
		Dataset: dataset,
		SHA256:  hex.EncodeToString(digest[:]),
	}, nil
}

func validateDataset(dataset Dataset) error {
	if dataset.SchemaVersion != DatasetSchemaVersion {
		return fmt.Errorf("validate dataset: schema_version must be %q", DatasetSchemaVersion)
	}
	if strings.TrimSpace(dataset.Name) == "" {
		return errors.New("validate dataset: name is required")
	}
	if strings.TrimSpace(dataset.Version) == "" {
		return errors.New("validate dataset: version is required")
	}
	if len(dataset.Documents) == 0 {
		return errors.New("validate dataset: at least one document is required")
	}
	if len(dataset.Queries) == 0 {
		return errors.New("validate dataset: at least one query is required")
	}

	documentIDs := make(map[string]struct{}, len(dataset.Documents))
	for i, document := range dataset.Documents {
		prefix := fmt.Sprintf("validate dataset: documents[%d]", i)
		if strings.TrimSpace(document.TenantID) == "" || strings.TrimSpace(document.DocumentID) == "" {
			return fmt.Errorf("%s tenant_id and document_id are required", prefix)
		}
		if strings.TrimSpace(document.Title) == "" || strings.TrimSpace(document.Content) == "" {
			return fmt.Errorf("%s title and content are required", prefix)
		}
		if len(document.Metadata) > 0 && !json.Valid(document.Metadata) {
			return fmt.Errorf("%s metadata must be valid JSON", prefix)
		}
		// Chunk IDs are derived from document_id without tenant_id in v1, so
		// document IDs must be globally unique inside one dataset artifact.
		if _, exists := documentIDs[document.DocumentID]; exists {
			return fmt.Errorf("%s duplicates global document_id %q", prefix, document.DocumentID)
		}
		documentIDs[document.DocumentID] = struct{}{}
	}

	queryIDs := make(map[string]struct{}, len(dataset.Queries))
	for i, query := range dataset.Queries {
		prefix := fmt.Sprintf("validate dataset: queries[%d]", i)
		if strings.TrimSpace(query.ID) == "" || strings.TrimSpace(query.Category) == "" {
			return fmt.Errorf("%s id and category are required", prefix)
		}
		if _, exists := queryIDs[query.ID]; exists {
			return fmt.Errorf("%s duplicates query id %q", prefix, query.ID)
		}
		queryIDs[query.ID] = struct{}{}
		if strings.TrimSpace(query.TenantID) == "" || strings.TrimSpace(query.Query) == "" {
			return fmt.Errorf("%s tenant_id and query are required", prefix)
		}
		if len(query.GoldChunkIDs) == 0 {
			return fmt.Errorf("%s gold_chunk_ids must not be empty", prefix)
		}
		if hasBlankOrDuplicate(query.GoldChunkIDs) {
			return fmt.Errorf("%s gold_chunk_ids must be non-empty and unique", prefix)
		}
		if hasBlankOrDuplicate(query.ForbiddenChunkIDs) {
			return fmt.Errorf("%s forbidden_chunk_ids must be non-empty and unique", prefix)
		}
		gold := make(map[string]struct{}, len(query.GoldChunkIDs))
		for _, chunkID := range query.GoldChunkIDs {
			gold[chunkID] = struct{}{}
		}
		for _, chunkID := range query.ForbiddenChunkIDs {
			if _, exists := gold[chunkID]; exists {
				return fmt.Errorf("%s chunk %q cannot be both gold and forbidden", prefix, chunkID)
			}
		}
	}
	return nil
}

func hasBlankOrDuplicate(values []string) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if strings.TrimSpace(value) == "" {
			return true
		}
		if _, exists := seen[value]; exists {
			return true
		}
		seen[value] = struct{}{}
	}
	return false
}
