CREATE TABLE IF NOT EXISTS documents (
    tenant_id       text        NOT NULL,
    document_id     text        NOT NULL,
    current_version integer     NOT NULL DEFAULT 0 CHECK (current_version >= 0),
    created_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, document_id)
);

CREATE TABLE IF NOT EXISTS document_versions (
    tenant_id          text        NOT NULL,
    document_id        text        NOT NULL,
    version            integer     NOT NULL CHECK (version > 0),
    fingerprint        text        NOT NULL CHECK (fingerprint <> ''),
    title              text        NOT NULL,
    source_uri         text        NOT NULL,
    allowed_principals text[]      NOT NULL DEFAULT '{}'::text[],
    metadata           jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at         timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, document_id, version),
    FOREIGN KEY (tenant_id, document_id)
        REFERENCES documents (tenant_id, document_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS document_versions_fingerprint_idx
    ON document_versions (tenant_id, document_id, version, fingerprint);

CREATE TABLE IF NOT EXISTS chunks (
    tenant_id       text        NOT NULL,
    chunk_id        text        NOT NULL,
    document_id     text        NOT NULL,
    document_version integer    NOT NULL CHECK (document_version > 0),
    ordinal         integer     NOT NULL CHECK (ordinal >= 0),
    title           text        NOT NULL,
    heading_path    text[]      NOT NULL DEFAULT '{}'::text[],
    heading_text    text        NOT NULL DEFAULT '',
    raw_text        text        NOT NULL,
    indexed_text    text        NOT NULL,
    token_count     integer     NOT NULL CHECK (token_count >= 0),
    search_vector   tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple'::regconfig, coalesce(title, '')), 'A') ||
        setweight(to_tsvector('simple'::regconfig, coalesce(heading_text, '')), 'B') ||
        setweight(to_tsvector('simple'::regconfig, coalesce(indexed_text, '')), 'D')
    ) STORED,
    created_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, chunk_id),
    UNIQUE (tenant_id, document_id, document_version, ordinal),
    FOREIGN KEY (tenant_id, document_id, document_version)
        REFERENCES document_versions (tenant_id, document_id, version)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS chunks_search_vector_idx
    ON chunks USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS chunks_current_version_idx
    ON chunks (tenant_id, document_id, document_version, ordinal);
