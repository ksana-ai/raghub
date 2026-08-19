CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embedding_profiles (
    profile_id       text        PRIMARY KEY,
    provider         text        NOT NULL CHECK (provider <> ''),
    model            text        NOT NULL CHECK (model <> ''),
    dimensions       integer     NOT NULL CHECK (dimensions = 1024),
    document_recipe  text        NOT NULL CHECK (document_recipe <> ''),
    query_recipe     text        NOT NULL CHECK (query_recipe <> ''),
    created_at       timestamptz NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    tenant_id       text         NOT NULL,
    chunk_id        text         NOT NULL,
    profile_id      text         NOT NULL,
    input_sha256    character(64) NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    embedding       vector(1024) NOT NULL CHECK (vector_dims(embedding) = 1024 AND vector_norm(embedding) > 0),
    created_at      timestamptz  NOT NULL DEFAULT clock_timestamp(),
    updated_at      timestamptz  NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (tenant_id, chunk_id, profile_id),
    FOREIGN KEY (tenant_id, chunk_id)
        REFERENCES chunks (tenant_id, chunk_id)
        ON DELETE CASCADE,
    FOREIGN KEY (profile_id)
        REFERENCES embedding_profiles (profile_id)
        ON DELETE RESTRICT
);

-- This B-tree narrows the authorization-scoped exact-search input. We
-- intentionally do not install HNSW in the exact Dense baseline: a shared ANN
-- index can lose filtered recall across tenants and ACLs.
CREATE INDEX IF NOT EXISTS chunk_embeddings_scope_idx
    ON chunk_embeddings (tenant_id, profile_id, chunk_id);
