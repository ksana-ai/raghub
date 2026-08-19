CREATE OR REPLACE FUNCTION raghub_reject_embedding_profile_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'embedding profiles are immutable; create a new profile_id';
END;
$$;

DROP TRIGGER IF EXISTS embedding_profiles_immutable ON embedding_profiles;
CREATE TRIGGER embedding_profiles_immutable
BEFORE UPDATE OR DELETE ON embedding_profiles
FOR EACH ROW
EXECUTE FUNCTION raghub_reject_embedding_profile_mutation();

CREATE OR REPLACE FUNCTION raghub_reject_chunk_embedding_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'chunk embeddings are immutable; create a new profile_id';
END;
$$;

DROP TRIGGER IF EXISTS chunk_embeddings_immutable ON chunk_embeddings;
CREATE TRIGGER chunk_embeddings_immutable
BEFORE UPDATE ON chunk_embeddings
FOR EACH ROW
EXECUTE FUNCTION raghub_reject_chunk_embedding_update();
