ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS raw_content text;

UPDATE document_versions
SET raw_content = ''
WHERE raw_content IS NULL;

ALTER TABLE document_versions
    ALTER COLUMN raw_content SET NOT NULL;
