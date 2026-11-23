-- Tag catalog: embeddings of the tag vocabulary for the third search engine.
-- The query is matched against short labels first (the symmetric case where a
-- bi-encoder works well), then expanded into memories through their tags.
-- Tags are normalised with lower() at indexing and at match time.
-- Filled incrementally on store and in bulk by build_tag_index.py.

CREATE TABLE IF NOT EXISTS tag_embeddings (
    tag TEXT PRIMARY KEY,
    embedding vector(1024),
    memory_count INT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A vocabulary of a few thousand tags scans in milliseconds; no vector index needed.
