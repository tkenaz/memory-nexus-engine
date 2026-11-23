-- Upgrade path: bring memories.embedding to vector(1024) (BGE-M3).
--
-- Fresh installs already get vector(1024) from 001, and this migration is a no-op
-- for them. It matters for databases created by older snapshots, where 001 made
-- vector(384). init_db.py re-runs every migration on each call, so this file must
-- be safe to repeat.
--
-- If the column is not 1024 and already holds vectors, the migration stops instead
-- of discarding them. Vectors of another model cannot be cast; they must be
-- recomputed:
--   UPDATE memories SET embedding = NULL;   -- then re-run: python init_db.py
--   python reembed_all.py && python build_tag_index.py --rebuild
--
-- Note: switching from e5-large to BGE-M3 keeps the dimension (1024) but changes
-- the vector space. This migration cannot detect that; re-embed in that case too.

DO $$
DECLARE
    current_dim int;
    populated   bigint;
BEGIN
    SELECT a.atttypmod INTO current_dim
    FROM pg_attribute a
    WHERE a.attrelid = 'memories'::regclass
      AND a.attname = 'embedding'
      AND NOT a.attisdropped;

    IF current_dim = 1024 THEN
        RETURN;
    END IF;

    SELECT count(*) INTO populated FROM memories WHERE embedding IS NOT NULL;
    IF populated > 0 THEN
        RAISE EXCEPTION
            'memories.embedding is vector(%) and holds % vectors; they cannot be cast to vector(1024). Run UPDATE memories SET embedding = NULL, re-run init_db.py, then reembed_all.py and build_tag_index.py --rebuild.',
            current_dim, populated;
    END IF;

    -- Views and the vector index depend on the column type.
    DROP VIEW IF EXISTS memory_stats_basic;
    DROP VIEW IF EXISTS memory_stats;
    DROP INDEX IF EXISTS idx_memories_embedding;

    ALTER TABLE memories ALTER COLUMN embedding TYPE vector(1024);

    CREATE INDEX IF NOT EXISTS idx_memories_embedding
        ON memories USING hnsw (embedding vector_cosine_ops);
END $$;

CREATE OR REPLACE VIEW memory_stats_basic AS
SELECT
    COUNT(*) AS total_memories,
    COUNT(embedding) AS memories_with_embeddings,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') AS memories_last_24h,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') AS memories_last_week,
    pg_size_pretty(pg_total_relation_size('memories')) AS table_size
FROM memories;

CREATE OR REPLACE VIEW memory_stats AS
SELECT
    (SELECT COUNT(*) FROM memories) AS total_memories,
    (SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL) AS memories_with_embeddings,
    (SELECT COUNT(*) FROM memory_links) AS total_links,
    (SELECT COUNT(DISTINCT link_type) FROM memory_links) AS link_types_used,
    (SELECT COUNT(*) FROM memory_patterns) AS patterns_detected,
    (SELECT COUNT(*) FROM memory_patterns WHERE promoted = true) AS promoted_patterns,
    (SELECT json_object_agg(link_type, cnt)
       FROM (SELECT link_type, COUNT(*) AS cnt FROM memory_links GROUP BY link_type) t) AS links_by_type;
