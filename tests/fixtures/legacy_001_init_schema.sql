-- Migration: Initialize Memory Nexus schema
-- Description: Creates base tables, extensions, and indices for memory storage
-- Date: 2025-01-24

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

-- UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Vector similarity search (pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- MAIN TABLES
-- =============================================================================

-- Memories table: stores memory content, embeddings, and metadata
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    content TEXT NOT NULL,

    -- Semantic embedding for similarity search
    -- Dimension: 384 for MiniLM, 768 for BERT, 1024 for e5-large
    -- Change dimension in .env EMBEDDING_DIMENSION if using different model
    embedding vector(384),

    -- Flexible metadata storage (type, tags, source, etc)
    metadata JSONB DEFAULT '{}',

    -- SASH-F structure (added in migration 002)
    sashf JSONB DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- INDICES FOR PERFORMANCE
-- =============================================================================

-- Vector similarity search index (IVFFlat for cosine similarity)
-- lists=100 is good for ~10K-100K memories, adjust for larger datasets
CREATE INDEX IF NOT EXISTS idx_memories_embedding
ON memories USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Metadata search (GIN index for JSONB)
CREATE INDEX IF NOT EXISTS idx_memories_metadata
ON memories USING GIN (metadata);

-- Chronological queries
CREATE INDEX IF NOT EXISTS idx_memories_created
ON memories (created_at DESC);

-- Full-text search on content
CREATE INDEX IF NOT EXISTS idx_memories_content_search
ON memories USING GIN (to_tsvector('english', content));

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_memories_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_memories_updated_at
BEFORE UPDATE ON memories
FOR EACH ROW
EXECUTE FUNCTION update_memories_updated_at();

-- =============================================================================
-- UTILITY VIEWS
-- =============================================================================

-- Basic stats view
CREATE OR REPLACE VIEW memory_stats_basic AS
SELECT
    COUNT(*) as total_memories,
    COUNT(embedding) as memories_with_embeddings,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as memories_last_24h,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as memories_last_week,
    pg_size_pretty(pg_total_relation_size('memories')) as table_size
FROM memories;

-- =============================================================================
-- NOTES
-- =============================================================================

-- After running this migration:
-- 1. Run migration 002 to add SASH-F graph structure
-- 2. Check index creation: SELECT * FROM pg_stat_user_indexes WHERE relname = 'memories';
-- 3. For large datasets (>100K memories), consider adjusting ivfflat lists parameter
--
-- To rebuild vector index after changing lists:
--   DROP INDEX idx_memories_embedding;
--   CREATE INDEX idx_memories_embedding ON memories
--   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1000);
