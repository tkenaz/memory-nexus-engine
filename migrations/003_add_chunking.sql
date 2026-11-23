-- Migration 003: Add chunking support
-- parent_id links chunks back to their parent memory
-- chunk_index tracks ordering within a parent

-- Add columns
ALTER TABLE memories ADD COLUMN IF NOT EXISTS parent_id TEXT REFERENCES memories(id) ON DELETE CASCADE;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS chunk_index INTEGER;

-- Index for finding chunks by parent
CREATE INDEX IF NOT EXISTS idx_memories_parent_id ON memories(parent_id);

-- Partial index: only standalone/parent records (for get_recent, stats, etc.)
CREATE INDEX IF NOT EXISTS idx_memories_is_parent ON memories(created_at DESC) WHERE parent_id IS NULL;

-- The temporal link trigger (create_temporal_links) is defined once, in 002,
-- and already skips chunks.

