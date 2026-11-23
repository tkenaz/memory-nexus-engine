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

-- Update temporal trigger to skip chunks (if trigger exists)
-- Chunks inherit temporal context from parent, no need for separate links
DO $$
BEGIN
    -- Check if the trigger function exists and update it
    IF EXISTS (SELECT 1 FROM pg_proc WHERE proname = 'create_temporal_links') THEN
        CREATE OR REPLACE FUNCTION create_temporal_links()
        RETURNS TRIGGER AS $func$
        BEGIN
            -- Skip chunks — only create temporal links for parent/standalone memories
            IF NEW.parent_id IS NOT NULL THEN
                RETURN NEW;
            END IF;

            -- Link to recent memories (within 24h)
            INSERT INTO memory_links (from_id, to_id, link_type, weight, confidence)
            SELECT NEW.id, id, 'after', 0.7, 0.7
            FROM memories
            WHERE id != NEW.id
              AND parent_id IS NULL
              AND created_at > NOW() - INTERVAL '24 hours'
              AND created_at < NEW.created_at
            ORDER BY created_at DESC
            LIMIT 5
            ON CONFLICT DO NOTHING;

            RETURN NEW;
        END;
        $func$ LANGUAGE plpgsql;
    END IF;
END;
$$;
