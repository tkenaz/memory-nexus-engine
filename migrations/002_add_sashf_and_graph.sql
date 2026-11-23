-- Migration: Add SASH-F structure and graph capabilities to Memory Nexus
-- Date: 2025-10-03

-- 1. Add SASH-F structure to memories table
ALTER TABLE memories 
ADD COLUMN IF NOT EXISTS sashf JSONB DEFAULT '{}';

-- 2. Add indices for SASH-F components
CREATE INDEX IF NOT EXISTS idx_memories_sashf_actors 
ON memories USING GIN ((sashf->'actors'));

CREATE INDEX IF NOT EXISTS idx_memories_sashf_act 
ON memories USING GIN ((sashf->'act'));

CREATE INDEX IF NOT EXISTS idx_memories_sashf_manner 
ON memories USING GIN ((sashf->'manner'));

CREATE INDEX IF NOT EXISTS idx_memories_sashf_affect 
ON memories USING GIN ((sashf->'affect'));

CREATE INDEX IF NOT EXISTS idx_memories_sashf_qualia 
ON memories USING GIN ((sashf->'qualia'));

CREATE INDEX IF NOT EXISTS idx_memories_sashf_tags
ON memories USING GIN ((sashf->'tags'));

CREATE INDEX IF NOT EXISTS idx_memories_sashf_time_bucket
ON memories ((sashf->>'time_bucket'));

-- 3. Create memory links table for graph structure
CREATE TABLE IF NOT EXISTS memory_links (
    from_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    link_type VARCHAR(50) NOT NULL,
    weight FLOAT DEFAULT 1.0 CHECK (weight >= 0.0 AND weight <= 1.0),
    confidence FLOAT DEFAULT 0.8 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (from_id, to_id, link_type)
);

-- 4. Indices for efficient graph traversal
CREATE INDEX IF NOT EXISTS idx_links_from ON memory_links(from_id);
CREATE INDEX IF NOT EXISTS idx_links_to ON memory_links(to_id);
CREATE INDEX IF NOT EXISTS idx_links_type ON memory_links(link_type);
CREATE INDEX IF NOT EXISTS idx_links_weight ON memory_links(weight DESC);
CREATE INDEX IF NOT EXISTS idx_links_created ON memory_links(created_at DESC);

-- 5. Create patterns table for recurring patterns
CREATE TABLE IF NOT EXISTS memory_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_key TEXT UNIQUE NOT NULL,
    occurrences INTEGER DEFAULT 1,
    confidence FLOAT DEFAULT 0.5 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    
    -- Pattern components (normalized)
    actors TEXT[] DEFAULT '{}',
    act TEXT,
    manner TEXT,
    affect_bucket TEXT,
    
    -- Examples and timeline
    example_ids UUID[] DEFAULT '{}',
    first_seen TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP DEFAULT NOW(),
    
    -- Meta
    promoted BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- 6. Indices for patterns
CREATE INDEX IF NOT EXISTS idx_patterns_key ON memory_patterns(pattern_key);
CREATE INDEX IF NOT EXISTS idx_patterns_occurrences ON memory_patterns(occurrences DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_promoted ON memory_patterns(promoted);
CREATE INDEX IF NOT EXISTS idx_patterns_actors ON memory_patterns USING GIN(actors);

-- 7. Create function for automatic link creation
CREATE OR REPLACE FUNCTION create_temporal_links()
RETURNS TRIGGER AS $$
BEGIN
    -- Link a new memory to up to five memories from the 24 hours before it:
    -- (from_id = new, to_id = earlier, 'after') reads "new is after earlier".
    -- Graph expansion walks from_id -> to_id, i.e. from a hit to what preceded it.
    -- The window is relative to NEW.created_at, so imported historical memories
    -- are linked as well. Chunks are skipped: they belong to their parent, and
    -- graph expansion only walks parents. (parent_id comes from migration 003;
    -- plpgsql resolves columns at run time, after all migrations have run.)
    IF NEW.parent_id IS NOT NULL THEN
        RETURN NEW;
    END IF;

    INSERT INTO memory_links (from_id, to_id, link_type, weight, confidence)
    SELECT NEW.id, m.id, 'after',
           CASE
               WHEN NEW.created_at - m.created_at < INTERVAL '1 hour' THEN 0.9
               WHEN NEW.created_at - m.created_at < INTERVAL '6 hours' THEN 0.7
               ELSE 0.5
           END,
           0.7
    FROM memories m
    WHERE m.id != NEW.id
      AND m.parent_id IS NULL
      AND m.created_at > NEW.created_at - INTERVAL '24 hours'
      AND m.created_at < NEW.created_at
    ORDER BY m.created_at DESC
    LIMIT 5
    ON CONFLICT DO NOTHING;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 8. Create trigger for automatic temporal linking
DROP TRIGGER IF EXISTS auto_temporal_links ON memories;
CREATE TRIGGER auto_temporal_links
AFTER INSERT ON memories
FOR EACH ROW
EXECUTE FUNCTION create_temporal_links();

-- 9. Function to find similar memories for semantic linking
CREATE OR REPLACE FUNCTION find_and_link_similar(
    memory_id TEXT,
    similarity_threshold FLOAT DEFAULT 0.85
)
RETURNS INTEGER AS $$
DECLARE
    linked_count INTEGER := 0;
    similar_memory RECORD;
BEGIN
    FOR similar_memory IN
        SELECT 
            m2.id,
            1.0 - (m1.embedding <=> m2.embedding) as similarity
        FROM memories m1, memories m2
        WHERE m1.id = memory_id
        AND m2.id != memory_id
        AND m1.embedding IS NOT NULL
        AND m2.embedding IS NOT NULL
        AND 1.0 - (m1.embedding <=> m2.embedding) > similarity_threshold
        ORDER BY similarity DESC
        LIMIT 10
    LOOP
        INSERT INTO memory_links (from_id, to_id, link_type, weight, confidence)
        VALUES (memory_id, similar_memory.id, 'similar', similar_memory.similarity, similar_memory.similarity)
        ON CONFLICT DO NOTHING;
        
        linked_count := linked_count + 1;
    END LOOP;
    
    RETURN linked_count;
END;
$$ LANGUAGE plpgsql;

-- 10. View for easy querying of linked memories
CREATE OR REPLACE VIEW memory_graph AS
SELECT 
    m1.id as from_id,
    m1.content as from_content,
    m1.sashf as from_sashf,
    ml.link_type,
    ml.weight,
    ml.confidence,
    m2.id as to_id,
    m2.content as to_content,
    m2.sashf as to_sashf,
    ml.created_at as link_created
FROM memory_links ml
JOIN memories m1 ON ml.from_id = m1.id
JOIN memories m2 ON ml.to_id = m2.id;

-- 11. Function for graph traversal (BFS)
CREATE OR REPLACE FUNCTION traverse_memory_graph(
    start_id TEXT,
    max_depth INTEGER DEFAULT 2,
    link_types TEXT[] DEFAULT ARRAY['similar', 'after', 'caused_by']
)
RETURNS TABLE(
    memory_id TEXT,
    content TEXT,
    depth INTEGER,
    path_weight FLOAT,
    path TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE graph_traverse AS (
        -- Base case: starting memory
        SELECT 
            m.id::TEXT as memory_id,
            m.content,
            0 as depth,
            1.0::FLOAT as path_weight,
            ARRAY[m.id::TEXT] as path
        FROM memories m
        WHERE m.id = start_id
        
        UNION ALL
        
        -- Recursive case: follow links
        SELECT 
            ml.to_id as memory_id,
            m.content,
            gt.depth + 1,
            gt.path_weight * ml.weight * 0.8,  -- Decay factor
            gt.path || ml.to_id
        FROM graph_traverse gt
        JOIN memory_links ml ON gt.memory_id = ml.from_id
        JOIN memories m ON ml.to_id = m.id
        WHERE gt.depth < max_depth
        AND ml.link_type = ANY(link_types)
        AND NOT ml.to_id = ANY(gt.path)  -- Prevent cycles
    )
    SELECT DISTINCT ON (graph_traverse.memory_id)
        graph_traverse.memory_id,
        graph_traverse.content,
        graph_traverse.depth,
        graph_traverse.path_weight,
        graph_traverse.path
    FROM graph_traverse
    ORDER BY graph_traverse.memory_id, graph_traverse.path_weight DESC;
END;
$$ LANGUAGE plpgsql;

-- 12. Add update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_memory_links_updated_at ON memory_links;
CREATE TRIGGER update_memory_links_updated_at
BEFORE UPDATE ON memory_links
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS update_memory_patterns_updated_at ON memory_patterns;
CREATE TRIGGER update_memory_patterns_updated_at
BEFORE UPDATE ON memory_patterns
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- 13. Stats view
CREATE OR REPLACE VIEW memory_stats AS
SELECT 
    (SELECT COUNT(*) FROM memories) as total_memories,
    (SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL) as memories_with_embeddings,
    (SELECT COUNT(*) FROM memory_links) as total_links,
    (SELECT COUNT(DISTINCT link_type) FROM memory_links) as link_types_used,
    (SELECT COUNT(*) FROM memory_patterns) as patterns_detected,
    (SELECT COUNT(*) FROM memory_patterns WHERE promoted = true) as promoted_patterns,
    (SELECT json_object_agg(link_type, cnt) 
     FROM (SELECT link_type, COUNT(*) as cnt FROM memory_links GROUP BY link_type) t) as links_by_type;
