-- Fix memory_patterns.example_ids to use text[] instead of uuid[]
ALTER TABLE memory_patterns
ALTER COLUMN example_ids TYPE text[]
USING example_ids::text[];

-- Also update the type checking in patterns
COMMENT ON TABLE memory_patterns IS 'Updated to use text[] for example_ids to match memories.id type';