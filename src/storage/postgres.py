"""PostgreSQL storage with pgvector"""
import json
from typing import Dict, List, Optional, Any
from uuid import UUID
import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from ..models.memory import Memory, MemoryType
from config.settings import settings
import structlog

logger = structlog.get_logger()


class PostgresStorage:
    """PostgreSQL storage backend with pgvector"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize connection pool and create tables"""
        self.pool = await asyncpg.create_pool(
            settings.PG_DSN,
            min_size=10,
            max_size=20,
            command_timeout=60,
            init=self._init_connection
        )
        
        # Create tables
        await self._create_tables()
    
    async def _init_connection(self, conn):
        """Initialize each connection with pgvector"""
        await register_vector(conn)
        
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            
    async def _create_tables(self):
        """Create tables if not exist"""
        async with self.pool.acquire() as conn:
            # Enable extensions
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            
            # Check if table exists
            table_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'memories'
                )
            """)
            
            if not table_exists:
                # Create memories table with proper embedding dimension
                # Using string formatting for DDL (safe since it's our own constant)
                await conn.execute(f'''
                    CREATE TABLE memories (
                        id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
                        content TEXT NOT NULL,
                        embedding vector({settings.EMBEDDING_DIMENSION}),
                        metadata JSONB DEFAULT '{{}}',
                        sashf JSONB DEFAULT '{{}}',
                        parent_id TEXT REFERENCES memories(id) ON DELETE CASCADE,
                        chunk_index INTEGER,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                ''')
                
                logger.info("Created memories table")
            
            # Create indexes if not exist
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_embedding 
                ON memories USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_metadata 
                ON memories USING GIN (metadata)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories (created_at DESC)
            ''')

            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_parent_id
                ON memories(parent_id)
            ''')

            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_is_parent
                ON memories(created_at DESC) WHERE parent_id IS NULL
            ''')

            # Check and create memory_links table if needed
            links_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'memory_links'
                )
            """)
            
            if not links_exists:
                await conn.execute('''
                    CREATE TABLE memory_links (
                        from_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                        to_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                        link_type VARCHAR(50) NOT NULL,
                        weight FLOAT DEFAULT 1.0,
                        confidence FLOAT DEFAULT 0.8,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (from_id, to_id, link_type)
                    )
                ''')
                
                await conn.execute('''
                    CREATE INDEX idx_links_from ON memory_links(from_id);
                    CREATE INDEX idx_links_to ON memory_links(to_id);
                    CREATE INDEX idx_links_type ON memory_links(link_type);
                ''')
                
                logger.info("Created memory_links table")
            
            # Check and create patterns table if needed
            patterns_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'memory_patterns'
                )
            """)
            
            if not patterns_exists:
                await conn.execute('''
                    CREATE TABLE memory_patterns (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        pattern_key TEXT UNIQUE NOT NULL,
                        occurrences INTEGER DEFAULT 1,
                        confidence FLOAT DEFAULT 0.5,
                        actors TEXT[] DEFAULT '{}',
                        act TEXT,
                        manner TEXT,
                        affect_bucket TEXT,
                        example_ids TEXT[] DEFAULT '{}',
                        first_seen TIMESTAMP DEFAULT NOW(),
                        last_seen TIMESTAMP DEFAULT NOW(),
                        promoted BOOLEAN DEFAULT FALSE,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                ''')
                
                logger.info("Created memory_patterns table")
    
    async def store(
        self,
        memory: Memory
    ) -> str:
        """Store a memory"""
        # Generate ID
        import hashlib
        import time
        memory_id = f"mem_{int(time.time() * 1000)}_{hashlib.md5(memory.content.encode()).hexdigest()[:8]}"

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO memories (id, content, embedding, metadata, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            memory_id,
            memory.content,
            memory.embedding,
            json.dumps(memory.metadata) if memory.metadata else '{}',
            memory.created_at)

            return memory_id
    
    async def search(
        self,
        embedding: List[float],
        limit: int = 10,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Search memories by embedding similarity"""
        min_sim = min_similarity or settings.MIN_SIMILARITY
        
        async with self.pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT 
                    id,
                    content,
                    metadata,
                    1.0 - (embedding <=> $1) as similarity,
                    created_at
                FROM memories
                WHERE embedding IS NOT NULL
                AND 1.0 - (embedding <=> $1) > $2
                ORDER BY embedding <=> $1
                LIMIT $3
            """, embedding, min_sim, limit)
            
            return [dict(r) for r in results]
    
    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory by ID"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT * FROM memories WHERE id = $1
            """, memory_id)
            
            return dict(result) if result else None
    
    async def update(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update memory"""
        # Build update query dynamically
        set_clauses = []
        values = []
        param_count = 1
        
        for key, value in updates.items():
            if key in ['content', 'embedding', 'metadata']:
                set_clauses.append(f"{key} = ${param_count}")
                if key == 'metadata':
                    values.append(json.dumps(value))
                else:
                    values.append(value)
                param_count += 1
        
        if not set_clauses:
            return False
        
        values.append(memory_id)
        query = f"""
            UPDATE memories 
            SET {', '.join(set_clauses)}, updated_at = NOW()
            WHERE id = ${param_count}
        """
        
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *values)
            return result != "UPDATE 0"
    
    async def delete(self, memory_id: str) -> bool:
        """Delete memory"""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM memories WHERE id = $1
            """, memory_id)
            
            return result != "DELETE 0"
