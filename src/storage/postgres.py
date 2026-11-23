"""PostgreSQL storage with pgvector"""
import json
from typing import Dict, List, Optional, Any
from uuid import UUID
import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from ..models.memory import Memory, MemoryType
from config.settings import settings
from .migrations import apply_migrations
import sys

import structlog

# MCP talks JSON-RPC over stdout; anything else printed there breaks the protocol.
# structlog's default logger writes to stdout, so route it to stderr explicitly.
logger = structlog.wrap_logger(structlog.PrintLogger(file=sys.stderr))


class PostgresStorage:
    """PostgreSQL storage backend with pgvector"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Apply the schema (if AUTO_MIGRATE) and open the connection pool."""
        # Migrations run on a plain connection BEFORE the pool exists: every pool
        # connection registers the pgvector type on init, which fails on a fresh
        # database until migration 001 has created the extension.
        if settings.AUTO_MIGRATE:
            conn = await asyncpg.connect(settings.PG_DSN)
            try:
                applied = await apply_migrations(conn)
            finally:
                await conn.close()
            logger.info("migrations_applied", count=applied)

        self.pool = await asyncpg.create_pool(
            settings.PG_DSN,
            min_size=settings.PG_POOL_MIN,
            max_size=settings.PG_POOL_MAX,
            command_timeout=60,
            init=self._init_connection
        )

        if not settings.AUTO_MIGRATE:
            # Least-privilege runtime role without DDL rights: an owner role
            # applies migrations separately (python init_db.py).
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            logger.info("auto_migrate_disabled_skipping_ddl")
    
    async def _init_connection(self, conn):
        """Initialize each connection with pgvector"""
        await register_vector(conn)
        
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            
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
    
