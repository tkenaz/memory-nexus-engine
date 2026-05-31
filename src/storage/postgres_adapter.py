"""PostgreSQL storage adapter для Memory Nexus с поддержкой временных фильтров"""
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector

from config.settings import settings
import structlog

logger = structlog.get_logger()


class PostgresStorage:
    """PostgreSQL storage backend для существующей БД"""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize connection pool"""
        self.pool = await asyncpg.create_pool(
            settings.PG_DSN,
            min_size=10,
            max_size=20,
            command_timeout=60
        )
        
        logger.info("postgres_initialized", pool_size=self.pool.get_size())
            
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            
    async def store_memory(
        self, 
        content: str, 
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory in existing table structure"""
        async with self.pool.acquire() as conn:
            # Handle different embedding formats
            if isinstance(embedding, str):
                # If it's a string representation like '[1,2,3]', parse it
                if embedding.startswith('[') and embedding.endswith(']'):
                    import ast
                    try:
                        embedding_list = ast.literal_eval(embedding)
                        logger.warning("store_memory_string_embedding",
                                     msg="Received embedding as string, parsed to list")
                    except:
                        raise ValueError(f"Cannot parse embedding string: {embedding[:100]}...")
                else:
                    raise ValueError(f"Invalid embedding format: {type(embedding)}")
            elif isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            elif isinstance(embedding, list):
                embedding_list = embedding
            else:
                raise ValueError(f"Unsupported embedding type: {type(embedding)}")
                
            # Generate ID like existing system
            import hashlib
            import time
            memory_id = f"mem_{int(time.time() * 1000)}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            
            # Convert to PostgreSQL array format
            if isinstance(embedding_list, list):
                embedding_str = '[' + ','.join(str(x) for x in embedding_list) + ']'
            else:
                raise ValueError(f"Expected list, got {type(embedding_list)}")
            
            # For pgvector, we pass the list directly and let asyncpg handle conversion
            await conn.execute('''
                INSERT INTO memories (id, content, embedding, metadata, created_at, updated_at)
                VALUES ($1, $2, $3::vector, $4, NOW(), NOW())
            ''', 
                memory_id,
                content,
                embedding_str,  # Pass as formatted string
                json.dumps(metadata or {})
            )
            
            logger.info("memory_stored", memory_id=memory_id)
            return memory_id
            
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get memory by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM memories WHERE id = $1
            ''', memory_id)
            
            if not row:
                return None
                
            return {
                'id': row['id'],
                'content': row['content'],
                'embedding': row['embedding'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            }
            
    async def semantic_search(
        self, 
        embedding: List[float], 
        limit: int = 10,
        min_similarity: float = 0.01,  # Низкий порог для e5-large!
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Vector similarity search с поддержкой фильтров по датам и metadata"""
        async with self.pool.acquire() as conn:
            # Convert embedding to PostgreSQL array format
            if isinstance(embedding, list):
                embedding_str = '[' + ','.join(str(x) for x in embedding) + ']'
            elif isinstance(embedding, np.ndarray):
                embedding_str = '[' + ','.join(str(x) for x in embedding.tolist()) + ']'
            else:
                raise ValueError(f"Unsupported embedding type: {type(embedding)}")
            
            # Build query dynamically
            query_parts = [
                """SELECT *, 
                   1 - (embedding <=> $1::vector) as similarity
                FROM memories
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> $1::vector) >= $2"""
            ]
            
            params = [embedding_str, min_similarity]
            param_count = 3
            
            # Add filters if specified
            if metadata_filter:
                for key, value in metadata_filter.items():
                    if key == 'created_at' and isinstance(value, dict):
                        # Обрабатываем фильтры по датам
                        if '$gte' in value:
                            query_parts.append(f'AND created_at >= ${param_count}')
                            params.append(value['$gte'])
                            param_count += 1
                        if '$lt' in value:
                            query_parts.append(f'AND created_at < ${param_count}')
                            params.append(value['$lt'])
                            param_count += 1
                        if '$lte' in value:
                            query_parts.append(f'AND created_at <= ${param_count}')
                            params.append(value['$lte'])
                            param_count += 1
                        if '$gt' in value:
                            query_parts.append(f'AND created_at > ${param_count}')
                            params.append(value['$gt'])
                            param_count += 1
                    else:
                        # Обычные фильтры по metadata (если не created_at)
                        # Пока пропускаем, чтобы не сломать существующую логику
                        logger.debug("metadata_filter_skipped", key=key, value=value)
            
            # Final query
            query = '\n'.join(query_parts)
            query += f' ORDER BY similarity DESC LIMIT ${param_count}'
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            
            results = []
            for row in rows:
                results.append({
                    'id': row['id'],
                    'content': row['content'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'created_at': row['created_at'],
                    'similarity': float(row['similarity'])
                })
                
            logger.debug("semantic_search_results", 
                        count=len(results),
                        filters=metadata_filter,
                        min_sim=min(r['similarity'] for r in results) if results else 0)
            
            return results
            
    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Keyword-based search с поддержкой временных фильтров"""
        async with self.pool.acquire() as conn:
            # Проверяем сначала поиск по ID
            if query.startswith('mem_'):
                rows = await conn.fetch("""
                    SELECT *, 1.0 as rank
                    FROM memories
                    WHERE id = $1
                    LIMIT 1
                """, query)
            else:
                # Full-text search по content
                # Создаем поисковый запрос
                search_query = ' & '.join(word for word in query.split() if len(word) > 2)
                
                if not search_query:
                    # Если все слова короткие или пустой запрос после очистки временных маркеров
                    # используем простой поиск по всем записям с фильтрами
                    base_query = "SELECT *, 1.0 as rank FROM memories WHERE 1=1"
                    params = []
                    param_count = 1
                    
                    # Добавляем фильтры по датам
                    if metadata_filter and 'created_at' in metadata_filter:
                        date_filter = metadata_filter['created_at']
                        if '$gte' in date_filter:
                            base_query += f' AND created_at >= ${param_count}'
                            params.append(date_filter['$gte'])
                            param_count += 1
                        if '$lt' in date_filter:
                            base_query += f' AND created_at < ${param_count}'
                            params.append(date_filter['$lt'])
                            param_count += 1
                    
                    base_query += f' ORDER BY created_at DESC LIMIT ${param_count}'
                    params.append(limit)
                    
                    rows = await conn.fetch(base_query, *params)
                else:
                    # Используем полнотекстовый поиск
                    base_query = """
                        SELECT *,
                            ts_rank_cd(to_tsvector('russian', content), 
                                      to_tsquery('russian', $1)) as rank
                        FROM memories
                        WHERE to_tsvector('russian', content) @@ to_tsquery('russian', $1)
                    """
                    params = [search_query]
                    param_count = 2
                    
                    # Добавляем фильтры по датам
                    if metadata_filter and 'created_at' in metadata_filter:
                        date_filter = metadata_filter['created_at']
                        if '$gte' in date_filter:
                            base_query += f' AND created_at >= ${param_count}'
                            params.append(date_filter['$gte'])
                            param_count += 1
                        if '$lt' in date_filter:
                            base_query += f' AND created_at < ${param_count}'
                            params.append(date_filter['$lt'])
                            param_count += 1
                    
                    base_query += f' ORDER BY rank DESC LIMIT ${param_count}'
                    params.append(limit)
                    
                    rows = await conn.fetch(base_query, *params)
                        
            # Конвертируем результаты
            results = []
            for row in rows:
                results.append({
                    'id': row['id'],
                    'content': row['content'],
                    'embedding': row['embedding'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'rank': float(row['rank'])
                })
                
            logger.info("keyword_search_complete",
                       query=query,
                       filters=metadata_filter,
                       results_count=len(results),
                       max_rank=max(r['rank'] for r in results) if results else 0)
                       
            return results
            
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]):
        """Update memory fields"""
        async with self.pool.acquire() as conn:
            # Build update query dynamically
            set_clauses = []
            params = []
            param_count = 1
            
            if 'content' in updates:
                set_clauses.append(f'content = ${param_count}')
                params.append(updates['content'])
                param_count += 1
                
            if 'embedding' in updates:
                set_clauses.append(f'embedding = ${param_count}::vector')
                embedding_val = updates['embedding']
                if isinstance(embedding_val, list):
                    embedding_str = '[' + ','.join(str(x) for x in embedding_val) + ']'
                elif isinstance(embedding_val, np.ndarray):
                    embedding_str = '[' + ','.join(str(x) for x in embedding_val.tolist()) + ']'
                else:
                    raise ValueError(f"Unsupported embedding type: {type(embedding_val)}")
                params.append(embedding_str)
                param_count += 1
                
            if 'metadata' in updates:
                set_clauses.append(f'metadata = ${param_count}')
                params.append(json.dumps(updates['metadata']))
                param_count += 1
                
            # Always update updated_at
            set_clauses.append('updated_at = NOW()')
            
            # Add memory_id as last param
            params.append(memory_id)
            
            query = f'''
                UPDATE memories 
                SET {', '.join(set_clauses)}
                WHERE id = ${param_count}
            '''
            
            await conn.execute(query, *params)
            
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as total_memories,
                    COUNT(embedding) as memories_with_embeddings,
                    MIN(created_at) as oldest_memory,
                    MAX(created_at) as newest_memory,
                    pg_size_pretty(pg_total_relation_size('memories')) as table_size
                FROM memories
            ''')
            
            return dict(stats)
