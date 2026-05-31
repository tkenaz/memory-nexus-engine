"""Enhanced PostgreSQL storage with SASH-F structure and REAL semantic search"""
import json
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta
import asyncpg
import asyncio
import numpy as np
import os
import sys
import logging

# Suppress ALL logging to avoid breaking MCP JSON protocol
logging.basicConfig(level=logging.CRITICAL, handlers=[])
logging.getLogger().setLevel(logging.CRITICAL)

# Suppress transformers/sentence-transformers output to avoid breaking MCP JSON protocol
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from FlagEmbedding import BGEM3FlagModel
from pgvector.asyncpg import register_vector

from ..models.sashf import SceneFrame, MemoryLink, LinkType, Pattern, Affect
from ..core.sashf_parser import SASHFParser
from ..core.query_intent import QueryIntentDetector, QueryIntent
from ..core import chunker
from .postgres import PostgresStorage
from config.settings import settings
import re
import calendar

# Create a null logger that does nothing
class NullLogger:
    def info(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass

logger = NullLogger()


class EnhancedPostgresStorage(PostgresStorage):
    """Enhanced storage with SASH-F structure and graph capabilities"""
    
    def __init__(self):
        super().__init__()
        self.parser = SASHFParser()
        self.intent_detector = QueryIntentDetector()
        self.encoder = None
        self._model_lock = asyncio.Lock()
    
    async def _ensure_encoder(self):
        """Lazy load encoder model (BGE-M3)"""
        if self.encoder is None:
            async with self._model_lock:
                if self.encoder is None:
                    logger.info("loading_encoder_model", model=settings.EMBEDDING_MODEL)
                    loop = asyncio.get_event_loop()

                    # Suppress stdout/stderr during model loading to avoid breaking MCP protocol
                    def load_model_silently():
                        import warnings
                        warnings.filterwarnings('ignore')
                        # Redirect stdout/stderr temporarily
                        old_stdout = sys.stdout
                        old_stderr = sys.stderr
                        try:
                            sys.stdout = open(os.devnull, 'w')
                            sys.stderr = open(os.devnull, 'w')
                            model = BGEM3FlagModel(settings.EMBEDDING_MODEL, use_fp16=True)
                            return model
                        finally:
                            sys.stdout.close()
                            sys.stderr.close()
                            sys.stdout = old_stdout
                            sys.stderr = old_stderr

                    self.encoder = await loop.run_in_executor(None, load_model_silently)
                    logger.info("encoder_model_loaded", model="BGE-M3")
    
    async def encode_text(self, text: str) -> List[float]:
        """Encode text to embedding using BGE-M3"""
        await self._ensure_encoder()

        # BGE-M3 doesn't need prefix, works directly
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.encoder.encode(
                [text],
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
        )

        # BGE-M3 returns dict with 'dense_vecs'
        embedding = result['dense_vecs'][0]
        return embedding.tolist()
    
    def _preprocess_temporal_query(self, query_text: str) -> Tuple[str, Optional[Dict]]:
        """Extract temporal markers from query and convert to metadata filters"""
        clean_query = query_text
        metadata_filter = None
        
        # Get current time in UTC with timezone awareness
        from datetime import timezone
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        logger.info("preprocessing_temporal_query", 
                   query=query_text,
                   current_time=now.isoformat())
        
        # Month name mapping (Russian and English)
        month_map = {
            'января': 1, 'january': 1, 'jan': 1,
            'февраля': 2, 'february': 2, 'feb': 2,
            'марта': 3, 'march': 3, 'mar': 3,
            'апреля': 4, 'april': 4, 'apr': 4,
            'мая': 5, 'may': 5,
            'июня': 6, 'june': 6, 'jun': 6,
            'июля': 7, 'july': 7, 'jul': 7,
            'августа': 8, 'august': 8, 'aug': 8,
            'сентября': 9, 'september': 9, 'sep': 9,
            'октября': 10, 'october': 10, 'oct': 10,
            'ноября': 11, 'november': 11, 'nov': 11,
            'декабря': 12, 'december': 12, 'dec': 12,
            # Also add nominative case (without 'я' at the end)
            'октябрь': 10, 'сентябрь': 9, 'ноябрь': 11, 'декабрь': 12
        }
        
        # Check for month + year patterns (e.g., "октябрь 2025", "October 2025")
        month_year_pattern = r'(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|october|october|september|november|january|february|march|april|may|june|july|august|октябрь|сентябрь|ноябрь|декабрь)\s+(\d{4})'
        match = re.search(month_year_pattern, query_text, re.IGNORECASE)
        if match:
            month_name = match.group(1).lower()
            year = int(match.group(2))
            month = month_map.get(month_name)
            if month:
                # Get first and last day of month (timezone-aware)
                from datetime import timezone
                _, last_day = calendar.monthrange(year, month)
                start_date = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
                end_date = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
                
                metadata_filter = {
                    'created_at': {
                        '$gte': start_date,
                        '$lte': end_date
                    }
                }
                # Remove the month+year from query
                clean_query = re.sub(month_year_pattern, '', query_text, flags=re.IGNORECASE).strip()
                logger.info("month_year_found", month=month, year=year, start=start_date, end=end_date)
                return clean_query, metadata_filter
        
        # Check for specific dates (e.g., "21 августа", "August 21")
        date_patterns = [
            r'(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)',
            r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(\d{4})-(\d{2})-(\d{2})',  # ISO format
            r'(\d{2})\.(\d{2})\.(\d{4})'  # DD.MM.YYYY
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 2 and match.group(2).isalpha():  # Day + Month name
                        day = int(match.group(1))
                        month_name = match.group(2).lower()
                        month = month_map.get(month_name)
                        if month:
                            year = now.year
                            from datetime import timezone
                            date_obj = datetime(year, month, day, tzinfo=timezone.utc)
                            metadata_filter = {
                                'created_at': {
                                    '$gte': date_obj.replace(hour=0, minute=0, second=0),
                                    '$lt': (date_obj + timedelta(days=1)).replace(hour=0, minute=0, second=0)
                                }
                            }
                            clean_query = re.sub(pattern, '', query_text, flags=re.IGNORECASE).strip()
                            logger.info("specific_date_found", date=date_obj.date())
                            break
                except Exception as e:
                    logger.warning("date_parsing_failed", pattern=pattern, error=str(e))
        
        # Define relative temporal patterns if no specific date found
        if not metadata_filter:
            temporal_patterns = {
                r'\bсегодня\b|\btoday\b': {
                    'created_at': {'$gte': today_start}
                },
                r'\bвчера\b|\byesterday\b': {
                    'created_at': {
                        '$gte': today_start - timedelta(days=1),
                        '$lt': today_start
                    }
                },
                r'\bэта неделя\b|\bthis week\b': {
                    'created_at': {'$gte': today_start - timedelta(days=today_start.weekday())}
                },
                r'\bпрошлая неделя\b|\blast week\b': {
                    'created_at': {
                        '$gte': today_start - timedelta(days=today_start.weekday() + 7),
                        '$lt': today_start - timedelta(days=today_start.weekday())
                    }
                },
                r'\bэтот месяц\b|\bthis month\b': {
                    'created_at': {'$gte': today_start.replace(day=1)}
                },
                r'\bпрошлый месяц\b|\blast month\b': {
                    'created_at': {
                        '$gte': (today_start.replace(day=1) - timedelta(days=1)).replace(day=1),
                        '$lt': today_start.replace(day=1)
                    }
                }
            }
        
            # Check for temporal markers
            for pattern, filter_dict in temporal_patterns.items():
                if re.search(pattern, query_text, re.IGNORECASE):
                    metadata_filter = filter_dict
                    # Remove temporal words from query for better semantic search
                    clean_query = re.sub(pattern, '', query_text, flags=re.IGNORECASE).strip()
                    logger.info("temporal_marker_found", 
                               pattern=pattern,
                               filter=filter_dict)
                    break
        
        logger.info("temporal_preprocessing_result", 
                    original_query=query_text,
                    clean_query=clean_query,
                    metadata_filter=metadata_filter)
                    
        return clean_query, metadata_filter
    
    async def store_memory_enhanced(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store memory with SASH-F structure, auto-linking, and chunking for long content."""

        # Extract SASH-F structure
        sashf_data = self.parser.parse(content)

        # Convert Affect object to dict if present
        if sashf_data.get('affect'):
            affect = sashf_data['affect']
            if isinstance(affect, Affect):
                sashf_data['affect'] = {
                    'valence': affect.valence,
                    'arousal': affect.arousal,
                    'dominance': affect.dominance,
                    'bucket': affect.bucket,
                    'intensity': affect.intensity
                }

        # Add time bucket
        now = datetime.utcnow()
        hour = now.hour
        if 5 <= hour < 12:
            sashf_data['time_bucket'] = 'morning'
        elif 12 <= hour < 17:
            sashf_data['time_bucket'] = 'day'
        elif 17 <= hour < 22:
            sashf_data['time_bucket'] = 'evening'
        else:
            sashf_data['time_bucket'] = 'night'

        # Generate ID
        import hashlib
        import time
        memory_id = f"mem_{int(time.time() * 1000)}_{hashlib.md5(content.encode()).hexdigest()[:8]}"

        # Check if chunking is needed
        chunks = chunker.chunk_text(content)

        async with self.pool.acquire() as conn:
            if not chunks:
                # Short content — store as before with embedding
                embedding_array = self._to_numpy(embedding)

                await conn.execute("""
                    INSERT INTO memories (id, content, embedding, metadata, sashf, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, NOW(), NOW())
                """, memory_id, content, embedding_array, json.dumps(metadata or {}), json.dumps(sashf_data))
            else:
                # Long content — parent (no embedding) + chunk records (each with own embedding)
                await conn.execute("""
                    INSERT INTO memories (id, content, embedding, metadata, sashf, created_at, updated_at)
                    VALUES ($1, $2, NULL, $3, $4, NOW(), NOW())
                """, memory_id, content, json.dumps(metadata or {}), json.dumps(sashf_data))

                for chunk in chunks:
                    chunk_embedding = await self.encode_text(chunk.text)
                    chunk_embedding_array = np.array(chunk_embedding, dtype=np.float32)
                    chunk_id = f"{memory_id}_c{chunk.index}"

                    await conn.execute("""
                        INSERT INTO memories (id, content, embedding, metadata, sashf, parent_id, chunk_index, created_at, updated_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW())
                    """, chunk_id, chunk.text, chunk_embedding_array,
                        json.dumps(metadata or {}), json.dumps({}),
                        memory_id, chunk.index)

                logger.info("stored_chunked_memory",
                           memory_id=memory_id,
                           chunk_count=len(chunks))

            # Auto-link with similar memories (only for parent/standalone)
            asyncio.create_task(self._auto_link_memory(memory_id))

            # Check for patterns
            asyncio.create_task(self._check_patterns(memory_id, sashf_data))

            logger.info("stored_enhanced_memory",
                       memory_id=memory_id,
                       actors=sashf_data.get('actors'),
                       act=sashf_data.get('act'),
                       tags=sashf_data.get('tags'))

            return memory_id

    @staticmethod
    def _to_numpy(embedding) -> Optional[np.ndarray]:
        """Convert embedding to numpy array for pgvector."""
        if embedding is None:
            return None
        if isinstance(embedding, list):
            return np.array(embedding, dtype=np.float32)
        if isinstance(embedding, np.ndarray):
            return embedding.astype(np.float32)
        return None

    async def _resolve_to_parents(
        self,
        results: List[Dict],
        conn
    ) -> List[Dict]:
        """Replace chunk results with their parent's full content.

        - Standalone memories pass through unchanged.
        - Chunks are resolved to their parent (full content loaded).
        - If multiple chunks of the same parent match, keep the best score.
        - Preserves order by score.
        """
        if not results:
            return results

        resolved: Dict[str, Dict] = {}  # effective_id -> best result dict

        # Collect parent_ids we need to load
        parent_ids_to_load: set = set()
        for r in results:
            pid = r.get('parent_id')
            if pid:
                parent_ids_to_load.add(pid)

        # Batch-load parent content
        parents_content: Dict[str, Dict] = {}
        if parent_ids_to_load:
            parent_rows = await conn.fetch("""
                SELECT id, content, metadata, sashf, created_at
                FROM memories WHERE id = ANY($1)
            """, list(parent_ids_to_load))
            for pr in parent_rows:
                parents_content[pr['id']] = dict(pr)

        # Track chunk hit counts per parent for multi-chunk boost
        chunk_hits: Dict[str, int] = {}
        for r in results:
            pid = r.get('parent_id')
            if pid:
                chunk_hits[pid] = chunk_hits.get(pid, 0) + 1

        for r in results:
            parent_id = r.get('parent_id')
            effective_id = parent_id if parent_id else r['id']

            if effective_id in resolved:
                # Keep the higher score
                if float(r.get('similarity', 0)) > float(resolved[effective_id].get('similarity', 0)):
                    resolved[effective_id]['similarity'] = r['similarity']
                continue

            if parent_id and parent_id in parents_content:
                # Replace chunk with parent data but keep chunk's score
                parent = parents_content[parent_id]
                base_sim = float(r.get('similarity', 0))

                # Multi-chunk boost: if multiple chunks of same parent matched,
                # boost by 2% per extra hit (caps at 10% boost)
                hits = chunk_hits.get(parent_id, 1)
                multi_chunk_boost = min(0.10, (hits - 1) * 0.02) if hits > 1 else 0.0

                resolved[effective_id] = {
                    'id': parent['id'],
                    'content': parent['content'],
                    'metadata': parent.get('metadata', r.get('metadata')),
                    'sashf': parent.get('sashf', r.get('sashf')),
                    'created_at': parent.get('created_at', r.get('created_at')),
                    'similarity': base_sim + multi_chunk_boost,
                    'resolved_from_chunk': True,
                }
            else:
                # Standalone memory or parent not found — pass through
                resolved[effective_id] = dict(r)

        # Sort by similarity descending
        out = list(resolved.values())
        out.sort(key=lambda x: float(x.get('similarity', 0)), reverse=True)
        return out

    async def search_with_graph(
        self,
        query: str,
        limit: int = 10,
        use_graph: bool = True,
        graph_depth: int = 2,
        min_similarity: float = 0.15
    ) -> List[Dict]:
        """
        ПРАВИЛЬНАЯ АРХИТЕКТУРА ПОИСКА:
        1. Semantic search (vector similarity) - PRIMARY
        2. Keyword fallback (ts_rank) - only if semantic fails
        3. Graph expansion - on top results
        
        NOW WITH TEMPORAL FILTERING!
        """
        
        # STEP 0: INTENT DETECTION & TEMPORAL PREPROCESSING
        intent, intent_context = self.intent_detector.detect_intent(query)
        clean_query, temporal_filter = self._preprocess_temporal_query(query)

        # Enhance query based on intent
        if intent != QueryIntent.MIXED:
            enhanced_query = self.intent_detector.disambiguate_query(clean_query, intent, intent_context)
            if enhanced_query != clean_query:
                logger.info("query_enhanced_by_intent",
                           original=clean_query,
                           enhanced=enhanced_query,
                           intent=intent.value)
                clean_query = enhanced_query

        # Logging only (stderr would break MCP JSON protocol)
        logger.info("search_intent_temporal",
                    intent=intent.value,
                    original=query,
                    clean=clean_query,
                    temporal_filter=str(temporal_filter))

        logger.info("search_with_intent_and_temporal",
                   original_query=query,
                   clean_query=clean_query,
                   intent=intent.value,
                   entities=intent_context['entities'],
                   has_temporal_filter=temporal_filter is not None)
        
        async with self.pool.acquire() as conn:
            results = []
            search_method = "none"
            
            # SPECIAL CASE: Query is ONLY temporal markers (e.g., just "сегодня")
            if not clean_query.strip() and temporal_filter:
                logger.info("temporal_only_query", 
                           filter=temporal_filter)
                
                # Search ONLY by created_at, no semantic/keyword
                query_parts = ["""
                    SELECT
                        id,
                        content,
                        metadata,
                        sashf,
                        created_at,
                        1.0 as similarity
                    FROM memories
                    WHERE parent_id IS NULL
                """]
                params = []
                param_count = 1
                
                # Add temporal filters
                if 'created_at' in temporal_filter:
                    date_filter = temporal_filter['created_at']
                    if '$gte' in date_filter:
                        query_parts.append(f'AND created_at >= ${param_count}')
                        params.append(date_filter['$gte'])
                        param_count += 1
                    if '$lt' in date_filter:
                        query_parts.append(f'AND created_at < ${param_count}')
                        params.append(date_filter['$lt'])
                        param_count += 1
                    if '$lte' in date_filter:
                        query_parts.append(f'AND created_at <= ${param_count}')
                        params.append(date_filter['$lte'])
                        param_count += 1
                
                query_parts.append(f'ORDER BY created_at DESC LIMIT ${param_count}')
                params.append(limit)
                
                full_query = '\n'.join(query_parts)
                temporal_results = await conn.fetch(full_query, *params)
                results = [dict(r) for r in temporal_results]
                search_method = "temporal_only"
                
                logger.info("temporal_only_search_complete", found=len(results))
                
                # Return results directly without graph expansion
                # (graph expansion could be added later if needed)
                return results
            
            # Use cleaned query for semantic/keyword search
            search_query = clean_query if clean_query else query
            
            # STEP 1: SEMANTIC SEARCH (PRIMARY)
            error_msg = None
            try:
                # Generate query embedding
                query_embedding = await self.encode_text(search_query)
                # Pass as numpy array directly to pgvector
                embedding_array = np.array(query_embedding, dtype=np.float32)
                
                # Build dynamic query with temporal filters
                query_parts = ["""
                    SELECT
                        id,
                        content,
                        metadata,
                        sashf,
                        created_at,
                        parent_id,
                        chunk_index,
                        1.0 - (embedding <=> $1) as similarity
                    FROM memories
                    WHERE embedding IS NOT NULL
                    AND 1.0 - (embedding <=> $1) >= $2
                """]
                params = [embedding_array, min_similarity]
                param_count = 3
                
                # Add temporal filters if present
                if temporal_filter and 'created_at' in temporal_filter:
                    date_filter = temporal_filter['created_at']
                    if '$gte' in date_filter:
                        query_parts.append(f'AND created_at >= ${param_count}')
                        params.append(date_filter['$gte'])
                        param_count += 1
                    if '$lt' in date_filter:
                        query_parts.append(f'AND created_at < ${param_count}')
                        params.append(date_filter['$lt'])
                        param_count += 1
                    if '$lte' in date_filter:
                        query_parts.append(f'AND created_at <= ${param_count}')
                        params.append(date_filter['$lte'])
                        param_count += 1
                
                query_parts.append(f'ORDER BY embedding <=> $1 LIMIT ${param_count}')
                params.append(limit * 5)  # More candidates to survive chunk→parent dedup
                
                full_query = '\n'.join(query_parts)
                semantic_results = await conn.fetch(full_query, *params)
                
                results = [dict(r) for r in semantic_results]
                search_method = "semantic"
                
                logger.info("semantic_search_complete",
                           found=len(results),
                           max_score=max(r['similarity'] for r in results) if results else 0)
                
            except Exception as e:
                error_msg = f"Semantic search failed: {str(e)}"
                logger.error("semantic_search_error", error=str(e))
                # Don't re-raise, fallback to keyword search
                results = []
            
            # STEP 2: KEYWORD FALLBACK (if semantic gave poor results)
            if not results or (results and max(float(r.get('similarity', 0)) for r in results) < settings.KEYWORD_FALLBACK_THRESHOLD):
                logger.info("using_keyword_fallback", 
                           semantic_count=len(results),
                           max_semantic_score=max(float(r.get('similarity', 0)) for r in results) if results else 0)
                
                # Detect language for FTS config: Latin-only → 'english', otherwise → 'russian'
                fts_config = 'english' if re.match(r'^[a-zA-Z0-9\s\-_.]+$', search_query) else 'russian'

                # Build dynamic keyword query with temporal filters
                keyword_query_parts = [f"""
                    SELECT
                        id,
                        content,
                        metadata,
                        sashf,
                        created_at,
                        parent_id,
                        chunk_index,
                        ts_rank_cd(to_tsvector('{fts_config}', content),
                                  plainto_tsquery('{fts_config}', $1)) as similarity
                    FROM memories
                    WHERE to_tsvector('{fts_config}', content) @@ plainto_tsquery('{fts_config}', $1)
                    AND parent_id IS NULL
                """]
                keyword_params = [search_query]
                keyword_param_count = 2
                
                # Add temporal filters if present
                if temporal_filter and 'created_at' in temporal_filter:
                    date_filter = temporal_filter['created_at']
                    if '$gte' in date_filter:
                        keyword_query_parts.append(f'AND created_at >= ${keyword_param_count}')
                        keyword_params.append(date_filter['$gte'])
                        keyword_param_count += 1
                    if '$lt' in date_filter:
                        keyword_query_parts.append(f'AND created_at < ${keyword_param_count}')
                        keyword_params.append(date_filter['$lt'])
                        keyword_param_count += 1
                    if '$lte' in date_filter:
                        keyword_query_parts.append(f'AND created_at <= ${keyword_param_count}')
                        keyword_params.append(date_filter['$lte'])
                        keyword_param_count += 1
                
                keyword_query_parts.append(f'ORDER BY similarity DESC LIMIT ${keyword_param_count}')
                keyword_params.append(limit)
                
                full_keyword_query = '\n'.join(keyword_query_parts)
                keyword_results = await conn.fetch(full_keyword_query, *keyword_params)

                # Normalize keyword scores (ts_rank_cd returns [0, 32+], cosine is [0, 1])
                keyword_dict = {}
                if keyword_results:
                    kw_scores = [float(r['similarity']) for r in keyword_results]
                    kw_min, kw_max = min(kw_scores), max(kw_scores)
                    for r in keyword_results:
                        d = dict(r)
                        if kw_max > kw_min:
                            d['similarity'] = (float(d['similarity']) - kw_min) / (kw_max - kw_min)
                        else:
                            d['similarity'] = 1.0  # single result = best
                        keyword_dict[d['id']] = d

                # Merge with semantic results
                for r in results:
                    if r['id'] in keyword_dict:
                        # Boost score if found by both methods
                        r['similarity'] = float(r['similarity']) * settings.SEMANTIC_WEIGHT + float(keyword_dict[r['id']]['similarity']) * settings.KEYWORD_WEIGHT
                        del keyword_dict[r['id']]

                # Add remaining keyword results
                for kr in keyword_dict.values():
                    results.append(kr)
                
                search_method = "hybrid" if results else "keyword"

            # STEP 2.5: METADATA/TAGS SEARCH (dynamic scoring)
            # Two-tier: exact tag match (0.8) and substring in metadata (0.5)
            logger.info("searching_metadata_tags", query=search_query)

            existing_ids = {r['id'] for r in results}
            metadata_added = 0

            # Helper to build temporal WHERE clauses
            def _temporal_clauses(param_count, params_list):
                parts = []
                if temporal_filter and 'created_at' in temporal_filter:
                    date_filter = temporal_filter['created_at']
                    if '$gte' in date_filter:
                        parts.append(f'AND created_at >= ${param_count}')
                        params_list.append(date_filter['$gte'])
                        param_count += 1
                    if '$lt' in date_filter:
                        parts.append(f'AND created_at < ${param_count}')
                        params_list.append(date_filter['$lt'])
                        param_count += 1
                    if '$lte' in date_filter:
                        parts.append(f'AND created_at <= ${param_count}')
                        params_list.append(date_filter['$lte'])
                        param_count += 1
                return parts, param_count

            # 2.5a: Exact tag match (score = 0.8)
            tag_query_parts = ["""
                SELECT id, content, metadata, sashf, created_at, parent_id, chunk_index,
                       0.8 as similarity
                FROM memories
                WHERE metadata->'tags' ? $1
                AND parent_id IS NULL
            """]
            tag_params = [search_query]
            tag_pc = 2
            temporal_parts, tag_pc = _temporal_clauses(tag_pc, tag_params)
            tag_query_parts.extend(temporal_parts)
            tag_query_parts.append(f'ORDER BY created_at DESC LIMIT ${tag_pc}')
            tag_params.append(limit)
            try:
                tag_results = await conn.fetch('\n'.join(tag_query_parts), *tag_params)
                for mr in tag_results:
                    if mr['id'] not in existing_ids:
                        results.append(dict(mr))
                        existing_ids.add(mr['id'])
                        metadata_added += 1
            except Exception:
                pass  # tags column may not exist as jsonb array

            # 2.5b: Substring match in metadata (score = 0.5)
            sub_query_parts = ["""
                SELECT id, content, metadata, sashf, created_at, parent_id, chunk_index,
                       0.5 as similarity
                FROM memories
                WHERE metadata::text ILIKE $1
                AND parent_id IS NULL
            """]
            sub_params = [f'%{search_query}%']
            sub_pc = 2
            temporal_parts2, sub_pc = _temporal_clauses(sub_pc, sub_params)
            sub_query_parts.extend(temporal_parts2)
            sub_query_parts.append(f'ORDER BY created_at DESC LIMIT ${sub_pc}')
            sub_params.append(limit)
            sub_results = await conn.fetch('\n'.join(sub_query_parts), *sub_params)
            for mr in sub_results:
                if mr['id'] not in existing_ids:
                    results.append(dict(mr))
                    existing_ids.add(mr['id'])
                    metadata_added += 1

            if metadata_added > 0:
                logger.info("metadata_search_added", count=metadata_added)
                search_method = "hybrid+metadata"

            # STEP 2.9: RESOLVE CHUNKS TO PARENTS
            # Before graph expansion, collapse chunk results into parent records
            results = await self._resolve_to_parents(results, conn)

            # STEP 3: GRAPH EXPANSION (on top results — parents/standalone only)
            # Graph results are SUPPLEMENTARY - they should NOT outrank direct semantic matches
            if use_graph and results:
                top_ids = [r['id'] for r in results[:3] if not r.get('parent_id')]
                # Get max semantic similarity to scale graph results appropriately
                max_semantic_sim = max(float(r.get('similarity', 0)) for r in results) if results else 0.5

                for start_id in top_ids:
                    graph_results = await conn.fetch("""
                        WITH RECURSIVE graph AS (
                            SELECT m.id, m.content, m.metadata, m.sashf, 0 as depth, 1.0::float8 as weight
                            FROM memories m
                            WHERE m.id = $1
                            AND m.parent_id IS NULL

                            UNION ALL

                            SELECT m.id, m.content, m.metadata, m.sashf, g.depth + 1,
                                   (g.weight * ml.weight * 0.8)::float8
                            FROM graph g
                            JOIN memory_links ml ON g.id = ml.from_id
                            JOIN memories m ON ml.to_id = m.id
                            WHERE g.depth < $2
                            AND ml.link_type IN ('similar', 'after', 'caused_by')
                            AND m.parent_id IS NULL
                        )
                        SELECT DISTINCT id, content, metadata, sashf,
                               MIN(depth) as depth, MAX(weight) as weight
                        FROM graph
                        WHERE depth > 0
                        GROUP BY id, content, metadata, sashf
                        ORDER BY weight DESC
                        LIMIT 5
                    """, start_id, graph_depth)

                    stop_actors = set(settings.GRAPH_STOP_ACTORS)
                    for gr in graph_results:
                        # Filter hub nodes: skip if ALL actors are common/stop actors
                        try:
                            sashf_data = json.loads(gr['sashf']) if isinstance(gr['sashf'], str) else (gr['sashf'] or {})
                            actors = [a.lower() for a in sashf_data.get('actors', [])]
                            if actors and all(a in stop_actors for a in actors):
                                continue  # pure hub node, skip
                        except (json.JSONDecodeError, TypeError):
                            pass

                        # Scale graph weight relative to semantic results
                        # Graph results should enhance, not dominate - cap at 80% of max semantic
                        scaled_similarity = float(gr['weight']) * max_semantic_sim * 0.8
                        results.append({
                            'id': gr['id'],
                            'content': gr['content'],
                            'similarity': scaled_similarity,
                            'metadata': gr['metadata'],
                            'sashf': gr['sashf'],
                            'source': 'graph'  # Mark as graph-expanded
                        })

            # De-duplicate and apply recency boost
            seen = set()
            unique_results = []
            now = datetime.utcnow()

            for r in results:
                if r['id'] not in seen:
                    seen.add(r['id'])
                    # Ensure similarity is float
                    base_similarity = float(r.get('similarity', 0))

                    # Calculate recency boost (exponential decay over days)
                    if 'created_at' in r and r['created_at']:
                        # Handle both datetime object and string
                        if isinstance(r['created_at'], str):
                            created = datetime.fromisoformat(r['created_at'].replace('Z', '+00:00'))
                        else:
                            created = r['created_at']

                        # Make created timezone-aware if it isn't
                        if created.tzinfo is None:
                            from datetime import timezone
                            created = created.replace(tzinfo=timezone.utc)

                        # Make now timezone-aware
                        from datetime import timezone
                        now_aware = now.replace(tzinfo=timezone.utc)

                        days_old = (now_aware - created).days
                        # Boost newer memories: e^(-days/30) gives 1.0 for today, 0.36 for 30 days
                        recency_boost = np.exp(-days_old / 30.0)

                        # Apply intent-based boost factors
                        recency_weight = intent_context['boost_factors'].get('recency', 1.0)
                        if intent == QueryIntent.TEMPORAL:
                            # For temporal queries, recency is more important
                            r['similarity'] = base_similarity * 0.7 + recency_boost * 0.3 * recency_weight
                        else:
                            # Standard weighting — semantics dominate, recency is minor
                            r['similarity'] = base_similarity * 0.9 + recency_boost * 0.1 * recency_weight

                        # Additional boosts based on intent
                        if intent == QueryIntent.PROJECT and 'projects' in intent_context['entities']:
                            # Boost if metadata contains the project
                            if 'metadata' in r and r['metadata']:
                                metadata = r['metadata'] if isinstance(r['metadata'], dict) else json.loads(r['metadata'])
                                if any(proj in str(metadata).lower() for proj in intent_context['entities']['projects']):
                                    r['similarity'] *= intent_context['boost_factors'].get('metadata_match', 1.0)

                        r['recency_boost'] = recency_boost
                        r['days_old'] = days_old
                        r['intent'] = intent.value
                    else:
                        r['similarity'] = base_similarity
                        r['recency_boost'] = 0
                        r['days_old'] = None

                    unique_results.append(r)

            unique_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info("search_complete",
                       method=search_method,
                       total_results=len(unique_results),
                       top_score=unique_results[0]['similarity'] if unique_results else 0)
            
            return unique_results[:limit]
    
    # ... остальные методы без изменений ...
    async def _auto_link_memory(self, memory_id: str):
        """Automatically create links for new memory"""
        try:
            async with self.pool.acquire() as conn:
                has_embedding = await conn.fetchval("""
                    SELECT embedding IS NOT NULL FROM memories WHERE id = $1
                """, memory_id)
                
                if has_embedding:
                    similar = await conn.fetch("""
                        SELECT m2.id, 1.0 - (m1.embedding <=> m2.embedding) as similarity
                        FROM memories m1, memories m2
                        WHERE m1.id = $1
                        AND m2.id != $1
                        AND m1.embedding IS NOT NULL
                        AND m2.embedding IS NOT NULL
                        AND m2.parent_id IS NULL
                        AND 1.0 - (m1.embedding <=> m2.embedding) > 0.85
                        ORDER BY similarity DESC
                        LIMIT 10
                    """, memory_id)
                    
                    for sim in similar:
                        await conn.execute("""
                            INSERT INTO memory_links (from_id, to_id, link_type, weight, confidence)
                            VALUES ($1, $2, 'similar', $3, $3)
                            ON CONFLICT DO NOTHING
                        """, memory_id, sim['id'], float(sim['similarity']))
                    
                    if len(similar) > 0:
                        logger.info("auto_linked_similar", 
                                   memory_id=memory_id, 
                                   linked_count=len(similar))
                
                recent = await conn.fetch("""
                    SELECT id, created_at
                    FROM memories
                    WHERE id != $1
                    AND parent_id IS NULL
                    AND created_at > NOW() - INTERVAL '24 hours'
                    AND created_at < (SELECT created_at FROM memories WHERE id = $1)
                    ORDER BY created_at DESC
                    LIMIT 5
                """, memory_id)
                
                for rec in recent:
                    await conn.execute("""
                        INSERT INTO memory_links (from_id, to_id, link_type, weight, confidence)
                        VALUES ($1, $2, 'after', 0.7, 0.7)
                        ON CONFLICT DO NOTHING
                    """, rec['id'], memory_id)
                    
        except Exception as e:
            logger.error("auto_link_failed", error=str(e), memory_id=memory_id)
    
    async def _check_patterns(self, memory_id: str, sashf_data: Dict):
        """Check if this memory matches any patterns"""
        try:
            pattern_key = self._generate_pattern_key(sashf_data)
            if not pattern_key:
                return
            
            async with self.pool.acquire() as conn:
                existing = await conn.fetchrow("""
                    SELECT id, occurrences, example_ids
                    FROM memory_patterns
                    WHERE pattern_key = $1
                """, pattern_key)
                
                if existing:
                    current_examples = existing['example_ids'] or []
                    new_examples = ([memory_id] + current_examples)[:10]
                    
                    await conn.execute("""
                        UPDATE memory_patterns
                        SET occurrences = occurrences + 1,
                            last_seen = NOW(),
                            example_ids = $1,
                            confidence = LEAST(1.0, confidence + 0.05)
                        WHERE id = $2
                    """, new_examples, existing['id'])
                    
                    logger.info("pattern_updated", 
                               pattern_key=pattern_key,
                               occurrences=existing['occurrences'] + 1)
                else:
                    await conn.execute("""
                        INSERT INTO memory_patterns 
                        (pattern_key, actors, act, manner, affect_bucket, example_ids)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, 
                    pattern_key,
                    sashf_data.get('actors', []),
                    sashf_data.get('act'),
                    sashf_data.get('manner'),
                    sashf_data.get('affect', {}).get('bucket') if sashf_data.get('affect') else None,
                    [memory_id])
                    
                    logger.info("pattern_created", pattern_key=pattern_key)
                    
        except Exception as e:
            logger.error("pattern_check_failed", error=str(e))
    
    def _generate_pattern_key(self, sashf_data: Dict) -> Optional[str]:
        """Generate canonical pattern key from SASH-F data"""
        components = []
        
        if sashf_data.get('actors'):
            actors = sorted(sashf_data['actors'])[:2]
            components.append(f"A:{','.join(actors)}")
        
        if sashf_data.get('act'):
            components.append(f"V:{sashf_data['act']}")
        
        if sashf_data.get('manner'):
            components.append(f"M:{sashf_data['manner']}")
        
        if sashf_data.get('affect'):
            affect = sashf_data['affect']
            if isinstance(affect, dict) and 'bucket' in affect:
                components.append(f"E:{affect['bucket']}")
        
        if len(components) < 2:
            return None
        
        return '|'.join(components)
    
    async def add_memory_link(
        self,
        from_id: str,
        to_id: str,
        link_type: str,
        weight: float = 1.0,
        confidence: float = 0.8,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Manually add a link between memories"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO memory_links 
                    (from_id, to_id, link_type, weight, confidence, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (from_id, to_id, link_type) 
                    DO UPDATE SET 
                        weight = EXCLUDED.weight,
                        confidence = EXCLUDED.confidence,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                """, from_id, to_id, link_type, weight, confidence, 
                json.dumps(metadata or {}))
                
                logger.info("link_added", 
                           from_id=from_id, 
                           to_id=to_id, 
                           link_type=link_type)
                return True
                
        except Exception as e:
            logger.error("link_add_failed", error=str(e))
            return False
    
    async def get_memory_context(
        self, 
        memory_id: str,
        max_depth: int = 2
    ) -> Dict:
        """Get full context for a memory including graph neighbors"""
        async with self.pool.acquire() as conn:
            memory = await conn.fetchrow("""
                SELECT id, content, metadata, sashf, created_at
                FROM memories
                WHERE id = $1
            """, memory_id)
            
            if not memory:
                return None
            
            neighbors = await conn.fetch("""
                SELECT m.*, ml.link_type, ml.weight
                FROM memory_links ml
                JOIN memories m ON ml.to_id = m.id
                WHERE ml.from_id = $1
                ORDER BY ml.weight DESC
                LIMIT 10
            """, memory_id)
            
            patterns = await conn.fetch("""
                SELECT p.*
                FROM memory_patterns p
                WHERE $1::text = ANY(p.example_ids)
            """, memory_id)
            
            return {
                'memory': dict(memory),
                'graph_context': [dict(n) for n in neighbors],
                'patterns': [dict(p) for p in patterns]
            }
    
    async def get_patterns(
        self, 
        min_occurrences: int = 3,
        promoted_only: bool = False
    ) -> List[Dict]:
        """Get detected patterns"""
        async with self.pool.acquire() as conn:
            query = """
                SELECT * FROM memory_patterns
                WHERE occurrences >= $1
            """
            params = [min_occurrences]
            
            if promoted_only:
                query += " AND promoted = true"
            
            query += " ORDER BY occurrences DESC, confidence DESC"
            
            patterns = await conn.fetch(query, *params)
            return [dict(p) for p in patterns]
    
    async def promote_pattern(self, pattern_id: str) -> bool:
        """Mark a pattern as important/promoted"""
        try:
            async with self.pool.adjust() as conn:
                await conn.execute("""
                    UPDATE memory_patterns
                    SET promoted = true, confidence = 1.0
                    WHERE id = $1
                """, pattern_id)
                
                logger.info("pattern_promoted", pattern_id=pattern_id)
                return True
        except Exception as e:
            logger.error("pattern_promote_failed", error=str(e))
            return False
    
    async def get_graph_stats(self) -> Dict:
        """Get graph statistics"""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT
                    (SELECT COUNT(*) FROM memories WHERE parent_id IS NULL) as total_memories,
                    (SELECT COUNT(*) FROM memories WHERE parent_id IS NOT NULL) as total_chunks,
                    (SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL) as memories_with_embeddings,
                    (SELECT COUNT(*) FROM memory_links) as total_links,
                    (SELECT COUNT(DISTINCT link_type) FROM memory_links) as link_types_used,
                    (SELECT COUNT(*) FROM memory_patterns) as patterns_detected
            """)

            return dict(stats) if stats else {}
