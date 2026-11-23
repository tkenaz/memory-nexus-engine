"""Enhanced PostgreSQL storage with SASH-F structure and REAL semantic search"""
import json
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta, timezone
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

from ..core.encoder import load_bge_m3
from pgvector.asyncpg import register_vector

from ..models.sashf import SceneFrame, MemoryLink, LinkType, Pattern, Affect
from ..core.sashf_parser import SASHFParser
from ..search.rank_fusion import rrf_fuse
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


# Month names for date phrases in queries: Russian genitive ("21 августа") and
# nominative ("август 2025"), English full and abbreviated.
_MONTHS = {}
for _n, _forms in enumerate([
    ('января', 'январь', 'january', 'jan'), ('февраля', 'февраль', 'february', 'feb'),
    ('марта', 'март', 'march', 'mar'), ('апреля', 'апрель', 'april', 'apr'),
    ('мая', 'май', 'may'), ('июня', 'июнь', 'june', 'jun'),
    ('июля', 'июль', 'july', 'jul'), ('августа', 'август', 'august', 'aug'),
    ('сентября', 'сентябрь', 'september', 'sep'), ('октября', 'октябрь', 'october', 'oct'),
    ('ноября', 'ноябрь', 'november', 'nov'), ('декабря', 'декабрь', 'december', 'dec'),
], start=1):
    for _form in _forms:
        _MONTHS[_form] = _n
# Longest first, so "мая" does not shadow "май" and "march" wins over "mar".
_MONTH_NAMES_PATTERN = '|'.join(sorted(_MONTHS, key=len, reverse=True))


def _query_timezone():
    """Time zone for date phrases in queries: the machine's local zone."""
    return datetime.now().astimezone().tzinfo


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
                            model = load_bge_m3(settings.EMBEDDING_MODEL, use_fp16=True)
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
        """Turn a date phrase in the query into a created_at filter.

        Returns (query without the date phrase, filter or None). Days and periods
        are taken in the machine's local time zone: "today" is the user's today.
        Order matters: the most specific form wins (a full date before a month).
        """
        tz = _query_timezone()
        now = datetime.now(tz)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        def day_filter(year: int, month: int, day: int) -> Dict:
            start = datetime(year, month, day, tzinfo=tz)
            return {'created_at': {'$gte': start, '$lt': start + timedelta(days=1)}}

        def month_filter(year: int, month: int) -> Dict:
            _, last_day = calendar.monthrange(year, month)
            return {'created_at': {
                '$gte': datetime(year, month, 1, tzinfo=tz),
                '$lte': datetime(year, month, last_day, 23, 59, 59, tzinfo=tz),
            }}

        def strip(pattern: str) -> str:
            cleaned = re.sub(pattern, ' ', query_text, flags=re.IGNORECASE)
            return re.sub(r'\s+', ' ', cleaned).strip()

        months = _MONTH_NAMES_PATTERN
        absolute = [
            # 21 августа 2025 / 21 August 2025
            (rf'\b(\d{{1,2}})\s+({months})\s+(\d{{4}})\b',
             lambda m: day_filter(int(m[3]), _MONTHS[m[2].lower()], int(m[1]))),
            # August 21, 2025 / August 21 2025
            (rf'\b({months})\s+(\d{{1,2}}),?\s+(\d{{4}})\b',
             lambda m: day_filter(int(m[3]), _MONTHS[m[1].lower()], int(m[2]))),
            # 2025-10-03
            (r'\b(\d{4})-(\d{2})-(\d{2})\b',
             lambda m: day_filter(int(m[1]), int(m[2]), int(m[3]))),
            # 03.10.2025
            (r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b',
             lambda m: day_filter(int(m[3]), int(m[2]), int(m[1]))),
            # октябрь 2025 / октября 2025 / October 2025
            (rf'\b({months})\s+(\d{{4}})\b',
             lambda m: month_filter(int(m[2]), _MONTHS[m[1].lower()])),
            # 21 августа / 5 march (this year)
            (rf'\b(\d{{1,2}})\s+({months})\b',
             lambda m: day_filter(now.year, _MONTHS[m[2].lower()], int(m[1]))),
            # March 5 (this year)
            (rf'\b({months})\s+(\d{{1,2}})\b',
             lambda m: day_filter(now.year, _MONTHS[m[1].lower()], int(m[2]))),
        ]
        for pattern, build in absolute:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if not match:
                continue
            try:
                metadata_filter = build(match)
            except (ValueError, KeyError):
                continue  # e.g. 31.02.2025: not a date, keep looking
            return strip(pattern), metadata_filter

        week_start = today_start - timedelta(days=today_start.weekday())
        month_start = today_start.replace(day=1)
        relative = [
            (r'\bпозавчера\b|\bday before yesterday\b',
             {'created_at': {'$gte': today_start - timedelta(days=2),
                             '$lt': today_start - timedelta(days=1)}}),
            (r'\bсегодня\b|\btoday\b',
             {'created_at': {'$gte': today_start}}),
            (r'\bвчера\b|\byesterday\b',
             {'created_at': {'$gte': today_start - timedelta(days=1), '$lt': today_start}}),
            (r'\bэта неделя\b|\bна этой неделе\b|\bthis week\b',
             {'created_at': {'$gte': week_start}}),
            (r'\bпрошлая неделя\b|\bна прошлой неделе\b|\blast week\b',
             {'created_at': {'$gte': week_start - timedelta(days=7), '$lt': week_start}}),
            (r'\bэтот месяц\b|\bв этом месяце\b|\bthis month\b',
             {'created_at': {'$gte': month_start}}),
            (r'\bпрошлый месяц\b|\bв прошлом месяце\b|\blast month\b',
             {'created_at': {'$gte': (month_start - timedelta(days=1)).replace(day=1),
                             '$lt': month_start}}),
        ]
        for pattern, metadata_filter in relative:
            if re.search(pattern, query_text, re.IGNORECASE):
                return strip(pattern), metadata_filter

        return query_text, None

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

        # Time-of-day bucket in the machine's local time: an MCP stdio server runs
        # on the user's machine, and "morning" means the user's morning, not UTC's.
        hour = datetime.now().astimezone().hour
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

            # Keep tag catalog in sync (embed unseen tags, bump counts)
            meta_tags = (metadata or {}).get('tags')
            if isinstance(meta_tags, list):
                asyncio.create_task(self._upsert_tag_embeddings(meta_tags))

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

        for r in results:
            parent_id = r.get('parent_id')
            effective_id = parent_id if parent_id else r['id']

            if effective_id in resolved:
                # Keep the higher score
                if float(r.get('similarity', 0)) > float(resolved[effective_id].get('similarity', 0)):
                    resolved[effective_id]['similarity'] = r['similarity']
                continue

            if parent_id and parent_id in parents_content:
                # Replace chunk with parent data but keep chunk's score.
                # No multi-chunk bonus: long memories produce more chunks and
                # would systematically outrank short focused ones.
                parent = parents_content[parent_id]

                resolved[effective_id] = {
                    'id': parent['id'],
                    'content': parent['content'],
                    'metadata': parent.get('metadata', r.get('metadata')),
                    'sashf': parent.get('sashf', r.get('sashf')),
                    'created_at': parent.get('created_at', r.get('created_at')),
                    'similarity': float(r.get('similarity', 0)),
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
        min_similarity: Optional[float] = None
    ) -> List[Dict]:
        """
        Search pipeline:
        1. Semantic search: vector similarity over chunks, resolved to parents.
        2. Keyword search: PostgreSQL FTS (ts_rank_cd), always on, not a fallback.
        3. Tag catalog: the query embedding is matched against embeddings of
           metadata tags; the top tags expand into their memories.
        4. Reciprocal Rank Fusion of the three ranked lists. Dense scores are
           too compressed (mostly ~0.7-0.8) for an absolute threshold to work,
           so only ranks are fused.
        5. Metadata/tag match and graph expansion on top of the fused list,
           then a recency blend.

        Temporal markers in the query ("yesterday", "last week", "October 2025")
        become date filters on all engines.
        """
        if min_similarity is None:
            min_similarity = settings.MIN_SIMILARITY
        
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
            
            # Helper to build temporal WHERE clauses (used by all engines below)
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

            # Both the semantic and the tag engine need the query embedding; compute it once.
            embedding_array = None
            try:
                query_embedding = await self.encode_text(search_query)
                embedding_array = np.array(query_embedding, dtype=np.float32)
            except Exception as e:
                logger.error("query_encoding_error", error=str(e))

            # STEP 1: SEMANTIC SEARCH (PRIMARY)
            error_msg = None
            try:
                if embedding_array is None:
                    raise RuntimeError("query embedding unavailable")

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
                # Don't re-raise, keyword search still runs below
                results = []

            # STEP 1.5: RESOLVE CHUNKS TO PARENTS before fusion, so both
            # engines rank the same units (parent memories, not chunks)
            semantic_ranked = await self._resolve_to_parents(results, conn)

            # STEP 2: KEYWORD SEARCH (always on — RRF fusion, not fallback)
            # Detect language for FTS config: Latin-only → 'english', otherwise → 'russian'
            fts_config = 'english' if re.match(r'^[a-zA-Z0-9\s\-_.]+$', search_query) else 'russian'

            # OR semantics: plainto_tsquery ANDs all words and kills recall on
            # multi-word queries (a document must contain every word). Fusion only
            # needs a ranked list, and ts_rank_cd already ranks documents with
            # more matching words higher.
            keyword_query_parts = [f"""
                WITH q AS (
                    SELECT replace(plainto_tsquery('{fts_config}', $1)::text,
                                   ' & ', ' | ')::tsquery AS tsq
                )
                SELECT
                    id,
                    content,
                    metadata,
                    sashf,
                    created_at,
                    parent_id,
                    chunk_index,
                    ts_rank_cd(to_tsvector('{fts_config}', content), q.tsq) as similarity
                FROM memories, q
                WHERE to_tsvector('{fts_config}', content) @@ q.tsq
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
            keyword_params.append(limit * 3)  # deeper rank list for fusion

            full_keyword_query = '\n'.join(keyword_query_parts)
            try:
                keyword_rows = await conn.fetch(full_keyword_query, *keyword_params)
                keyword_ranked = [dict(r) for r in keyword_rows]
            except Exception as e:
                logger.error("keyword_search_error", error=str(e))
                keyword_ranked = []

            # STEP 2.1: TAG ENGINE (catalog). The query is matched against the tag
            # vocabulary (short-to-short: the symmetric case where a bi-encoder
            # works well); the top tags expand into memories. Catches anchors that
            # live only in metadata (a project name in the tags but not in the text).
            tag_ranked = []
            if embedding_array is not None:
                try:
                    tag_rows = await conn.fetch(f"""
                        SELECT tag, 1.0 - (embedding <=> $1) AS sim
                        FROM tag_embeddings
                        WHERE embedding IS NOT NULL
                        ORDER BY embedding <=> $1
                        LIMIT {int(settings.TAG_TOP_K)}
                    """, embedding_array)
                    matched_tags = {r['tag']: float(r['sim']) for r in tag_rows
                                    if float(r['sim']) >= settings.TAG_MIN_SIMILARITY}
                    if matched_tags:
                        logger.info("tag_engine_matched", tags=matched_tags)
                        tag_mem_parts = ["""
                            SELECT m.id, m.content, m.metadata, m.sashf, m.created_at,
                                   m.parent_id, m.chunk_index,
                                   max(ts.sim) AS similarity
                            FROM memories m
                            JOIN LATERAL jsonb_array_elements_text(
                                CASE WHEN jsonb_typeof(m.metadata->'tags') = 'array'
                                     THEN m.metadata->'tags' ELSE '[]'::jsonb END
                            ) AS t(tag) ON TRUE
                            JOIN unnest($1::text[], $2::float8[]) AS ts(tag, sim)
                              ON ts.tag = lower(t.tag)
                            WHERE m.parent_id IS NULL
                        """]
                        tag_mem_params = [list(matched_tags.keys()), list(matched_tags.values())]
                        tag_pc = 3
                        t_parts, tag_pc = _temporal_clauses(tag_pc, tag_mem_params)
                        tag_mem_parts.extend(t_parts)
                        tag_mem_parts.append(f"""
                            GROUP BY m.id, m.content, m.metadata, m.sashf, m.created_at,
                                     m.parent_id, m.chunk_index
                            ORDER BY similarity DESC, m.created_at DESC
                            LIMIT ${tag_pc}
                        """)
                        tag_mem_params.append(limit * 3)
                        tag_mem_rows = await conn.fetch('\n'.join(tag_mem_parts), *tag_mem_params)
                        tag_ranked = [dict(r) for r in tag_mem_rows]
                except Exception as e:
                    logger.error("tag_engine_error", error=str(e))

            # STEP 2.1b: METADATA MATCHES as two more ranked lists for RRF
            # (exact tag match; query as a substring of the metadata JSON), newest
            # first. They used to be appended after fusion with fixed scores of
            # 0.8 / 0.5, which sat on a different scale from the fused (0, 1] scores
            # and could outrank direct matches.
            tag_exact_ranked: List[Dict] = []
            metadata_ranked: List[Dict] = []

            tag_query_parts = ["""
                SELECT id, content, metadata, sashf, created_at, parent_id, chunk_index
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
                tag_exact_ranked = [dict(r) for r in
                                    await conn.fetch('\n'.join(tag_query_parts), *tag_params)]
            except Exception:
                pass  # metadata->'tags' may not be a JSON array

            # Escape LIKE wildcards so '%' and '_' in the query match literally.
            like_query = (search_query.replace('\\', '\\\\')
                          .replace('%', '\\%').replace('_', '\\_'))
            sub_query_parts = ["""
                SELECT id, content, metadata, sashf, created_at, parent_id, chunk_index
                FROM memories
                WHERE metadata::text ILIKE $1 ESCAPE '\\'
                AND parent_id IS NULL
            """]
            sub_params = [f'%{like_query}%']
            sub_pc = 2
            temporal_parts2, sub_pc = _temporal_clauses(sub_pc, sub_params)
            sub_query_parts.extend(temporal_parts2)
            sub_query_parts.append(f'ORDER BY created_at DESC LIMIT ${sub_pc}')
            sub_params.append(limit)
            metadata_ranked = [dict(r) for r in
                               await conn.fetch('\n'.join(sub_query_parts), *sub_params)]

            # STEP 2.2: RECIPROCAL RANK FUSION
            # ts_rank, cosine and tag similarity live on incomparable scales, so
            # only ranks are fused. Scores are normalised to (0, 1] so downstream
            # scoring (recency blend, metadata 0.8/0.5, graph cap at 0.8*max)
            # keeps its scale.
            engine_lists = [(name, lst) for name, lst in
                            (('semantic', semantic_ranked),
                             ('keyword', keyword_ranked),
                             ('tags', tag_ranked),
                             ('tag_exact', tag_exact_ranked),
                             ('metadata', metadata_ranked)) if lst]
            results = rrf_fuse(engine_lists, k=settings.RRF_K)

            if len(engine_lists) > 1:
                search_method = "rrf_hybrid"
            elif engine_lists:
                search_method = engine_lists[0][0]

            logger.info("rrf_fusion_complete",
                       semantic_count=len(semantic_ranked),
                       keyword_count=len(keyword_ranked),
                       tag_count=len(tag_ranked),
                       fused_count=len(results))

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
                            SELECT m.id, m.content, m.metadata, m.sashf, m.created_at,
                                   0 as depth, 1.0::float8 as weight
                            FROM memories m
                            WHERE m.id = $1
                            AND m.parent_id IS NULL

                            UNION ALL

                            SELECT m.id, m.content, m.metadata, m.sashf, m.created_at, g.depth + 1,
                                   (g.weight * ml.weight * 0.8)::float8
                            FROM graph g
                            -- Links are stored once (e.g. 'after': newer -> earlier) but walked
                            -- in both directions, so a hit reaches both what came before it
                            -- and what came after it.
                            JOIN memory_links ml ON g.id IN (ml.from_id, ml.to_id)
                            JOIN memories m ON m.id = CASE WHEN ml.from_id = g.id
                                                           THEN ml.to_id ELSE ml.from_id END
                            WHERE g.depth < $2
                            AND ml.link_type IN ('similar', 'after', 'caused_by')
                            AND m.parent_id IS NULL
                        )
                        SELECT DISTINCT id, content, metadata, sashf, created_at,
                               MIN(depth) as depth, MAX(weight) as weight
                        FROM graph
                        WHERE depth > 0
                        GROUP BY id, content, metadata, sashf, created_at
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

                        # Graph neighbours are capped at 80% of the best direct score.
                        # They can outrank weaker direct matches; capping them below
                        # every direct match would push them out of any top-N.
                        scaled_similarity = float(gr['weight']) * max_semantic_sim * 0.8
                        results.append({
                            'id': gr['id'],
                            'content': gr['content'],
                            'similarity': scaled_similarity,
                            'metadata': gr['metadata'],
                            'sashf': gr['sashf'],
                            'created_at': gr['created_at'],  # so the recency blend applies
                            'source': 'graph'  # Mark as graph-expanded
                        })

            # De-duplicate and apply recency boost
            seen = set()
            unique_results = []
            now = datetime.now(timezone.utc)

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
                            created = created.replace(tzinfo=timezone.utc)

                        days_old = (now - created).days
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
    
    async def _upsert_tag_embeddings(self, tags: List[str]):
        """Sync the tag catalog: embed unseen tags, increment counters.

        Tags are normalised with lower(), as in build_tag_index.py and the tag
        search engine.
        """
        try:
            norm = sorted({t.strip().lower() for t in tags
                           if isinstance(t, str) and t.strip()})
            if not norm:
                return
            async with self.pool.acquire() as conn:
                known = {r['tag'] for r in await conn.fetch(
                    "SELECT tag FROM tag_embeddings WHERE tag = ANY($1)", norm)}

                seen = [t for t in norm if t in known]
                if seen:
                    await conn.execute("""
                        UPDATE tag_embeddings
                        SET memory_count = memory_count + 1, updated_at = now()
                        WHERE tag = ANY($1)
                    """, seen)

                for tag in (t for t in norm if t not in known):
                    emb = await self.encode_text(tag)
                    await conn.execute("""
                        INSERT INTO tag_embeddings (tag, embedding, memory_count)
                        VALUES ($1, $2, 1)
                        ON CONFLICT (tag) DO UPDATE
                        SET memory_count = tag_embeddings.memory_count + 1,
                            updated_at = now()
                    """, tag, np.array(emb, dtype=np.float32))
        except Exception as e:
            logger.error("tag_upsert_error", error=str(e))

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
                
                # Temporal 'after' links are created by the auto_temporal_links
                # trigger (migration 002), not here.

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
