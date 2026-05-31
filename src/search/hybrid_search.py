"""Hybrid search implementation для существующей БД"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from FlagEmbedding import BGEM3FlagModel

from ..storage.postgres_adapter import PostgresStorage
from ..models.memory_simple import SearchQuery, SearchResult, Memory
from config.settings import settings
import structlog

logger = structlog.get_logger()


class HybridSearch:
    """Гибридный поиск с учетом особенностей multilingual-e5-large"""

    def __init__(self):
        self.storage = PostgresStorage()
        self.encoder = None
        self.reranker = None
        self._model_lock = asyncio.Lock()
        self._reranker_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize components"""
        await self.storage.initialize()
        # Lazy load модели при первом использовании
        logger.info("hybrid_search_initialized")
        
    async def _ensure_encoder(self):
        """Lazy load encoder model (BGE-M3)"""
        if self.encoder is None:
            async with self._model_lock:
                if self.encoder is None:  # Double check
                    logger.info("loading_encoder_model", model=settings.EMBEDDING_MODEL)
                    # Загружаем BGE-M3 в отдельном потоке
                    loop = asyncio.get_event_loop()
                    self.encoder = await loop.run_in_executor(
                        None,
                        lambda: BGEM3FlagModel(settings.EMBEDDING_MODEL, use_fp16=True)
                    )
                    logger.info("encoder_model_loaded", model="BGE-M3")

    async def _ensure_reranker(self):
        """Lazy load reranker model"""
        if self.reranker is None:
            async with self._reranker_lock:
                if self.reranker is None:  # Double check
                    logger.info("loading_reranker_model", model=settings.RERANK_MODEL)
                    loop = asyncio.get_event_loop()
                    self.reranker = await loop.run_in_executor(
                        None,
                        CrossEncoder,
                        settings.RERANK_MODEL
                    )
                    logger.info("reranker_model_loaded")

    async def encode_text(self, text: str) -> List[float]:
        """Encode text to embedding using BGE-M3"""
        await self._ensure_encoder()

        # BGE-M3 не требует prefix, работает напрямую
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

        # BGE-M3 возвращает dict с 'dense_vecs'
        embedding = result['dense_vecs'][0]
        return embedding.tolist()
        
    def _preprocess_temporal_query(self, query_text: str) -> Tuple[str, Optional[Dict]]:
        """Extract temporal markers from query and convert to metadata filters"""
        import re
        from datetime import datetime, timedelta, timezone

        clean_query = query_text
        metadata_filter = None

        # Get current time in UTC (timezone-aware)
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        logger.info("preprocessing_temporal_query", 
                   query=query_text,
                   current_time=now.isoformat())
        
        # Check for specific dates first (e.g., "21 августа", "August 21")
        date_patterns = [
            r'(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)',
            r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)',
            r'(\d{4})-(\d{2})-(\d{2})',  # ISO format
            r'(\d{2})\.(\d{2})\.(\d{4})'  # DD.MM.YYYY
        ]
        
        month_map = {
            'января': 1, 'january': 1, 'февраля': 2, 'february': 2,
            'марта': 3, 'march': 3, 'апреля': 4, 'april': 4,
            'мая': 5, 'may': 5, 'июня': 6, 'june': 6,
            'июля': 7, 'july': 7, 'августа': 8, 'august': 8,
            'сентября': 9, 'september': 9, 'октября': 10, 'october': 10,
            'ноября': 11, 'november': 11, 'декабря': 12, 'december': 12
        }
        
        for pattern in date_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 2:  # Day + Month name
                        day = int(match.group(1))
                        month_name = match.group(2).lower()
                        month = month_map.get(month_name)
                        if month:
                            year = now.year
                            # Create timezone-aware datetime in UTC
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

        # Check for month patterns (e.g., "октябрь", "в ноябре", "october 2025")
        if not metadata_filter:
            # Pattern для месяцев с опциональным годом
            month_patterns = [
                # Русские: "октябрь", "в октябре", "октябрь 2025"
                r'(?:в\s+)?(январе?|феврале?|марте?|апреле?|мае?|июне?|июле?|августе?|сентябре?|октябре?|ноябре?|декабре?)(?:\s+(\d{4}))?',
                # Английские: "october", "in october", "october 2025"
                r'(?:in\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+(\d{4}))?'
            ]

            month_name_map = {
                'январ': 1, 'january': 1,
                'феврал': 2, 'february': 2,
                'март': 3, 'march': 3,
                'апрел': 4, 'april': 4,
                'ма': 5, 'may': 5,
                'июн': 6, 'june': 6,
                'июл': 7, 'july': 7,
                'август': 8, 'august': 8,
                'сентябр': 9, 'september': 9,
                'октябр': 10, 'october': 10,
                'ноябр': 11, 'november': 11,
                'декабр': 12, 'december': 12
            }

            for pattern in month_patterns:
                match = re.search(pattern, query_text, re.IGNORECASE)
                if match:
                    try:
                        month_name = match.group(1).lower()
                        year_str = match.group(2) if len(match.groups()) > 1 else None

                        # Находим месяц по началу названия
                        month = None
                        for key, val in month_name_map.items():
                            if month_name.startswith(key):
                                month = val
                                break

                        if month:
                            year = int(year_str) if year_str else now.year

                            # Первый день месяца
                            month_start = datetime(year, month, 1, tzinfo=timezone.utc)

                            # Первый день следующего месяца
                            if month == 12:
                                month_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                            else:
                                month_end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

                            metadata_filter = {
                                'created_at': {
                                    '$gte': month_start,
                                    '$lt': month_end
                                }
                            }

                            clean_query = re.sub(pattern, '', query_text, flags=re.IGNORECASE).strip()
                            logger.info("month_filter_found",
                                       month=month,
                                       year=year,
                                       range=f"{month_start.date()} to {month_end.date()}")
                            break
                    except Exception as e:
                        logger.warning("month_parsing_failed", pattern=pattern, error=str(e))

        # Define temporal patterns
        if not metadata_filter:
            temporal_patterns = {
                r'\bсегодня\b|\btoday\b': {
                    'created_at': {'$gte': today_start}  # datetime object, not string
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
                
        # Also check for "утром", "вечером" etc
        time_of_day_patterns = {
            r'\bутром\b|\bmorning\b': (4, 12),
            r'\bднем\b|\bafternoon\b': (12, 18),
            r'\bвечером\b|\bevening\b': (18, 23),
            r'\bночью\b|\bnight\b': (0, 4)
        }
        
        for pattern, (hour_start, hour_end) in time_of_day_patterns.items():
            if re.search(pattern, clean_query, re.IGNORECASE):
                # Remove time of day words
                clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE).strip()
                # We could add hour filters here if metadata had time granularity
                break
        
        logger.info("temporal_preprocessing_result", 
                    original_query=query_text,
                    clean_query=clean_query,
                    metadata_filter=metadata_filter)
                    
        return clean_query, metadata_filter
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Hybrid search с fallback стратегией по рекомендации титана"""
        results = []
        semantic_results = []
        keyword_results = []
        
        # Preprocess query for temporal markers
        clean_query, temporal_filter = self._preprocess_temporal_query(query.query)
        
        # Merge temporal filter with existing metadata filters
        if temporal_filter:
            if query.metadata_filters:
                query.metadata_filters.update(temporal_filter)
            else:
                query.metadata_filters = temporal_filter
        
        # Use cleaned query for search
        search_query = clean_query if clean_query else query.query
        
        # 1. Всегда пробуем semantic search если включен
        if query.use_hybrid:
            try:
                embedding = await self.encode_text(search_query)
                semantic_raw = await self.storage.semantic_search(
                    embedding=embedding,
                    limit=query.limit * 2,  # Берем больше для объединения с keyword
                    min_similarity=query.min_similarity or settings.MIN_SIMILARITY,
                    metadata_filter=query.metadata_filters
                )
                
                # Конвертируем в SearchResult
                for res in semantic_raw:
                    memory = Memory(
                        id=res['id'],
                        content=res['content'],
                        embedding=None,  # Не возвращаем embedding в результатах
                        metadata=res['metadata'],
                        created_at=res['created_at'],
                        updated_at=res.get('updated_at', res['created_at'])
                    )
                    semantic_results.append(SearchResult(
                        memory=memory,
                        score=float(res['similarity']),
                        search_type="semantic"
                    ))
                    
                logger.info("semantic_search_complete", 
                          count=len(semantic_results),
                          max_score=max(r.score for r in semantic_results) if semantic_results else 0)
                
            except Exception as e:
                logger.error("semantic_search_failed", error=str(e))
                
        # 2. Определяем нужен ли keyword fallback (по рекомендации титана)
        need_keyword_fallback = False
        
        if not query.use_hybrid:
            # Если hybrid отключен - используем только keyword
            need_keyword_fallback = True
        elif not semantic_results:
            # Если semantic ничего не нашел
            need_keyword_fallback = True
            logger.info("keyword_fallback_reason", reason="no_semantic_results")
        elif max(r.score for r in semantic_results) < settings.KEYWORD_FALLBACK_THRESHOLD:
            # Если лучший semantic score < threshold (рекомендация титана)
            need_keyword_fallback = True
            logger.info("keyword_fallback_reason", 
                       reason="low_similarity", 
                       max_score=max(r.score for r in semantic_results),
                       threshold=settings.KEYWORD_FALLBACK_THRESHOLD)
                       
        # 3. Выполняем keyword search если нужен
        if need_keyword_fallback:
            logger.info("executing_keyword_search")
            
            keyword_raw = await self.storage.keyword_search(
                query=search_query,
                limit=query.limit,
                metadata_filter=query.metadata_filters
            )
            
            for res in keyword_raw:
                memory = Memory(
                    id=res['id'],
                    content=res['content'],
                    embedding=None,
                    metadata=res['metadata'],
                    created_at=res['created_at'],
                    updated_at=res.get('updated_at', res['created_at'])
                )
                keyword_results.append(SearchResult(
                    memory=memory,
                    score=float(res['rank']),
                    search_type="keyword"
                ))
                
        # 4. Объединяем результаты
        if semantic_results and keyword_results:
            # Hybrid mode - нужно правильно объединить
            # Убираем дубликаты и применяем веса (0.7 semantic + 0.3 keyword)
            seen_ids = set()
            
            # Сначала добавляем semantic с весом
            for res in semantic_results:
                if res.memory.id not in seen_ids:
                    res.score *= settings.SEMANTIC_WEIGHT  # Вес semantic из настроек
                    results.append(res)
                    seen_ids.add(res.memory.id)
                    
            # Потом keyword с весом
            for res in keyword_results:
                if res.memory.id not in seen_ids:
                    res.score *= settings.KEYWORD_WEIGHT  # Вес keyword из настроек
                    results.append(res)
                    seen_ids.add(res.memory.id)
                else:
                    # Если уже есть - добавляем keyword score к существующему
                    for existing in results:
                        if existing.memory.id == res.memory.id:
                            existing.score += res.score * settings.KEYWORD_WEIGHT
                            existing.search_type = "hybrid"  # Помечаем как найденное обоими методами
                            break
        else:
            # Используем что есть
            results = semantic_results + keyword_results
            
        # Сортируем по score и обрезаем до limit
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:query.limit]

        # Apply reranking if enabled
        if query.use_reranking and settings.USE_RERANKING and results:
            logger.info("applying_reranking", results_count=len(results))
            results = await self.rerank_results(
                query=search_query,
                results=results,
                top_k=settings.RERANK_TOP_K
            )

        logger.info("hybrid_search_complete",
                   total_results=len(results),
                   search_types=list(set(r.search_type for r in results)))

        return results
        
    def _normalize_and_merge_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores from different search types"""
        # Группируем по типу
        by_type = {}
        for r in results:
            by_type.setdefault(r.search_type, []).append(r)
            
        # Нормализуем scores внутри каждого типа
        normalized = []
        
        for search_type, type_results in by_type.items():
            if not type_results:
                continue
                
            scores = [r.score for r in type_results]
            min_score = min(scores)
            max_score = max(scores)
            
            # Avoid division by zero
            if max_score == min_score:
                weight = settings.SEMANTIC_WEIGHT if search_type == "semantic" else settings.KEYWORD_WEIGHT
                for r in type_results:
                    r.score = weight
                    normalized.append(r)
            else:
                # Normalize to 0-1
                for r in type_results:
                    norm_score = (r.score - min_score) / (max_score - min_score)
                    # Apply weight based on search type
                    if search_type == "semantic":
                        r.score = norm_score * settings.SEMANTIC_WEIGHT
                    else:
                        r.score = norm_score * settings.KEYWORD_WEIGHT
                    normalized.append(r)
                    
        return normalized
        
    async def rerank_results(self, query: str, results: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """Rerank results using cross-encoder"""
        if not results:
            return results

        await self._ensure_reranker()

        logger.info("reranking_results",
                   input_count=len(results),
                   top_k=top_k)

        # Создаем пары (query, content) для cross-encoder
        pairs = [[query, result.memory.content] for result in results]

        # Получаем scores от cross-encoder
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self.reranker.predict(pairs)
        )

        # Обновляем scores в результатах
        for result, score in zip(results, scores):
            result.score = float(score)
            # Помечаем что результат прошел через reranking
            if result.search_type != "hybrid":
                result.search_type = f"{result.search_type}_reranked"
            else:
                result.search_type = "hybrid_reranked"

        # Сортируем по новым scores и берем top_k
        reranked = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]

        logger.info("reranking_complete",
                   output_count=len(reranked),
                   max_score=max(r.score for r in reranked) if reranked else 0,
                   min_score=min(r.score for r in reranked) if reranked else 0)

        return reranked
        
    async def close(self):
        """Cleanup resources"""
        await self.storage.close()
