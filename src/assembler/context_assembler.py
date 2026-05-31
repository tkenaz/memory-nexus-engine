"""
Context Assembler - MVP версия для упаковки памяти в промпты
"""

import tiktoken
import heapq
import re
from datetime import datetime as dt, timezone
from typing import List, Dict, Any, Set, Optional
import logging

logger = logging.getLogger(__name__)

# Инициализация токенизатора
enc = tiktoken.get_encoding("cl100k_base")

# Русские и английские стоп-слова
STOPWORDS = {
    # Русские
    "и", "в", "на", "по", "с", "или", "для", "как", "что", "это", "не", "но", "к", "у", "из", "о",
    "то", "все", "она", "так", "его", "только", "мне", "было", "вот", "от", "меня", "еще", "нет",
    # Английские  
    "the", "is", "at", "which", "on", "and", "a", "an", "as", "are", "was", "were", "of", "to",
    "in", "it", "that", "this", "for", "with", "be", "have", "has", "had", "do", "does"
}


class ContextAssembler:
    """Упаковщик контекста для промптов - MVP версия"""
    
    def __init__(self, budget: int = 50000):
        """
        Args:
            budget: Максимальное количество токенов для контекста
        """
        self.budget = budget
        self.enc = enc
        logger.info(f"ContextAssembler initialized with budget={budget}")
        
    def _tokenize_text(self, text: str) -> Set[str]:
        """Извлекаем значимые токены из текста"""
        # Простая токенизация с фильтрацией
        tokens = re.findall(r'\w+', text.lower())
        return {t for t in tokens if t not in STOPWORDS and len(t) > 2}
    
    def _parse_datetime(self, dt_value: Any) -> dt:
        """Безопасный парсинг datetime из разных форматов"""
        if isinstance(dt_value, dt):
            return dt_value
        if isinstance(dt_value, str):
            # Пробуем разные форматы
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ]:
                try:
                    # Убираем временную зону если есть
                    clean_dt = dt_value.replace('Z', '').split('+')[0]
                    parsed = dt.strptime(clean_dt, fmt)
                    # Make timezone-aware
                    return parsed.replace(tzinfo=timezone.utc)
                except:
                    continue
        return dt.now(timezone.utc)  # fallback
    
    def _score(self, memory: Dict[str, Any], query_tokens: Set[str], now: dt) -> float:
        """
        Мультифакторная оценка релевантности
        
        Args:
            memory: Словарь с данными памяти
            query_tokens: Токены из поискового запроса
            now: Текущее время для расчета recency
            
        Returns:
            Скор от 0 до 1
        """
        # 1. Семантическая близость (из memory-nexus)
        similarity = memory.get('score', 0)
        
        # 2. Временная близость (последние 30 дней важнее)
        created_at = self._parse_datetime(memory.get('created_at', now))
        days_ago = (now - created_at).days
        recency = max(0, 30 - days_ago) / 30 if days_ago < 30 else 0
        
        # 3. Пересечение ключевых слов
        content = memory.get('content', '')
        memory_tokens = self._tokenize_text(content)
        
        # Учитываем и метаданные в токенах
        metadata = memory.get('metadata', {})
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, str):
                    memory_tokens.update(self._tokenize_text(value))
        
        # Считаем overlap
        overlap_count = len(memory_tokens & query_tokens)
        key_overlap = overlap_count / (len(query_tokens) or 1)
        
        # 4. Бонус за определенные типы (если есть)
        type_bonus = 0
        if metadata.get('type') in ['claude_emotion_state', 'technical_discovery', 'important']:
            type_bonus = 0.1
        
        # Финальный скор с весами
        # Увеличиваем вес ключевых слов для лучшей релевантности
        final_score = (
            0.40 * similarity +      # Меньше вес семантике
            0.25 * recency +         # Меньше времени 
            0.30 * key_overlap +     # Больше вес точным совпадениям
            0.05 * type_bonus
        )
        
        logger.debug(f"Score components: sim={similarity:.2f}, rec={recency:.2f}, "
                    f"key={key_overlap:.2f}, type={type_bonus:.2f} -> {final_score:.2f}")
        
        return final_score
    
    def _format_memory(self, memory: Dict[str, Any], include_metadata: bool = True) -> str:
        """Форматирование одной памяти для вывода"""
        content = memory.get('content', '').strip()
        metadata = memory.get('metadata', {})
        created_at = memory.get('created_at', '')
        
        # Обрезаем дату до дня
        if created_at:
            date_str = str(created_at)[:10]
        else:
            date_str = "unknown"
        
        # Форматируем с метаданными или без
        if include_metadata and metadata and isinstance(metadata, dict):
            # Выбираем только важные метаданные
            important_keys = ['type', 'context', 'emotion', 'trigger', 'language']
            important_meta = {k: v for k, v in metadata.items() 
                            if k in important_keys and v}
            
            if important_meta:
                meta_str = ', '.join(f'{k}:{v}' for k, v in important_meta.items())
                return f"[{date_str}] {content} ({meta_str})"
            
        return f"[{date_str}] {content}"
    
    def assemble(
        self, 
        query: str, 
        memories: List[Dict], 
        include_metadata: bool = True,
        include_stats: bool = True
    ) -> str:
        """
        Собираем контекст из memories
        
        Args:
            query: Поисковый запрос
            memories: Список найденных memories
            include_metadata: Включать ли метаданные
            include_stats: Добавлять ли статистику в конец
            
        Returns:
            Отформатированный контекст для промпта
        """
        if not memories:
            return "### Релевантные воспоминания\n\nНичего не найдено."

        now = dt.now(timezone.utc)
        query_tokens = self._tokenize_text(query)
        
        logger.info(f"Assembling context for query: '{query[:50]}...', "
                   f"found {len(memories)} memories, query tokens: {len(query_tokens)}")
        
        # Ранжируем memories по скору
        scored_memories = [
            (self._score(m, query_tokens, now), m) 
            for m in memories
        ]
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        
        # Берем топ N (не больше 50)
        top_memories = scored_memories[:min(50, len(scored_memories))]
        
        # Упаковываем в контекст с учетом бюджета токенов
        lines = []
        used_tokens = 0
        
        # Заголовок
        header = "### Контекст из памяти\n\n"
        header_tokens = len(self.enc.encode(header))
        used_tokens += header_tokens
        
        # Резервируем место для статистики
        stats_reserve = 100 if include_stats else 0
        effective_budget = self.budget - stats_reserve
        
        # Добавляем memories пока влезают в бюджет
        included_count = 0
        for score, memory in top_memories:
            # Форматируем память
            line = self._format_memory(memory, include_metadata)
            
            # Считаем токены
            line_tokens = len(self.enc.encode(line + "\n"))
            
            # Проверяем бюджет
            if used_tokens + line_tokens > effective_budget:
                # Пробуем сократить если память длинная
                content = memory.get('content', '')
                if len(content) > 200:
                    truncated = content[:197] + "..."
                    memory_truncated = memory.copy()
                    memory_truncated['content'] = truncated
                    line = self._format_memory(memory_truncated, include_metadata)
                    line_tokens = len(self.enc.encode(line + "\n"))
                    
                    if used_tokens + line_tokens > effective_budget:
                        logger.debug(f"Budget exceeded at memory {included_count}, "
                                   f"used={used_tokens}, limit={effective_budget}")
                        break
            
            # Финальная проверка перед добавлением
            if used_tokens + line_tokens > effective_budget:
                logger.info(f"Stopping at {included_count} memories due to budget. "
                           f"Would use {used_tokens + line_tokens}, limit is {effective_budget}")
                break
                
            lines.append(line)
            used_tokens += line_tokens
            included_count += 1
        
        # Собираем результат
        result = header + "\n".join(lines)
        
        # Добавляем статистику если нужно
        if include_stats:
            stats = (
                f"\n\n---\n"
                f"_Статистика: найдено {len(memories)}, "
                f"отобрано {len(top_memories)}, "
                f"включено {included_count}, "
                f"токенов {used_tokens}/{self.budget}_"
            )
            result += stats
        
        logger.info(f"Context assembled: {included_count} memories, "
                   f"{used_tokens} tokens used")
        
        return result
    
    def compress_cluster(self, memories: List[Dict], max_tokens: int = 1000) -> str:
        """
        Компрессия группы похожих memories (для V1)
        
        TODO: Implement LLM-based compression
        """
        # Пока просто берем первые N
        result = []
        tokens = 0
        
        for memory in memories:
            line = self._format_memory(memory, include_metadata=False)
            line_tokens = len(self.enc.encode(line))
            
            if tokens + line_tokens > max_tokens:
                break
                
            result.append(line)
            tokens += line_tokens
        
        return "\n".join(result)
