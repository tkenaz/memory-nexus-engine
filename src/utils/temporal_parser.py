"""Temporal query parser для обработки временных маркеров в запросах"""
from datetime import datetime, timedelta, date, timezone
from typing import Tuple, Dict, Optional
import re


class TemporalParser:
    """Парсер временных выражений для превращения их в фильтры по датам"""
    
    # Временные маркеры на русском и английском
    TEMPORAL_MARKERS = {
        # Относительные даты
        'сегодня': 0,
        'today': 0,
        'вчера': -1,
        'yesterday': -1,
        'позавчера': -2,
        'позапозавчера': -3,
        
        # Время суток
        'утром': 'morning',
        'утро': 'morning',
        'morning': 'morning',
        'днем': 'afternoon',
        'день': 'afternoon',
        'afternoon': 'afternoon',
        'вечером': 'evening',
        'вечер': 'evening',
        'evening': 'evening',
        'ночью': 'night',
        'ночь': 'night',
        'night': 'night',
    }
    
    TIME_RANGES = {
        'morning': (5, 12),    # 5:00 - 12:00
        'afternoon': (12, 18),  # 12:00 - 18:00
        'evening': (18, 22),   # 18:00 - 22:00
        'night': (22, 5),      # 22:00 - 5:00 (next day)
    }
    
    def parse_query(self, query: str) -> Tuple[str, Optional[Dict]]:
        """
        Парсит запрос, извлекает временные маркеры и возвращает очищенный запрос + фильтры
        
        Args:
            query: Исходный поисковый запрос
            
        Returns:
            (clean_query, date_filters) - очищенный запрос и фильтры для поиска
        """
        original_query = query
        query_lower = query.lower()
        
        # Инициализируем фильтры
        date_filter = None
        time_filter = None
        
        # Ищем дату
        for marker, days_offset in self.TEMPORAL_MARKERS.items():
            if isinstance(days_offset, int) and marker in query_lower:
                # Это дата
                target_date = date.today() + timedelta(days=days_offset)
                date_filter = target_date
                # Убираем маркер из запроса
                query = re.sub(r'\b' + re.escape(marker) + r'\b', '', query, flags=re.IGNORECASE)
                break
        
        # Ищем время суток
        for marker, time_key in self.TEMPORAL_MARKERS.items():
            if isinstance(time_key, str) and marker in query_lower:
                # Это время суток
                time_filter = time_key
                # Убираем маркер из запроса
                query = re.sub(r'\b' + re.escape(marker) + r'\b', '', query, flags=re.IGNORECASE)
                break
        
        # Очищаем запрос от лишних пробелов
        clean_query = ' '.join(query.split())
        
        # Формируем фильтры
        filters = None
        if date_filter or time_filter:
            filters = {}
            
            if date_filter:
                # Определяем диапазон дат
                if time_filter:
                    # Есть и дата и время
                    start_hour, end_hour = self.TIME_RANGES[time_filter]
                    
                    if time_filter == 'night' and end_hour < start_hour:
                        # Ночь переходит на следующий день
                        filters['created_at'] = {
                            '$gte': datetime.combine(date_filter, datetime.min.time().replace(hour=start_hour)).replace(tzinfo=timezone.utc),
                            '$lt': datetime.combine(date_filter + timedelta(days=1), datetime.min.time().replace(hour=end_hour)).replace(tzinfo=timezone.utc)
                        }
                    else:
                        filters['created_at'] = {
                            '$gte': datetime.combine(date_filter, datetime.min.time().replace(hour=start_hour)).replace(tzinfo=timezone.utc),
                            '$lt': datetime.combine(date_filter, datetime.min.time().replace(hour=end_hour)).replace(tzinfo=timezone.utc)
                        }
                else:
                    # Только дата - весь день
                    filters['created_at'] = {
                        '$gte': datetime.combine(date_filter, datetime.min.time()).replace(tzinfo=timezone.utc),
                        '$lt': datetime.combine(date_filter + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc)
                    }
            elif time_filter:
                # Только время суток сегодня
                start_hour, end_hour = self.TIME_RANGES[time_filter]
                today = date.today()

                if time_filter == 'night':
                    # Проверяем текущее время
                    now = datetime.now(timezone.utc)
                    if now.hour < 5:
                        # Сейчас ночь (после полуночи), ищем с вчера вечера
                        filters['created_at'] = {
                            '$gte': datetime.combine(today - timedelta(days=1), datetime.min.time().replace(hour=22)).replace(tzinfo=timezone.utc),
                            '$lt': datetime.combine(today, datetime.min.time().replace(hour=5)).replace(tzinfo=timezone.utc)
                        }
                    else:
                        # Сейчас день, ищем сегодняшнюю ночь
                        filters['created_at'] = {
                            '$gte': datetime.combine(today, datetime.min.time().replace(hour=22)).replace(tzinfo=timezone.utc),
                            '$lt': datetime.combine(today + timedelta(days=1), datetime.min.time().replace(hour=5)).replace(tzinfo=timezone.utc)
                        }
                else:
                    filters['created_at'] = {
                        '$gte': datetime.combine(today, datetime.min.time().replace(hour=start_hour)).replace(tzinfo=timezone.utc),
                        '$lt': datetime.combine(today, datetime.min.time().replace(hour=end_hour)).replace(tzinfo=timezone.utc)
                    }
        
        return clean_query, filters
    
    def parse_specific_date(self, query: str) -> Tuple[str, Optional[Dict]]:
        """
        Парсит конкретные даты типа "21 августа", "August 21", "21.08.2025"
        
        TODO: Реализовать парсинг конкретных дат
        """
        # Patterns for specific dates
        patterns = [
            # DD.MM.YYYY or DD/MM/YYYY
            r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b',
            # DD месяц YYYY
            r'\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s*(\d{4})?\b',
            # DD month YYYY
            r'\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?\b',
        ]
        
        # TODO: Implement actual parsing
        return query, None
