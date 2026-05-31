"""Query Intent Detection for better context understanding"""
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class QueryIntent(Enum):
    """Types of query intents"""
    TEMPORAL = "temporal"       # "последние", "вчера", "на этой неделе"
    PERSON = "person"           # names of people, e.g. "Alice", "Bob"
    PROJECT = "project"         # names of projects, e.g. "MyApp", "Atlas"
    TECHNICAL = "technical"     # "код", "баг", "архитектура"
    EMOTIONAL = "emotional"     # "устала", "радость", "грустно"
    FACTUAL = "factual"        # конкретные факты, числа
    MIXED = "mixed"            # смешанный тип


class QueryIntentDetector:
    """Detects the intent and context of search queries"""

    def __init__(self):
        # Known entities — customize for your own world.
        # People whose names should trigger PERSON intent, e.g. {'alice', 'боб'}.
        self.persons: set = set()

        # Projects whose names should trigger PROJECT intent, e.g. {'my-app', 'атлас'}.
        self.projects: set = {'memory-nexus', 'memory nexus'}

        self.technical_markers = {
            'код', 'code', 'баг', 'bug', 'ошибка', 'error',
            'архитектура', 'architecture', 'api', 'mcp',
            'функция', 'function', 'класс', 'class',
            'тест', 'test', 'debug', 'fix', 'починить'
        }

        self.temporal_markers = {
            'последние', 'последний', 'recent', 'latest',
            'сегодня', 'today', 'вчера', 'yesterday',
            'неделя', 'week', 'месяц', 'month',
            'утром', 'вечером', 'днем', 'ночью'
        }

        self.emotional_markers = {
            'устал', 'устала', 'tired', 'выгорел', 'выгорела',
            'радость', 'счастье', 'happy', 'грустно', 'sad',
            'злой', 'злая', 'angry', 'спокойно', 'calm',
            'волнуюсь', 'worried', 'excited', 'взволнован'
        }

    def detect_intent(self, query: str) -> Tuple[QueryIntent, Dict[str, any]]:
        """
        Detect query intent and extract context

        Returns:
            (intent, context) where context contains:
            - entities: detected entities
            - boost_factors: what to prioritize
            - filters: additional filters to apply
        """
        query_lower = query.lower()
        context = {
            'entities': {
                'persons': [],
                'projects': [],
                'technical': [],
                'temporal': [],
                'emotional': []
            },
            'boost_factors': {},
            'filters': {}
        }

        intents = []

        # Detect persons
        for person in self.persons:
            if person in query_lower:
                context['entities']['persons'].append(person)
                intents.append(QueryIntent.PERSON)

        # Detect projects
        for project in self.projects:
            if project in query_lower:
                context['entities']['projects'].append(project)
                intents.append(QueryIntent.PROJECT)

        # Detect technical terms
        for tech in self.technical_markers:
            if tech in query_lower:
                context['entities']['technical'].append(tech)
                intents.append(QueryIntent.TECHNICAL)

        # Detect temporal markers
        for temporal in self.temporal_markers:
            if temporal in query_lower:
                context['entities']['temporal'].append(temporal)
                intents.append(QueryIntent.TEMPORAL)

        # Detect emotional markers
        for emotion in self.emotional_markers:
            if emotion in query_lower:
                context['entities']['emotional'].append(emotion)
                intents.append(QueryIntent.EMOTIONAL)

        # Detect factual queries (numbers, dates, specific facts)
        if re.search(r'\d+', query) or re.search(r'\$|€|₽', query):
            intents.append(QueryIntent.FACTUAL)

        # Determine primary intent
        if not intents:
            primary_intent = QueryIntent.FACTUAL  # Default
        elif len(intents) == 1:
            primary_intent = intents[0]
        else:
            primary_intent = QueryIntent.MIXED

        # Set boost factors based on intent
        if primary_intent == QueryIntent.TEMPORAL:
            context['boost_factors']['recency'] = 2.0  # Double recency weight

        elif primary_intent == QueryIntent.PROJECT:
            # Boost memories with project tags
            context['boost_factors']['metadata_match'] = 1.5
            context['filters']['metadata'] = {
                'project': context['entities']['projects']
            }

        elif primary_intent == QueryIntent.PERSON:
            # Boost memories mentioning these persons
            context['boost_factors']['actor_match'] = 1.5

        elif primary_intent == QueryIntent.EMOTIONAL:
            # Boost memories with similar emotional states
            context['boost_factors']['affect_match'] = 1.5

        elif primary_intent == QueryIntent.TECHNICAL:
            # Boost technical content
            context['boost_factors']['tag_match'] = 1.5
            context['filters']['tags'] = ['technical', 'work']

        return primary_intent, context

    def disambiguate_query(self, query: str, intent: QueryIntent, context: Dict) -> str:
        """
        Disambiguate query based on detected intent

        For example, if "Атлас" is detected as a project,
        add context to improve search
        """
        enhanced_query = query

        if intent == QueryIntent.PROJECT and context['entities']['projects']:
            # Add project context
            project = context['entities']['projects'][0]
            if project in ['atlas', 'атлас']:
                # Disambiguate Atlas project from mythology/geography
                enhanced_query = f"{query} проект система"

        elif intent == QueryIntent.TEMPORAL:
            # Temporal queries often need the subject
            if not any(context['entities'][k] for k in ['persons', 'projects']):
                # No known subject in a temporal query — leave it as-is.
                enhanced_query = query

        return enhanced_query