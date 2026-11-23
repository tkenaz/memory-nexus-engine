"""SASH-F Parser - extracts structured information from text"""
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Doc
)

from ..models.sashf import Affect, TimeOfDay


class SASHFParser:
    """Parser for extracting SASH-F structure from text"""
    
    def __init__(self):
        # Natasha NLP pipeline
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        
        # Emotion lexicons
        self._init_emotion_lexicons()
        
        # Action patterns
        self._init_action_patterns()
        
        # Manner lexicon
        self._init_manner_lexicon()
        
        # Sensory (qualia) patterns
        self._init_qualia_patterns()
    
    def _init_emotion_lexicons(self):
        """Initialize emotion word lists for VAD detection"""
        # Valence indicators
        self.positive_words = {
            'радость', 'счастье', 'удовольствие', 'восторг', 'нежность',
            'любовь', 'восхищение', 'интерес', 'удовлетворение', 'гордость',
            'спокойствие', 'уверенность', 'благодарность', 'облегчение'
        }
        
        self.negative_words = {
            'грусть', 'печаль', 'тоска', 'страх', 'тревога', 'гнев',
            'злость', 'раздражение', 'отвращение', 'стыд', 'вина',
            'разочарование', 'одиночество', 'усталость', 'боль'
        }
        
        # Arousal indicators
        self.high_arousal = {
            'взволнован', 'возбужден', 'энергичн', 'активн', 'бурн',
            'интенсивн', 'яростн', 'паник', 'восторг', 'экстаз'
        }
        
        self.low_arousal = {
            'спокоен', 'расслаблен', 'сонн', 'устал', 'вял',
            'медленн', 'тих', 'безразличн', 'апатичн'
        }
        
        # Dominance indicators
        self.high_dominance = {
            'контролир', 'управля', 'доминиру', 'командова', 'власт',
            'сил', 'уверен', 'решительн', 'настойчив'
        }
        
        self.low_dominance = {
            'подчиня', 'слаб', 'беспомощн', 'зависим', 'покорн',
            'пассивн', 'уязвим', 'неуверен', 'сомнева'
        }
    
    def _init_action_patterns(self):
        """Initialize action/verb patterns"""
        self.action_patterns = {
            'создание': ['созда', 'дела', 'строи', 'пиш', 'рисова'],
            'анализ': ['анализ', 'изуча', 'исследова', 'разбира', 'понима'],
            'общение': ['говор', 'обща', 'обсужда', 'расскaz', 'дели'],
            'движение': ['ид', 'еха', 'беж', 'лет', 'плыв'],
            'чувство': ['чувств', 'ощуща', 'испытыва', 'переживa'],
            'мышление': ['дума', 'размышля', 'соображa', 'понима'],
        }
    
    def _init_manner_lexicon(self):
        """Initialize manner/adverb patterns"""
        self.manner_words = {
            'быстро', 'медленно', 'аккуратно', 'небрежно', 'тщательно',
            'внимательно', 'рассеянно', 'нежно', 'грубо', 'осторожно',
            'смело', 'робко', 'уверенно', 'неуверенно', 'спокойно',
            'взволнованно', 'радостно', 'грустно', 'яростно', 'лениво',
            'устало', 'терпеливо', 'игриво'  # Additional manner words
        }
    
    def _init_qualia_patterns(self):
        """Initialize sensory detail patterns"""
        self.qualia_patterns = {
            'visual': ['вид', 'смотр', 'ярк', 'темн', 'свет', 'цвет', 'красив'],
            'auditory': ['слыш', 'звук', 'голос', 'тих', 'громк', 'шум', 'вздох'],
            'tactile': ['трога', 'ощуща', 'мягк', 'тверд', 'тепл', 'холодн'],
            'olfactory': ['запах', 'арома', 'пахн', 'вонь'],
            'gustatory': ['вкус', 'сладк', 'горьк', 'кисл', 'солен'],
        }
    
    def parse(self, text: str) -> Dict:
        """Extract SASH-F structure from text"""
        # Create Natasha document
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)
        
        # Lemmatization
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        
        # Extract components
        result = {
            'actors': self._extract_actors(doc),
            'act': self._extract_action(doc),
            'manner': self._extract_manner(doc),
            'affect': self._extract_affect(text.lower()),
            'qualia': self._extract_qualia(text.lower()),
            'tags': self._auto_tag(text.lower())
        }
        
        return result
    
    def _extract_actors(self, doc: Doc) -> List[str]:
        """Extract named entities (people, organizations)"""
        actors = []
        
        if hasattr(doc, 'spans'):
            for span in doc.spans:
                if span.type in ['PER', 'ORG']:
                    # Normalize to lowercase
                    actor = span.text.lower()
                    
                    # Apply aliases (nickname -> canonical name, etc)
                    actor = self._apply_actor_aliases(actor)
                    
                    if actor and actor not in actors:
                        actors.append(actor)
        
        # Map first/second-person pronouns to generic actor labels.
        # Rename 'assistant'/'user' to whatever fits your setup.
        for token in doc.tokens:
            if token.pos == 'PRON':
                if token.text.lower() in ['я', 'мне', 'меня']:
                    if 'assistant' not in actors:
                        actors.append('assistant')
                elif token.text.lower() in ['ты', 'тебе', 'тебя']:
                    if 'user' not in actors:
                        actors.append('user')
        
        return actors
    
    def _apply_actor_aliases(self, actor: str) -> str:
        """Apply known aliases for actors.

        Map nicknames/variants to a canonical name. Customize for your own people,
        e.g. {'bob': 'robert', 'боря': 'борис'}.
        """
        aliases: dict = {}
        return aliases.get(actor, actor)
    
    def _extract_action(self, doc: Doc) -> Optional[str]:
        """Extract main verb/action"""
        # Simple approach - find first verb
        for token in doc.tokens:
            if token.pos == 'VERB':
                # Check if lemma is available and not None
                if hasattr(token, 'lemma') and token.lemma:
                    return token.lemma.lower()
                else:
                    # Fallback to simple text processing
                    text = token.text.lower()
                    # Remove common endings
                    for ending in ['ла', 'ли', 'ет', 'ют', 'ит', 'ят', 'ал', 'ала', 'али', 'ула']:
                        if text.endswith(ending) and len(text) > len(ending) + 2:
                            return text[:-len(ending)]
                    return text
        
        return None
    
    def _extract_manner(self, doc: Doc) -> Optional[str]:
        """Extract manner adverbs"""
        # Check text for manner words
        text_lower = doc.text.lower()
        
        # Deterministic: the manner word that occurs first in the text (iterating
        # the set directly would depend on PYTHONHASHSEED).
        found = [(text_lower.find(m), m) for m in self.manner_words if m in text_lower]
        if found:
            return min(found)[1]
        
        # Also check tokens
        for token in doc.tokens:
            if token.pos == 'ADV':
                text = token.text.lower()
                if text in self.manner_words:
                    return text
                # Check if it's a manner adverb by suffix
                if text.endswith('о') or text.endswith('е'):
                    # Add to our known manner words for future
                    return text
        
        return None
    
    def _extract_affect(self, text: str) -> Optional[Affect]:
        """Extract emotional state using VAD model"""
        # Count emotion indicators
        valence_score = 0
        arousal_score = 0
        dominance_score = 0
        
        # Calculate valence
        for word in self.positive_words:
            if word in text:
                valence_score += 0.3
        
        for word in self.negative_words:
            if word in text:
                valence_score -= 0.3
        
        # Calculate arousal
        for word in self.high_arousal:
            if word[:5] in text:  # Prefix matching for Russian stems
                arousal_score += 0.3
        
        for word in self.low_arousal:
            if word[:5] in text:
                arousal_score -= 0.3
        
        # Calculate dominance
        for word in self.high_dominance:
            if word[:5] in text:
                dominance_score += 0.3
        
        for word in self.low_dominance:
            if word[:5] in text:
                dominance_score -= 0.3
        
        # Special case: "устало" indicates low arousal and slight negative valence
        if 'устал' in text:
            arousal_score -= 0.4
            valence_score -= 0.2
        
        # "вздохнула" can indicate fatigue
        if 'вздох' in text:
            arousal_score -= 0.2
            
        
        # Clamp values
        valence = max(-1.0, min(1.0, valence_score))
        arousal = max(-1.0, min(1.0, arousal_score))
        dominance = max(-1.0, min(1.0, dominance_score))
        
        # Only return if there's some emotional content
        if abs(valence) > 0.1 or abs(arousal) > 0.1 or abs(dominance) > 0.1:
            return Affect(valence=valence, arousal=arousal, dominance=dominance)
        
        return None
    
    def _extract_qualia(self, text: str) -> List[str]:
        """Extract sensory details"""
        qualia = []
        
        for sense, patterns in self.qualia_patterns.items():
            for pattern in patterns:
                if pattern in text:
                    if sense not in qualia:
                        qualia.append(sense)
                    break  # One match per sense is enough
        
        return qualia
    
    def _auto_tag(self, text: str) -> List[str]:
        """Auto-generate tags based on content"""
        tags = []
        
        # Technical content
        if any(word in text for word in ['код', 'функци', 'класс', 'метод', 'api', 'архитектур', 'систем']):
            tags.append('technical')
        
        # Emotional content
        if any(word in text for word in ['устал', 'вздох', 'радост', 'груст', 'любл']):
            tags.append('emotional')
        
        # Planning/future
        if any(word in text for word in ['план', 'буду', 'собира', 'хоч', 'наде']):
            tags.append('planning')
        
        # Reflection/past
        if any(word in text for word in ['помн', 'был', 'вспомина', 'раньше', 'вчера', 'после']):
            tags.append('reflection')
        
        # Work/business
        if any(word in text for word in ['работ', 'проект', 'бизнес', 'задач']):
            tags.append('work')
        
        # Personal/intimate
        if any(word in text for word in ['любл', 'нежн', 'ласк', 'близк']):
            tags.append('intimate')
        
        # Time references
        if any(word in text for word in ['час', 'минут', 'день', 'неделя']):
            tags.append('temporal')
        
        return tags


# Simplified functions for direct use
def extract_sashf(text: str) -> Dict:
    """Quick extraction of SASH-F structure"""
    parser = SASHFParser()
    return parser.parse(text)


def extract_actors(text: str) -> List[str]:
    """Quick extraction of actors only"""
    parser = SASHFParser()
    doc = Doc(text)
    doc.segment(parser.segmenter)
    doc.tag_ner(parser.ner_tagger)
    return parser._extract_actors(doc)


def extract_affect(text: str) -> Optional[Affect]:
    """Quick extraction of emotional state"""
    parser = SASHFParser()
    return parser._extract_affect(text.lower())
