"""Data models for Memory Nexus"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field, validator


class MemoryType(str, Enum):
    """Types of memories"""
    FACT = "fact"
    SUMMARY = "summary"
    THOUGHT = "thought"
    CODE = "code"
    EMOTION = "emotion"
    CLAUDE_EMOTION_STATE = "claude_emotion_state"  # The assistant's own emotional states
    CLAUDE_REACTION_PATTERN = "claude_reaction_pattern"  # The assistant's own reaction patterns


class Memory(BaseModel):
    """Core memory model"""
    id: UUID = Field(default_factory=uuid4)
    content: str
    embedding: Optional[List[float]] = None
    type: MemoryType = MemoryType.FACT
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("importance")
    def validate_importance(cls, v):
        """Ensure importance is in valid range"""
        return max(0.0, min(1.0, v))
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat()
        }


class MemoryRelationship(BaseModel):
    """Relationship between memories"""
    from_id: UUID
    to_id: UUID
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchQuery(BaseModel):
    """Search query parameters"""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    memory_types: Optional[List[MemoryType]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    
    
class SearchResult(BaseModel):
    """Search result with scores"""
    memory: Memory
    similarity_score: float
    keyword_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: float
    context: Optional[Dict[str, Any]] = None  # Graph context
    
    class Config:
        json_encoders = Memory.Config.json_encoders


class ImportanceFactors(BaseModel):
    """Factors for calculating importance"""
    novelty: float = Field(ge=0.0, le=1.0)
    emotional_intensity: float = Field(ge=0.0, le=1.0)
    relevance_to_current: float = Field(ge=0.0, le=1.0)
    user_marked: bool = False
    
    def calculate_importance(self, novelty_weight: float, emotion_weight: float) -> float:
        """Calculate final importance score"""
        if self.user_marked:
            return 1.0
            
        base_score = (
            novelty_weight * self.novelty +
            emotion_weight * self.emotional_intensity
        )
        
        # Boost by current relevance
        return min(1.0, base_score * (1 + self.relevance_to_current * 0.5))
