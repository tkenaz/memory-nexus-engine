"""SASH-F (Scene-Actor-State-History-Frame) models for structured memory"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field, validator


class TimeOfDay(str, Enum):
    """Time of day buckets"""
    NIGHT = "night"      # 22:00-04:59
    MORNING = "morning"  # 05:00-11:59
    DAY = "day"          # 12:00-16:59
    EVENING = "evening"  # 17:00-21:59


class Affect(BaseModel):
    """Emotional state in VAD model"""
    valence: float = Field(0.0, ge=-1.0, le=1.0)  # Negative to positive
    arousal: float = Field(0.0, ge=-1.0, le=1.0)  # Calm to excited
    dominance: float = Field(0.0, ge=-1.0, le=1.0)  # Submissive to dominant
    
    @property
    def bucket(self) -> str:
        """Get emotional bucket for indexing"""
        v = round(self.valence, 1)
        a = round(self.arousal, 1)
        d = round(self.dominance, 1)
        return f"V{v:+.1f}A{a:+.1f}D{d:+.1f}"
    
    @property
    def intensity(self) -> float:
        """Overall emotional intensity"""
        return (abs(self.valence) + abs(self.arousal) + abs(self.dominance)) / 3


class LinkType(str, Enum):
    """Types of relationships between memories"""
    # Temporal
    AFTER = "after"
    BEFORE = "before"
    DURING = "during"
    OVERLAPS = "overlaps"
    
    # Semantic
    SIMILAR = "similar"
    CONTRADICTS = "contradicts"
    ELABORATES = "elaborates"
    SUMMARIZES = "summarizes"
    
    # Causal
    CAUSED_BY = "caused_by"
    LEADS_TO = "leads_to"
    ENABLES = "enables"
    PREVENTS = "prevents"
    
    # Reference
    MENTIONS = "mentions"
    REPLIES_TO = "replies_to"
    CONTINUES = "continues"


class SceneFrame(BaseModel):
    """Enhanced memory with SASH-F structure"""
    # Core fields
    id: UUID = Field(default_factory=uuid4)
    content: str  # Original text
    
    # SASH-F Structure
    actors: List[str] = Field(default_factory=list)  # Entities involved
    act: Optional[str] = None  # Main action/verb
    manner: Optional[str] = None  # How it was done
    affect: Optional[Affect] = None  # Emotional coloring
    qualia: List[str] = Field(default_factory=list)  # Sensory details
    
    # Temporal
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    time_bucket: Optional[TimeOfDay] = None
    duration_seconds: Optional[int] = None
    
    # Meta
    source: str = Field(default="manual")  # manual, inferred, system
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    
    # Storage
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Auto-set time bucket if not provided
        if self.time_bucket is None:
            hour = self.timestamp.hour
            if 5 <= hour < 12:
                self.time_bucket = TimeOfDay.MORNING
            elif 12 <= hour < 17:
                self.time_bucket = TimeOfDay.DAY
            elif 17 <= hour < 22:
                self.time_bucket = TimeOfDay.EVENING
            else:
                self.time_bucket = TimeOfDay.NIGHT
    
    @property
    def importance(self) -> float:
        """Calculate importance based on multiple factors"""
        base = self.confidence
        
        # Emotional memories are more important
        if self.affect:
            base += self.affect.intensity * 0.2
        
        # Memories with multiple actors are more important
        if len(self.actors) > 1:
            base += 0.1
        
        # Tagged memories are explicitly important
        if self.tags:
            base += 0.1
            
        return min(1.0, base)
    
    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            Enum: lambda v: v.value
        }


class MemoryLink(BaseModel):
    """Enhanced link between memories with metadata"""
    from_id: UUID
    to_id: UUID
    link_type: LinkType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = SceneFrame.Config.json_encoders


class Pattern(BaseModel):
    """Recurring pattern detected in memories"""
    id: UUID = Field(default_factory=uuid4)
    pattern_key: str  # Canonical representation
    occurrences: int = 1
    confidence: float = Field(default=0.5)
    
    # Pattern components
    actors: List[str] = Field(default_factory=list)
    act: Optional[str] = None
    manner: Optional[str] = None
    affect_bucket: Optional[str] = None
    
    # Examples and timeline
    example_ids: List[UUID] = Field(default_factory=list, max_items=10)
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    
    # Meta
    promoted: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_occurrence(self, frame_id: UUID):
        """Update pattern with new occurrence"""
        self.occurrences += 1
        self.last_seen = datetime.utcnow()
        
        # Keep only recent examples
        if frame_id not in self.example_ids:
            self.example_ids = ([frame_id] + self.example_ids)[:10]
        
        # Boost confidence with repetition
        self.confidence = min(1.0, self.confidence + 0.05)
    
    class Config:
        json_encoders = SceneFrame.Config.json_encoders
