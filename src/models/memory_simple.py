"""Simplified models for existing memory database"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class Memory(BaseModel):
    """Memory model matching existing DB structure"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchQuery(BaseModel):
    """Search query parameters"""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    min_similarity: Optional[float] = Field(default=0.01, ge=0.0, le=1.0)  # Low threshold!
    metadata_filters: Optional[Dict[str, Any]] = None
    use_hybrid: bool = True  # Enable hybrid search
    
    
class SearchResult(BaseModel):
    """Search result with scores"""
    memory: Memory
    score: float  # similarity for semantic, rank for keyword
    search_type: str  # "semantic" or "keyword"
    
    class Config:
        json_encoders = Memory.Config.json_encoders


class MemoryStats(BaseModel):
    """Database statistics"""
    total_memories: int
    memories_with_embeddings: int
    oldest_memory: datetime
    newest_memory: datetime
    table_size: str
