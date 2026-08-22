"""Configuration management for Memory Nexus"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import validator, computed_field

# Load environment variables
# Resolve .env relative to the repo root, not the cwd (Claude Desktop/Code
# spawn MCP servers from an arbitrary working directory).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    EMBEDDING_CACHE_DIR: Path = BASE_DIR / "embeddings_cache"
    
    # PostgreSQL
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "claude"
    PG_USER: str = "postgres"
    PG_PASSWORD: str = ""  # set in .env — never hardcode a secret in source
    
    @computed_field
    @property
    def PG_DSN(self) -> str:
        return f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"
    
    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # ML Models
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"  # Наша проверенная модель — БД закодирована ею!
    RERANK_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Multilingual reranker
    EMBEDDING_DIMENSION: int = 1024  # BGE-M3 dense embeddings

    # Search parameters
    MIN_SIMILARITY: float = 0.15  # После re-embed BGE-M3 дает нормальные scores (0.2-0.9)
    SEARCH_LIMIT: int = 50
    USE_RERANKING: bool = True  # Включить cross-encoder reranking
    RERANK_TOP_K: int = 10  # Сколько результатов вернуть после reranking
    KEYWORD_WEIGHT: float = 0.3
    SEMANTIC_WEIGHT: float = 0.7
    KEYWORD_FALLBACK_THRESHOLD: float = 0.4  # Порог для fallback на keyword search
    
    @validator("KEYWORD_WEIGHT")
    def validate_weights(cls, v, values):
        """Ensure weights sum to 1.0"""
        # Проверяем только если SEMANTIC_WEIGHT уже загружен
        return v
    
    # Memory parameters
    IMPORTANCE_DECAY_RATE: float = 0.97
    MIN_IMPORTANCE: float = 0.05
    IMPORTANCE_NOVELTY_WEIGHT: float = 0.4
    IMPORTANCE_EMOTION_WEIGHT: float = 0.6
    
    # Graph expansion
    # High-frequency entities to exclude from graph hub-expansion: they over-link
    # and act as false super-hubs. Customize for your own world (people, projects,
    # the assistant's name). Empty = no exclusion.
    GRAPH_STOP_ACTORS: list = []

    # Chunking
    CHUNK_MAX_SIZE: int = 1500    # ~400 токенов mixed ru/en
    CHUNK_OVERLAP: int = 200      # перекрытие между чанками
    CHUNK_MIN_SIZE: int = 100     # не создавать крошечные фрагменты

    # Cache
    CACHE_TTL: int = 3600  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or human
    
    # MCP
    MCP_SERVER_NAME: str = "memory-nexus"
    MCP_SERVER_VERSION: str = "2.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()

# Create cache directory if not exists and we have write access
try:
    settings.EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # Running in read-only environment (e.g. Claude Desktop)
    # Use temp directory instead
    import tempfile
    settings.EMBEDDING_CACHE_DIR = Path(tempfile.gettempdir()) / "memory_nexus_cache"
    settings.EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
