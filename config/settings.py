"""Configuration for Memory Nexus.

Values come from environment variables or from a .env file in the repository
root (see .env.example). Every setting below is read by the code; there are no
placeholders for features that do not exist.
"""
from pathlib import Path

from dotenv import load_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings

# Resolve .env relative to the repo root, not the cwd: MCP hosts (Claude Desktop,
# Claude Code) spawn servers from an arbitrary working directory.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseSettings):
    """Application settings."""

    # PostgreSQL
    PG_HOST: str = "localhost"
    PG_PORT: int = 5432
    PG_DB: str = "memory_nexus"
    PG_USER: str = "postgres"
    PG_PASSWORD: str = ""  # set in .env; never hardcode a secret in source

    # Connection pool per server process. An MCP stdio server serves one client,
    # so a few connections are enough; several clients means several processes.
    PG_POOL_MIN: int = 1
    PG_POOL_MAX: int = 5

    # When false, the server connects as a least-privilege runtime role and skips
    # startup DDL; an owner role applies migrations separately (python init_db.py).
    AUTO_MIGRATE: bool = True

    @computed_field
    @property
    def PG_DSN(self) -> str:
        return f"postgresql://{self.PG_USER}:{self.PG_PASSWORD}@{self.PG_HOST}:{self.PG_PORT}/{self.PG_DB}"

    # Embeddings
    # BGE-M3, loaded through FlagEmbedding (BGEM3FlagModel). Earlier snapshots set
    # e5-large here while the code loaded it through the BGE-M3 wrapper (no e5
    # query:/passage: prefixes, wrong pooling), so vectors were silently degraded.
    # Changing the model means re-embedding: reembed_all.py, then
    # build_tag_index.py --rebuild.
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024  # must match the vector(N) columns in migrations/

    # Search
    MIN_SIMILARITY: float = 0.15  # BGE-M3 cosine scores typically fall in 0.2-0.9
    RRF_K: int = 60  # Reciprocal Rank Fusion: score = sum(1 / (RRF_K + rank))

    # Tag catalog: third retrieval engine fused by RRF (see migrations/004)
    TAG_TOP_K: int = 8               # candidate tags taken from the catalog
    TAG_MIN_SIMILARITY: float = 0.5  # query-to-tag cosine threshold (BGE-M3)

    # Graph expansion
    # High-frequency entities to exclude from graph hub-expansion: they over-link
    # and act as false super-hubs. Customize for your own world (people, projects,
    # the assistant's name). Empty = no exclusion.
    GRAPH_STOP_ACTORS: list = []

    # Chunking
    CHUNK_MAX_SIZE: int = 1500    # characters, ~400 tokens of mixed-language text
    CHUNK_OVERLAP: int = 200      # overlap between neighbouring chunks
    CHUNK_MIN_SIZE: int = 100     # do not create tiny fragments

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
