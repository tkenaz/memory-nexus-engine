# Memory Nexus

MCP server for persistent conversational memory with hybrid search, SASH-F structure extraction, and graph relationships. Built for Claude Desktop / Claude Code.

## What it does

- **Stores memories** with automatic SASH-F parsing (actors, actions, affect, manner, qualia, temporal tags)
- **Hybrid search**: semantic (pgvector cosine similarity) + keyword fallback (PostgreSQL FTS)
- **Graph expansion**: finds related memories through shared actors and links
- **Chunking**: long memories are split into focused chunks, each with its own embedding
- **Pattern detection**: discovers recurring patterns across memories
- **Temporal search**: understands "yesterday", "last week", date ranges

## Stack

- **Python 3.11+**, async (asyncpg)
- **PostgreSQL 14+** with pgvector extension
- **Embedding model**: `intfloat/multilingual-e5-large` (1024 dims)
- **Reranking**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **NLP**: Natasha (Russian NER), pymorphy2
- **MCP**: `mcp` Python SDK (stdio transport)

## MCP Tools

| Tool | Description |
|------|-------------|
| `store_memory` | Store memory with auto SASH-F extraction and chunking |
| `search_memories_enhanced` | Hybrid semantic + keyword search with graph expansion |
| `get_recent_memories` | Chronological retrieval (no semantic search) |
| `get_memory_stats` | Database stats: memory count, links, patterns |
| `get_patterns` | Recurring patterns across memories |

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

Key settings in `config/settings.py`:
- `PG_*` -- PostgreSQL connection
- `EMBEDDING_MODEL` -- sentence-transformers model
- `MIN_SIMILARITY` -- search threshold (default: 0.15)
- `CHUNK_MAX_SIZE` -- chunking threshold (default: 1500 chars)

### 3. Initialize database

```bash
python init_db.py
```

Applies migrations from `migrations/` (schema, SASH-F, chunking).

### 4. Add to Claude Desktop

In `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory-nexus": {
      "command": "python3",
      "args": ["/path/to/memory-nexus-v2/run_sashf_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/memory-nexus-v2"
      }
    }
  }
}
```

For Claude Code, add to `.claude/settings.json`:

```json
{
  "mcpServers": {
    "memory-nexus": {
      "command": "python3",
      "args": ["/path/to/memory-nexus-v2/run_sashf_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/memory-nexus-v2"
      }
    }
  }
}
```

## Project structure

```
memory-nexus-v2/
├── run_sashf_server.py          # Entry point
├── init_db.py                   # Database initialization
├── config/
│   └── settings.py              # All configuration (Pydantic)
├── src/
│   ├── mcp/
│   │   ├── server_sashf.py      # MCP server (active)
│   │   └── server_v2.py         # Legacy MCP server
│   ├── core/
│   │   ├── sashf_parser.py      # SASH-F structure extraction
│   │   ├── query_intent.py      # Query intent analysis
│   │   └── chunker.py           # Memory chunking
│   ├── models/
│   │   ├── memory.py            # Memory model
│   │   ├── memory_simple.py     # Simplified memory model
│   │   └── sashf.py             # SASH-F data model
│   ├── storage/
│   │   ├── postgres_enhanced.py # Main storage (search, store, graph)
│   │   ├── postgres.py          # Base PostgreSQL adapter
│   │   └── postgres_adapter.py  # Connection adapter
│   ├── search/
│   │   └── hybrid_search.py     # Hybrid search engine
│   ├── assembler/
│   │   └── context_assembler.py # Context budget packing
│   └── utils/
│       └── temporal_parser.py   # Date/time parsing
├── migrations/
│   ├── 001_init_schema.sql
│   ├── 002_add_sashf_and_graph.sql
│   └── 003_add_chunking.sql
├── scripts/
│   └── rechunk_existing.py      # Re-chunk existing long memories
└── test_*.py                    # Tests
```

## Utilities

- `reembed_all.py` -- re-generate embeddings for all memories (after model change)
- `scripts/rechunk_existing.py` -- chunk existing long memories
- `run_migration.sh` -- apply SQL migrations
- `update_mcp.sh` -- update MCP SDK version

## Search pipeline

1. **Query intent analysis** -- detect temporal markers, extract search terms
2. **Semantic search** -- pgvector cosine similarity (top 50)
3. **Keyword fallback** -- PostgreSQL FTS if semantic scores < threshold
4. **Graph expansion** -- traverse actor/link relationships (depth 2)
5. **Cross-encoder reranking** -- rerank merged results
6. **Recency boost** -- slight preference for recent memories
7. **Chunking resolution** -- if a chunk matched, return parent memory
