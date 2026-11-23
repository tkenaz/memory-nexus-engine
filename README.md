# Memory Nexus

An MCP server that gives an assistant long-term memory: it stores what happened,
and finds it again when a later conversation needs it. PostgreSQL + pgvector,
local embeddings (BGE-M3), no external API calls at query time.

It grew out of a single-user deployment with mixed Russian and English content,
and its design choices reflect that. This README describes what the code does
today, including where it is weak.

## How retrieval works

Dense vectors alone are not enough for personal memory. Cosine scores from a
good embedding model are compressed (most results land between 0.7 and 0.8),
so a threshold cannot separate "the memory you mean" from "a memory about the
same kind of thing". Names, project labels and rare words are where dense
search is weakest and where people anchor their questions.

So `search_memories_enhanced` runs several retrievers and fuses their ranks:

1. **Semantic.** BGE-M3 dense vectors, cosine similarity over chunks; matched
   chunks resolve to their parent memory.
2. **Keyword.** PostgreSQL full-text search with OR semantics, always on (not a
   fallback). `ts_rank_cd` ranks documents with more matching words higher.
3. **Tag catalog.** The query embedding is compared with embeddings of the tag
   vocabulary (short label against short query: the symmetric case where a
   bi-encoder works well). The best tags expand into their memories. This finds
   memories whose anchor lives only in metadata, not in the text.
4. **Metadata.** Two more ranked lists, newest first: memories whose tags
   contain the query exactly, and memories whose metadata contains the query as
   a substring. They decide the order mainly for short queries such as a project
   name.
5. **Reciprocal Rank Fusion** (`src/search/rank_fusion.py`): each list
   contributes `1 / (60 + rank)`; the retrievers' own scores are ignored because
   their scales are not comparable. Fused scores are normalised to (0, 1].

After fusion: graph expansion from the top three results (links of type
`similar`, `after`, `caused_by`, depth 2). A graph neighbour scores at most 80%
of the best direct result, so it can outrank weaker direct matches. Then a
recency blend for every result (exponential decay over 30 days; 10% of the score,
30% for temporal queries).

Temporal phrases such as "yesterday", "last week", "October 2025", "2025-10-03",
"21 August 2025" (and their Russian equivalents) become date filters on every
retriever. Days are taken in the machine's local time zone. A query that
consists only of a temporal phrase skips retrieval and returns the memories from
that period, newest first.

## What gets stored

Each memory is a row in `memories` with content, metadata (type, tags, date),
an embedding, and a SASH-F frame.

- **SASH-F** (Scene-Actor-State-History-Frame): actors, action, manner,
  affect (valence / arousal / dominance), sensory qualia, time bucket, auto-tags.
  Extracted on store by `src/core/sashf_parser.py`.
- **Chunks.** Memories longer than 1500 characters are split into overlapping
  chunks, each with its own embedding; search returns the parent.
- **Links.** On store, a memory is linked to near-duplicates (`similar`, cosine
  > 0.85, up to 10) and, by a database trigger, to up to five memories from the
  24 hours before it (`after`: the newer memory points to the earlier one,
  weight 0.9 / 0.7 / 0.5 for under 1 h / 6 h / 24 h). Chunks are not linked.
  Graph expansion walks links in both directions, so it surfaces what was said
  just before and just after a matching memory.
- **Patterns.** Recurring actor/action/affect combinations are counted in
  `memory_patterns` and can be listed with `get_patterns`.
- **Tag catalog.** New tags are embedded on store into `tag_embeddings`.

## MCP tools

| Tool | What it does |
|------|--------------|
| `store_memory` | Store a memory: SASH-F extraction, chunking, embedding, auto-links, tag catalog update |
| `search_memories_enhanced` | Semantic, keyword, tag-catalog and metadata retrieval fused with RRF, then graph expansion and recency; optional date filters |
| `get_recent_memories` | Newest memories in order, no search. Long entries are cut with an explicit `[TRUNCATED: showing N of M characters ...]` marker and a pointer to `get_memory` |
| `get_memory` | One memory by id, full text |
| `get_memory_stats` | Counts of memories, embeddings, links, patterns |
| `get_patterns` | Recurring SASH-F patterns |

The truncation marker exists because a silent "..." is indistinguishable from a
complete record: a model reading it fills in the missing tail and then states it
as fact.

## Requirements

- Python 3.11
- PostgreSQL 14+ with pgvector >= 0.5.0 (HNSW index)
- About 4.3 GB of disk for the `BAAI/bge-m3` snapshot, downloaded from Hugging
  Face on first use. The device is chosen by FlagEmbedding; CPU is enough for
  interactive use, re-embedding a large database is faster on a GPU

## Quick start

With Docker for PostgreSQL. For a native installation and troubleshooting, see
[INSTALL.md](INSTALL.md).

```bash
# 1. PostgreSQL with pgvector (or use your own instance)
docker run -d --name memory-nexus-db -p 5433:5432 \
  -e POSTGRES_PASSWORD=change_me -e POSTGRES_DB=memory_nexus \
  pgvector/pgvector:pg16

# 2. Python environment
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configuration
cp .env.example .env   # set PG_PORT=5433 and PG_PASSWORD for the container above

# 4. Schema (idempotent; safe to re-run)
python init_db.py
```

Register the server with your MCP client. Claude Code:

```bash
claude mcp add memory-nexus -- /abs/path/.venv/bin/python /abs/path/run_sashf_server.py
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "memory-nexus": {
      "command": "/abs/path/.venv/bin/python",
      "args": ["/abs/path/run_sashf_server.py"]
    }
  }
}
```

The server reads `.env` from the repository root regardless of the working
directory the client starts it in. The first search loads BGE-M3, which takes
several seconds; later calls reuse it.

## Configuration

All settings are in `config/settings.py` and can be overridden from the
environment or `.env`. The ones you are likely to touch:

| Setting | Default | Meaning |
|---------|---------|---------|
| `PG_HOST`, `PG_PORT`, `PG_DB`, `PG_USER`, `PG_PASSWORD` | `localhost`, `5432`, `memory_nexus`, `postgres`, empty | Connection |
| `PG_POOL_MIN`, `PG_POOL_MAX` | `1`, `5` | Connections per server process (each MCP client starts its own process) |
| `AUTO_MIGRATE` | `true` | Apply `migrations/` on server start (serialised with an advisory lock). Set `false` to run as a role without DDL rights |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Loaded through FlagEmbedding's `BGEM3FlagModel`; other model families load but produce degraded vectors |
| `MIN_SIMILARITY` | `0.15` | Floor for the semantic retriever |
| `RRF_K` | `60` | RRF constant |
| `TAG_TOP_K`, `TAG_MIN_SIMILARITY` | `8`, `0.5` | Tag catalog candidates and threshold |
| `GRAPH_STOP_ACTORS` | empty | Actors too frequent to be useful for graph expansion (for example the user's and the assistant's names) |
| `CHUNK_MAX_SIZE`, `CHUNK_OVERLAP`, `CHUNK_MIN_SIZE` | `1500`, `200`, `100` | Chunking, in characters |

### Running with least privilege

The schema lives only in `migrations/`. `init_db.py` and the server (with
`AUTO_MIGRATE=true`) apply it with the same function, `src/storage/migrations.py`.
To keep DDL rights away from the process the model talks to:

```sql
-- as the owner role: python init_db.py
CREATE ROLE memory_nexus_rw LOGIN PASSWORD '...';
GRANT CONNECT ON DATABASE memory_nexus TO memory_nexus_rw;
GRANT USAGE ON SCHEMA public TO memory_nexus_rw;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO memory_nexus_rw;
```

Then run the server as `memory_nexus_rw` with `AUTO_MIGRATE=false`. The server
never deletes rows, so the runtime role does not need `DELETE`; the maintenance
scripts (`build_tag_index.py`, `reembed_all.py`, `rechunk_existing.py`) run as
the owner.

## Maintenance

- `python init_db.py`: apply or re-apply the schema.
- `python reembed_all.py [--dry-run]`: recompute all embeddings, required after
  changing `EMBEDDING_MODEL`.
- `python build_tag_index.py [--rebuild]`: build the tag catalog in bulk; run with
  `--rebuild` after re-embedding.
- `python scripts/rechunk_existing.py [--dry-run]`: split long memories stored
  before chunking existed.

## Testing

See `tests/README.md`. Unit tests need no database. Integration tests run only
against a throwaway database whose name contains `test`, and refuse to start
otherwise.

## Limitations

Stated plainly, so nobody discovers them in production:

- **Single tenant.** There is no user or tenant column. Every client connected to
  the same database reads and writes the same memories.
- **SASH-F extraction is tuned for Russian.** The lexicons and the NER (Natasha)
  are Russian; for English text most SASH-F fields stay empty. Storage and all
  retrievers work for both languages.
- **Full-text index is English only.** The keyword retriever picks the `english`
  or `russian` text-search configuration per query, but the GIN index exists for
  `english`; Russian keyword queries scan the table. Fine at tens of thousands of
  memories, not beyond.
- **No reranker.** Ranking is RRF plus the heuristics above. An earlier version
  documented a cross-encoder reranking step that the active code path never
  called; it has been removed rather than kept as a promise.
- **No migration bookkeeping.** Every file in `migrations/` runs on every
  `init_db.py` (and on every server start with `AUTO_MIGRATE=true`), so each file
  must stay idempotent.
- **Store-only.** There are no tools to edit or delete a memory; corrections are
  new memories. Deleting is a manual SQL operation.
- **Tag threshold is not tuned.** With correct vectors, unrelated short strings score
  about 0.50-0.53 against tags, so `TAG_MIN_SIMILARITY=0.5` lets unrelated tags vote.
  It should be chosen from real data; a strict xfail test documents the gap.
- **Fixed constants.** Auto-link thresholds (cosine > 0.85, 24-hour window), the
  graph cap (80%) and the recency blend are constants in code, not settings.
- **Retrieval quality is not benchmarked here.** There is no public evaluation
  set yet; measure on your own data before relying on recall.

## Changes since the previous snapshot

- Embeddings: the previous snapshot configured `intfloat/multilingual-e5-large`
  while the code loaded the model through the BGE-M3 wrapper, so vectors were
  not computed the way e5 expects (no `query:`/`passage:` prefixes, CLS pooling
  instead of mean pooling); retrieval quality with them was never measured. The
  model is now BGE-M3 end to end. Existing databases need
  `reembed_all.py` and `build_tag_index.py --rebuild`.
- Apple Silicon: PyTorch's attention kernel on MPS returned wrong BGE-M3 vectors
  for texts of about 8 tokens or fewer encoded on their own (cosine 0.14-0.40 to
  the CPU vector), which covers short queries and every tag. The model is now
  loaded with eager attention on MPS (`src/core/encoder.py`), identical to CPU
  and about 6x faster than it. Databases built on a Mac with an earlier version
  need `build_tag_index.py --rebuild`, and `reembed_all.py` if they hold very
  short memories.
- Search: keyword search is always on, a tag-catalog retriever was added, and all
  retrievers, including the metadata matches, are fused with RRF instead of a
  semantic-first fallback chain with fixed scores appended after it.
- Temporal links: one trigger definition (newer memory points to the earlier
  one, windowed on the memory's own timestamp so imports are linked too) instead
  of a trigger and a Python insert writing opposite directions.
- Schema: `vector(1024)` from the first migration; `005` upgrades older databases
  and refuses to discard existing vectors; HNSW instead of IVFFlat (IVFFlat built
  on an empty table has very low recall). The schema is defined only in
  `migrations/`; `init_db.py` uses the same `PG_*` settings as the server.
- Tools: `get_memory`, and an explicit truncation marker in `get_recent_memories`.
- Removed: the legacy server and search path, scripts that printed credentials or
  assumed one machine, and settings for components that do not exist (Redis,
  Neo4j, importance decay).

## Project layout

```
run_sashf_server.py          entry point (MCP over stdio)
init_db.py                   apply migrations/
build_tag_index.py           bulk tag catalog
reembed_all.py               recompute embeddings
config/settings.py           all settings
migrations/                  the schema, 001-006, idempotent
src/mcp/server_sashf.py      MCP tool definitions and handlers
src/storage/postgres_enhanced.py   store, search, links, patterns
src/storage/postgres.py      connection pool, base store
src/storage/migrations.py    migration runner
src/search/rank_fusion.py    Reciprocal Rank Fusion
src/core/                    SASH-F parser, query intent, chunker
scripts/rechunk_existing.py  chunk old long memories
tests/                       unit and integration tests
```

## License

MIT, see [LICENSE](LICENSE). Copyright 37lab OÜ, mv@37lab.ai.
