# Installing Memory Nexus without Docker

The README's quick start uses the `pgvector/pgvector` image. This page covers a
native PostgreSQL and the problems people actually hit.

## PostgreSQL with pgvector

macOS (Homebrew):

```bash
brew install postgresql@16 pgvector
brew services start postgresql@16
createdb memory_nexus
```

Debian/Ubuntu (PGDG packages):

```bash
sudo apt install postgresql-16 postgresql-16-pgvector
sudo -u postgres createdb memory_nexus
```

You do not need to create extensions by hand: migration 001 runs
`CREATE EXTENSION IF NOT EXISTS vector` and `"uuid-ossp"`, which requires a role
allowed to create extensions (the database owner on PostgreSQL 13+ for trusted
extensions, otherwise a superuser) for the first `init_db.py`.

Check the pgvector version (HNSW needs 0.5.0 or newer):

```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';
```

## Python

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # requirements-dev.txt for tests
cp .env.example .env                     # PG_* settings
python init_db.py
```

## Troubleshooting

**`init_db.py` cannot connect.** It uses the same `PG_*` settings as the server,
from the environment or `.env` in the repository root. It prints the target as
`user@host:port/db` without the password.

**Migration 005 stops with "cannot be cast to vector(1024)".** The database was
created by an older version with 384-dimensional vectors. The message contains
the exact steps: clear the old vectors, re-run `init_db.py`, then
`reembed_all.py` and `build_tag_index.py --rebuild`.

**The first search is slow, or fails without network.** BGE-M3 is downloaded from
Hugging Face on first use (about 4.3 GB). Pre-download it once:

```bash
python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3')"
```

For offline machines, copy the Hugging Face cache and set `HF_HUB_OFFLINE=1`.

**The MCP client does not show the server.** Run it by hand to see the error:
`.venv/bin/python run_sashf_server.py` (it waits for JSON-RPC on stdin; Ctrl-C to
stop). Claude Desktop logs are in `~/Library/Logs/Claude/mcp*.log` on macOS. Use
absolute paths in the client configuration.

**Anything printed to stdout breaks the MCP protocol.** If you add code, log to
stderr. The storage layer routes `structlog` to stderr for this reason.
