# Memory Nexus -- Installation Guide

## Requirements

- Python 3.11+
- PostgreSQL 14+ with pgvector extension
- ~500MB disk space (embedding model download)

## 1. Install dependencies

```bash
cd memory-nexus-v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Set up PostgreSQL

### macOS (Homebrew)
```bash
brew install postgresql@14
brew services start postgresql@14
```

### Ubuntu/Debian
```bash
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### Create database

```bash
createdb memory_nexus
psql memory_nexus -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql memory_nexus -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
```

## 3. Configure

```bash
cp .env.example .env
```

Edit `.env` with your database credentials. Key fields:

```bash
DATABASE_URL=postgresql://postgres:password@localhost:5432/memory_nexus
```

Or set individual `PG_*` variables in `.env` (see `config/settings.py` for all options).

## 4. Initialize database

```bash
python init_db.py
```

This creates all tables, indexes, triggers, and applies migrations (001-003).

## 5. Add to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "memory-nexus": {
      "command": "python3",
      "args": ["/absolute/path/to/memory-nexus-v2/run_sashf_server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/memory-nexus-v2"
      }
    }
  }
}
```

Use absolute paths. No `~` or relative paths.

Restart Claude Desktop.

## 6. Verify

After restart, the following MCP tools should appear:
- `store_memory`
- `search_memories_enhanced`
- `get_recent_memories`
- `get_memory_stats`
- `get_patterns`

## Troubleshooting

### "Module not found" errors
Ensure `PYTHONPATH` in Claude Desktop config points to the repository root.

### PostgreSQL connection failed
Check `.env` has correct credentials and PostgreSQL is running:
```bash
psql -U postgres -d memory_nexus -c "SELECT 1;"
```

### Embedding model download slow
First run downloads ~1.2GB model. Pre-download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"
```

### Claude Desktop doesn't show the server
1. Check logs: `~/Library/Logs/Claude/mcp*.log`
2. Verify JSON syntax in config
3. Ensure absolute paths
4. Try running manually: `python run_sashf_server.py`
