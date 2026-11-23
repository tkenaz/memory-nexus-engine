#!/usr/bin/env python3
"""Apply the Memory Nexus schema (migrations/*.sql) to the configured database.

Usage:
    python init_db.py

Connection settings come from config/settings.py: PG_HOST, PG_PORT, PG_DB,
PG_USER, PG_PASSWORD (environment or .env in the repository root). Use a role
that owns the schema. Safe to run repeatedly: every migration is idempotent.
"""
import asyncio
import sys

import asyncpg

from config.settings import settings
from src.storage.migrations import apply_migrations, migration_files


def _target() -> str:
    """Connection target without credentials, for messages."""
    return f"{settings.PG_USER}@{settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DB}"


async def main() -> int:
    files = migration_files()
    if not files:
        print("No migration files found in migrations/", file=sys.stderr)
        return 1

    print(f"Connecting to {_target()}")
    try:
        conn = await asyncpg.connect(settings.PG_DSN)
    except Exception as exc:  # connection errors vary by driver version
        print(f"Cannot connect to {_target()}: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        applied = await apply_migrations(
            conn, files, on_applied=lambda path: print(f"  applied {path.name}")
        )
        stats = await conn.fetchrow("SELECT * FROM memory_stats_basic")
    except asyncpg.PostgresError as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await conn.close()

    print(f"Schema is up to date ({applied} migration files).")
    if stats:
        print(f"Memories: {stats['total_memories']}, "
              f"with embeddings: {stats['memories_with_embeddings']}, "
              f"table size: {stats['table_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
