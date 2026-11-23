"""The single place where the database schema is applied.

The schema lives only in migrations/*.sql. Files run in name order, every time:
there is no table of applied migrations, so each file must be safe to repeat
(CREATE ... IF NOT EXISTS, guarded DO blocks). Both init_db.py and the server
(with AUTO_MIGRATE=true) call apply_migrations().
"""
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import asyncpg

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

# Session-level advisory lock key: several server processes often start at once
# (one per MCP client window); without it their DDL can deadlock or fail with
# "tuple concurrently updated". Any constant works if nothing else uses it.
MIGRATION_LOCK_KEY = 0x4D4E_0001


def migration_files(directory: Path = MIGRATIONS_DIR) -> List[Path]:
    """Return migrations/*.sql sorted by file name."""
    return sorted(directory.glob("*.sql"))


async def apply_migrations(
    conn: asyncpg.Connection,
    files: Optional[Sequence[Path]] = None,
    on_applied: Optional[Callable[[Path], None]] = None,
) -> int:
    """Execute migration files in order on an open connection.

    Each file runs as one simple-protocol call, so multi-statement SQL works.
    The first failing file raises; files after it do not run. A session-level
    advisory lock serialises concurrent callers (e.g. two servers starting).

    Args:
        conn: Connection with rights to run DDL (owner role).
        files: Files to apply; defaults to migration_files().
        on_applied: Optional callback after each successful file.

    Returns:
        Number of files applied.
    """
    files = migration_files() if files is None else list(files)
    await conn.execute("SELECT pg_advisory_lock($1)", MIGRATION_LOCK_KEY)
    try:
        for path in files:
            await conn.execute(path.read_text(encoding="utf-8"))
            if on_applied is not None:
                on_applied(path)
    finally:
        await conn.execute("SELECT pg_advisory_unlock($1)", MIGRATION_LOCK_KEY)
    return len(files)
