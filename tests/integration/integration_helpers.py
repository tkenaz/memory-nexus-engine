"""Helpers shared by the integration tests (fixtures live in conftest.py)."""
import asyncio
import os
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import asyncpg
import pytest

import dbsafety

TESTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_DIR.parent
FIXTURES = TESTS_DIR / "fixtures"

# Migrations exactly as they were before the BGE-M3 port (git HEAD e7e0935):
# memories.embedding vector(384) with an IVFFlat index. Applied in the order the
# old init_db.py used (sorted file names).
LEGACY_MIGRATIONS = [
    FIXTURES / "legacy_001_init_schema.sql",
    FIXTURES / "legacy_002_add_sashf_and_graph.sql",
    FIXTURES / "legacy_003_add_chunking.sql",
    FIXTURES / "legacy_fix_memory_patterns_example_ids.sql",
]

_SETTINGS_FIELDS = ("PG_HOST", "PG_PORT", "PG_DB", "PG_USER", "PG_PASSWORD", "AUTO_MIGRATE")


async def drain_background_tasks(timeout: float = 180.0) -> None:
    """Wait for fire-and-forget tasks (auto-link, patterns, tag catalog upsert).

    store_memory_enhanced() starts them with asyncio.create_task and returns
    immediately; tests must wait for them before reading their effects.
    """
    loop = asyncio.get_running_loop()
    me = asyncio.current_task()
    deadline = loop.time() + timeout
    while True:
        pending = [t for t in asyncio.all_tasks() if t is not me and not t.done()]
        if not pending:
            return
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(f"{len(pending)} background task(s) still running after {timeout}s")
        await asyncio.wait(pending, timeout=remaining)


class ThrowawayServer:
    """Creates and drops scenario databases on the throwaway server."""

    def __init__(self, target: dbsafety.TestDsn, conn: asyncpg.Connection):
        self.target = target
        self.conn = conn
        self.created = []

    async def connect(self, dsn: dbsafety.TestDsn, **kwargs) -> asyncpg.Connection:
        dbsafety.assert_safe(dsn)
        return await asyncpg.connect(dsn.url, **kwargs)

    async def create_database(self, scenario: str) -> dbsafety.TestDsn:
        name = f"{self.target.database}_{scenario}_{secrets.token_hex(4)}".lower()
        dsn = self.target.with_database(name)
        dbsafety.assert_safe(dsn)
        assert dbsafety.REQUIRED_DB_SUBSTRING in name
        try:
            await self.conn.execute(f'CREATE DATABASE "{name}"')
        except asyncpg.InsufficientPrivilegeError as exc:
            pytest.skip(f"the DSN role cannot CREATE DATABASE on the throwaway server: {exc}")
        self.created.append(name)
        return dsn

    async def drop_database(self, dsn: dbsafety.TestDsn) -> None:
        dbsafety.assert_safe(dsn)
        assert dsn.database != self.target.database, "never drop the DSN database itself"
        await self.conn.execute(f'DROP DATABASE IF EXISTS "{dsn.database}" WITH (FORCE)')
        if dsn.database in self.created:
            self.created.remove(dsn.database)


@contextmanager
def settings_pointed_at(dsn: dbsafety.TestDsn, auto_migrate: bool = True):
    """Temporarily point config.settings at a scenario database.

    The storage reads settings only inside initialize(); once the pool exists the
    settings can be restored.
    """
    from config.settings import settings

    saved = {f: getattr(settings, f) for f in _SETTINGS_FIELDS}
    try:
        settings.PG_HOST = dsn.host
        settings.PG_PORT = dsn.port
        settings.PG_DB = dsn.database
        settings.PG_USER = dsn.user
        settings.PG_PASSWORD = dsn.password
        settings.AUTO_MIGRATE = auto_migrate
        dbsafety.assert_safe(dbsafety.parse_dsn(settings.PG_DSN))  # what the code will use
        yield settings
    finally:
        for f, v in saved.items():
            setattr(settings, f, v)


def pg_env(dsn: dbsafety.TestDsn) -> dict:
    """Environment for running a repo script (init_db.py) against dsn."""
    dbsafety.assert_safe(dsn)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PG_")}
    env.update(dsn.pg_env())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


async def open_storage(dsn: dbsafety.TestDsn, encoder=None, auto_migrate: bool = True):
    """EnhancedPostgresStorage connected to dsn (initialize() as the server runs it)."""
    import src.storage.postgres_enhanced as pe

    storage = pe.EnhancedPostgresStorage()
    if encoder is not None:
        storage.encoder = encoder  # share the loaded model; _ensure_encoder skips loading
    with settings_pointed_at(dsn, auto_migrate=auto_migrate):
        await storage.initialize()
    return storage


async def store_like_server(storage, content: str, metadata: Optional[dict] = None) -> str:
    """What the MCP tool `store_memory` does (src/mcp/server_sashf.py), then wait
    for the background tasks it starts."""
    from src.core import chunker

    embedding = None
    if not chunker.needs_chunking(content):
        embedding = await storage.encode_text(content)
    memory_id = await storage.store_memory_enhanced(
        content=content, embedding=embedding, metadata=metadata or {}
    )
    await drain_background_tasks()
    return memory_id


async def column_type(conn, table: str, column: str) -> str:
    return await conn.fetchval(
        """SELECT format_type(atttypid, atttypmod) FROM pg_attribute
           WHERE attrelid = $1::regclass AND attname = $2 AND NOT attisdropped""",
        table, column,
    )


async def index_def(conn, name: str) -> Optional[str]:
    return await conn.fetchval("SELECT indexdef FROM pg_indexes WHERE indexname = $1", name)
