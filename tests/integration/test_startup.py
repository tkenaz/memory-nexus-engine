"""Server startup: PostgresStorage.initialize() with and without AUTO_MIGRATE.

No model is needed: initialize() does not load BGE-M3.
"""
import asyncio

import numpy as np
import pytest

from config.settings import settings
from integration_helpers import column_type, open_storage
from src.storage.migrations import apply_migrations


async def public_tables(conn):
    return {r[0] for r in await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}


async def close(*storages):
    for s in storages:
        await s.close()


def test_initialize_on_an_empty_database_applies_the_schema_and_registers_vector(
    fresh_db, db_conn, run
):
    # The pool registers the pgvector codec on every connection; on an empty
    # database that only works because migrations run first, on a plain connection.
    storage = run(open_storage(fresh_db, auto_migrate=True))
    try:
        async def roundtrip():
            async with storage.pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO memories (id, content, embedding) VALUES ('v', 'x', $1)",
                    np.full(1024, 0.5, dtype=np.float32),
                )
                return await conn.fetchval("SELECT embedding FROM memories WHERE id = 'v'")

        value = run(roundtrip())
        assert isinstance(value, np.ndarray) and value.shape == (1024,)
        assert storage.pool.get_min_size() == settings.PG_POOL_MIN
        assert storage.pool.get_max_size() == settings.PG_POOL_MAX
    finally:
        run(close(storage))
    assert run(column_type(db_conn, "memories", "embedding")) == "vector(1024)"


def test_server_starts_twice_on_the_same_database_with_auto_migrate(fresh_db, run):
    first = run(open_storage(fresh_db, auto_migrate=True))
    run(close(first))
    second = run(open_storage(fresh_db, auto_migrate=True))  # restart: migrations re-run
    run(close(second))


def test_two_servers_starting_at_once_on_an_empty_database(fresh_db, db_conn, run):
    # apply_migrations holds pg_advisory_lock, so the two runs serialise.
    async def scenario():
        import src.storage.postgres_enhanced as pe
        from integration_helpers import settings_pointed_at

        a, b = pe.EnhancedPostgresStorage(), pe.EnhancedPostgresStorage()
        with settings_pointed_at(fresh_db, auto_migrate=True):
            results = await asyncio.gather(a.initialize(), b.initialize(), return_exceptions=True)
        await close(a, b)
        return results

    assert run(scenario()) == [None, None]
    assert {"memories", "memory_links", "tag_embeddings"} <= run(public_tables(db_conn))


def test_without_auto_migrate_an_empty_database_is_left_untouched(fresh_db, db_conn, run):
    # A least-privilege runtime role cannot run DDL, so initialize() must not try.
    # On an unmigrated database the pool then cannot register the vector type.
    with pytest.raises(ValueError, match="vector"):
        run(open_storage(fresh_db, auto_migrate=False))

    assert run(public_tables(db_conn)) == set()
    assert run(db_conn.fetchval("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")) == 0


def test_without_auto_migrate_a_migrated_database_starts_and_no_ddl_runs(fresh_db, db_conn, run):
    run(apply_migrations(db_conn))
    run(db_conn.execute("DROP VIEW memory_stats"))  # a migration would re-create it

    storage = run(open_storage(fresh_db, auto_migrate=False))
    try:
        async def query():
            async with storage.pool.acquire() as conn:
                return await conn.fetchval("SELECT count(*) FROM memories")

        assert run(query()) == 0
    finally:
        run(close(storage))
    assert run(db_conn.fetchval("SELECT count(*) FROM pg_views WHERE viewname = 'memory_stats'")) == 0
