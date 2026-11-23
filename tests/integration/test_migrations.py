"""Schema: migrations/*.sql applied by src/storage/migrations.py and init_db.py."""
import asyncio
import subprocess
import sys

import asyncpg
import numpy as np
import pytest

from config.settings import settings
from integration_helpers import LEGACY_MIGRATIONS, REPO_ROOT, column_type, index_def, pg_env
from src.storage.migrations import apply_migrations, migration_files

TABLES = {"memories", "memory_links", "memory_patterns", "tag_embeddings"}
VIEWS = {"memory_stats_basic", "memory_stats", "memory_graph"}
TRIGGERS = {
    ("memories", "trigger_memories_updated_at"),
    ("memories", "auto_temporal_links"),
    ("memory_links", "update_memory_links_updated_at"),
    ("memory_patterns", "update_memory_patterns_updated_at"),
}


async def schema_snapshot(conn) -> dict:
    """Catalog facts that must not change when migrations are re-applied."""
    return {
        "columns": [tuple(r) for r in await conn.fetch(
            """SELECT table_name, column_name, data_type, udt_name FROM information_schema.columns
               WHERE table_schema = 'public' ORDER BY 1, 2""")],
        "indexes": [tuple(r) for r in await conn.fetch(
            "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' ORDER BY 1")],
        "triggers": [tuple(r) for r in await conn.fetch(
            """SELECT tgrelid::regclass::text, tgname FROM pg_trigger
               WHERE NOT tgisinternal ORDER BY 1, 2""")],
        "views": [r[0] for r in await conn.fetch(
            "SELECT viewname FROM pg_views WHERE schemaname = 'public' ORDER BY 1")],
        "functions": [tuple(r) for r in await conn.fetch(
            """SELECT proname, count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
               WHERE n.nspname = 'public' AND p.prolang = (SELECT oid FROM pg_language WHERE lanname = 'plpgsql')
               GROUP BY 1 ORDER BY 1""")],
    }


def init_db(dsn) -> subprocess.CompletedProcess:
    """Run the real CLI against dsn (PG_* environment, as documented)."""
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "init_db.py")],
        cwd=REPO_ROOT / "tests", env=pg_env(dsn),
        capture_output=True, text=True, timeout=120,
    )


def vec_literal(values) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


# --------------------------------------------------------------------- fresh install
def test_fresh_install_builds_the_bge_m3_schema(db_conn, run):
    applied = run(apply_migrations(db_conn))
    assert applied == len(migration_files()) >= 6

    async def facts():
        snap = await schema_snapshot(db_conn)
        return {
            "embedding": await column_type(db_conn, "memories", "embedding"),
            "tag_embedding": await column_type(db_conn, "tag_embeddings", "embedding"),
            "example_ids": await column_type(db_conn, "memory_patterns", "example_ids"),
            "vector_index": await index_def(db_conn, "idx_memories_embedding"),
            "tables": {r[0] for r in await db_conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")},
            "snap": snap,
        }

    f = run(facts())
    assert f["embedding"] == f"vector({settings.EMBEDDING_DIMENSION})" == "vector(1024)"
    assert f["tag_embedding"] == "vector(1024)"
    assert f["example_ids"] == "text[]"
    assert "USING hnsw (embedding vector_cosine_ops)" in f["vector_index"]
    assert TABLES <= f["tables"]
    assert VIEWS <= set(f["snap"]["views"])
    assert TRIGGERS <= set(f["snap"]["triggers"])
    # one definition of the temporal-link trigger function
    assert ("create_temporal_links", 1) in f["snap"]["functions"]


def test_applying_migrations_twice_is_idempotent(db_conn, run):
    run(apply_migrations(db_conn))
    before = run(schema_snapshot(db_conn))
    run(db_conn.execute("INSERT INTO memories (id, content) VALUES ('keep-me', 'survives a re-run')"))

    assert run(apply_migrations(db_conn)) == len(migration_files())

    assert run(schema_snapshot(db_conn)) == before
    assert run(db_conn.fetchval("SELECT content FROM memories WHERE id = 'keep-me'")) == "survives a re-run"


def test_init_db_cli_twice_on_a_fresh_database(fresh_db, db_conn, run):
    for attempt in (1, 2):
        proc = init_db(fresh_db)
        assert proc.returncode == 0, f"run {attempt}: {proc.stdout}\n{proc.stderr}"
        assert f"Schema is up to date ({len(migration_files())} migration files)" in proc.stdout
    assert run(column_type(db_conn, "memories", "embedding")) == "vector(1024)"


def test_concurrent_apply_migrations_on_an_empty_database(server, fresh_db, run):
    async def scenario():
        a, b = await server.connect(fresh_db), await server.connect(fresh_db)
        try:
            return await asyncio.gather(apply_migrations(a), apply_migrations(b))
        finally:
            await a.close()
            await b.close()

    assert run(scenario()) == [len(migration_files())] * 2
    conn = run(server.connect(fresh_db))
    try:
        assert run(column_type(conn, "memories", "embedding")) == "vector(1024)"
    finally:
        run(conn.close())


# ---------------------------------------------------------------- 384 -> 1024 upgrade
@pytest.fixture
def legacy_db(db_conn, run):
    """A database exactly as the pre-BGE-M3 snapshot created it: vector(384), IVFFlat."""
    run(apply_migrations(db_conn, LEGACY_MIGRATIONS))
    assert run(column_type(db_conn, "memories", "embedding")) == "vector(384)"
    assert "USING ivfflat" in run(index_def(db_conn, "idx_memories_embedding"))
    return db_conn


def test_005_refuses_to_drop_existing_384_vectors(legacy_db, run):
    vector = np.linspace(-1, 1, 384)
    run(legacy_db.execute(
        "INSERT INTO memories (id, content, embedding) VALUES ('old', 'old e5 era memory', $1::vector)",
        vec_literal(vector),
    ))
    # bring the legacy database up to 004 with today's files, then run 005 alone
    run(apply_migrations(legacy_db, [p for p in migration_files() if p.name < "005"]))
    m005 = [p for p in migration_files() if p.name.startswith("005")]

    with pytest.raises(asyncpg.RaiseError) as excinfo:
        run(apply_migrations(legacy_db, m005))

    message = str(excinfo.value)
    assert "vector(384)" in message and "holds 1 vectors" in message
    assert "cannot be cast to vector(1024)" in message
    assert "UPDATE memories SET embedding = NULL" in message  # tells the operator what to do
    # nothing changed: the whole file ran in one implicit transaction
    assert run(column_type(legacy_db, "memories", "embedding")) == "vector(384)"
    stored = run(legacy_db.fetchval("SELECT embedding::text FROM memories WHERE id = 'old'"))
    assert np.allclose(np.array(stored.strip("[]").split(","), dtype=float), vector, atol=1e-5)
    assert "USING ivfflat" in run(index_def(legacy_db, "idx_memories_embedding"))
    assert run(legacy_db.fetchval("SELECT total_memories FROM memory_stats_basic")) == 1


def test_init_db_upgrades_an_empty_384_column_to_1024_hnsw(fresh_db, legacy_db, run):
    run(legacy_db.execute("INSERT INTO memories (id, content) VALUES ('kept', 'text without a vector')"))

    proc = init_db(fresh_db)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert run(column_type(legacy_db, "memories", "embedding")) == "vector(1024)"
    assert "USING hnsw (embedding vector_cosine_ops)" in run(index_def(legacy_db, "idx_memories_embedding"))
    assert run(legacy_db.fetchval("SELECT content FROM memories WHERE id = 'kept'")) == "text without a vector"
    stats = run(legacy_db.fetchrow("SELECT * FROM memory_stats_basic"))
    assert (stats["total_memories"], stats["memories_with_embeddings"]) == (1, 0)
    assert run(legacy_db.fetchval("SELECT total_memories FROM memory_stats")) == 1
    # and the upgraded database stays re-runnable
    assert init_db(fresh_db).returncode == 0


def test_init_db_stops_with_a_clear_message_when_384_vectors_exist(fresh_db, legacy_db, run):
    run(legacy_db.execute(
        "INSERT INTO memories (id, content, embedding) VALUES ('old', 'x', $1::vector)",
        vec_literal([0.5] * 384),
    ))

    proc = init_db(fresh_db)

    assert proc.returncode != 0
    assert "cannot be cast to vector(1024)" in proc.stderr
    assert run(column_type(legacy_db, "memories", "embedding")) == "vector(384)"
    assert run(legacy_db.fetchval("SELECT count(*) FROM memories WHERE embedding IS NOT NULL")) == 1


def test_005_is_a_no_op_on_1024_with_data(db_conn, run):
    run(apply_migrations(db_conn))
    vector = vec_literal(np.full(1024, 0.25))
    run(db_conn.execute(
        "INSERT INTO memories (id, content, embedding) VALUES ('new', 'bge era', $1::vector)", vector))
    before = run(schema_snapshot(db_conn))

    run(apply_migrations(db_conn, [p for p in migration_files() if p.name.startswith("005")]))

    assert run(schema_snapshot(db_conn)) == before
    assert run(db_conn.fetchval("SELECT vector_dims(embedding) FROM memories WHERE id = 'new'")) == 1024
