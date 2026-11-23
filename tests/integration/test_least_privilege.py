"""README "Running with least privilege", executed for real.

The SQL is read from the README itself, so the test fails if the documented
recipe stops working. Only the database name and the password placeholder are
substituted. The owner (the DSN role) applies migrations; the runtime role then
stores and searches with AUTO_MIGRATE=false.
"""
import re
import secrets
import time

import asyncpg
import pytest

from integration_helpers import REPO_ROOT, open_storage, store_like_server
from src.storage.migrations import apply_migrations

SECTION = "### Running with least privilege"


def readme_recipe():
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    if SECTION not in text:
        pytest.fail(f"README.md has no '{SECTION}' section")
    block = re.search(r"```sql\n(.*?)```", text[text.index(SECTION):], re.S)
    assert block, "no ```sql block under the least-privilege section"
    statements = [s.strip() for s in re.sub(r"--[^\n]*", "", block.group(1)).split(";") if s.strip()]
    role = re.search(r"CREATE ROLE (\w+)", block.group(1)).group(1)
    return role, statements


def test_runtime_role_from_readme_can_store_and_search(server, fresh_db, db_conn, encoder, run):
    role, statements = readme_recipe()
    password = secrets.token_hex(12)

    # owner: schema first
    run(apply_migrations(db_conn))

    exists = run(server.conn.fetchval("SELECT 1 FROM pg_roles WHERE rolname = $1", role))
    if exists:
        pytest.skip(f"role {role} already exists on the throwaway server (left by an aborted run?)")

    for stmt in statements:
        stmt = stmt.replace("PASSWORD '...'", f"PASSWORD '{password}'")
        stmt = re.sub(r"ON DATABASE \w+", f'ON DATABASE "{fresh_db.database}"', stmt)
        run(db_conn.execute(stmt))

    runtime = fresh_db.__class__(fresh_db.host, fresh_db.port, fresh_db.database, role, password)
    try:
        storage = run(open_storage(runtime, encoder, auto_migrate=False))
        try:
            first = run(store_like_server(
                storage, "The harbour ferry timetable changed for the winter season.",
                {"tags": ["Harbour", "ferry"]}))
            time.sleep(1.1)
            second = run(store_like_server(
                storage, "Walked along the harbour wall and counted the moored sailing boats.",
                {"tags": ["harbour"]}))
            results = run(storage.search_with_graph("harbour ferry timetable", limit=5))
        finally:
            run(storage.close())

        # the trigger's INSERT into memory_links ran as the runtime role
        after_links = run(db_conn.fetch(
            "SELECT from_id, to_id, link_type FROM memory_links WHERE link_type = 'after'"))
        assert [tuple(r) for r in after_links] == [(second, first, "after")]
        # tag catalog: INSERT for new tags, UPDATE (ON CONFLICT / counter) for a known one
        catalog = {r["tag"]: r["memory_count"] for r in run(db_conn.fetch(
            "SELECT tag, memory_count FROM tag_embeddings WHERE embedding IS NOT NULL"))}
        assert catalog == {"harbour": 2, "ferry": 1}
        assert results and results[0]["id"] == first

        # and the role really is least-privilege
        async def denied():
            conn = await server.connect(runtime)
            outcomes = {}
            try:
                for name, sql in {
                    "create table": "CREATE TABLE sneaky (x int)",
                    "delete": "DELETE FROM memories",
                    "drop": "DROP TABLE memory_links",
                }.items():
                    try:
                        await conn.execute(sql)
                        outcomes[name] = "allowed"
                    except asyncpg.InsufficientPrivilegeError:
                        outcomes[name] = "denied"
            finally:
                await conn.close()
            return outcomes

        assert run(denied()) == {"create table": "denied", "delete": "denied", "drop": "denied"}
    finally:
        # DROP OWNED BY also revokes the CONNECT grant on the database (a shared object)
        run(db_conn.execute(f'DROP OWNED BY "{role}"'))
        run(server.conn.execute(f'DROP ROLE IF EXISTS "{role}"'))
