"""Temporal 'after' links, created only by the create_temporal_links trigger (migration 002).

Contract: from_id = the newer memory, to_id = an earlier one, link_type 'after';
window (NEW.created_at - 24h, NEW.created_at); weight 0.9 under 1 h, 0.7 under
6 h, else 0.5; confidence 0.7; at most 5 links; chunks never take part.
"""
import time

import pytest

from integration_helpers import open_storage, store_like_server
from src.storage.migrations import apply_migrations

LONG_TEXT = "\n".join(
    f"## Part {n}\n" + " ".join(f"Observation {n}.{i} about the tide table and the harbour." for i in range(22))
    for n in range(3)
)


async def links(conn):
    rows = await conn.fetch(
        "SELECT from_id, to_id, link_type, weight, confidence FROM memory_links ORDER BY 1, 2")
    return [(r["from_id"], r["to_id"], r["link_type"], round(r["weight"], 3), round(r["confidence"], 3))
            for r in rows]


def test_store_links_parents_only_newer_to_earlier(fresh_db, db_conn, encoder, run):
    assert len(LONG_TEXT) > 1500
    storage = run(open_storage(fresh_db, encoder))
    try:
        first = run(store_like_server(storage, "Short note: the ferry left ten minutes late."))
        time.sleep(1.1)
        second = run(store_like_server(storage, LONG_TEXT))
    finally:
        run(storage.close())

    chunk_ids = {r[0] for r in run(db_conn.fetch(
        "SELECT id FROM memories WHERE parent_id = $1", second))}
    assert len(chunk_ids) >= 2, "the long memory should have been chunked"

    assert run(links(db_conn)) == [(second, first, "after", 0.9, 0.7)]
    touching_chunks = run(db_conn.fetchval(
        "SELECT count(*) FROM memory_links WHERE from_id = ANY($1) OR to_id = ANY($1)", list(chunk_ids)))
    assert touching_chunks == 0


@pytest.fixture
def migrated(db_conn, run):
    run(apply_migrations(db_conn))
    return db_conn


def insert_at(conn, run, mid, ts, parent=None):
    run(conn.execute(
        "INSERT INTO memories (id, content, created_at, parent_id) VALUES ($1, $2, $3::text::timestamptz, $4)",
        mid, f"memory {mid}", ts, parent))


def test_imported_historical_memories_link_within_their_own_day(migrated, run):
    conn = migrated
    insert_at(conn, run, "now-1", "now")  # a memory from today: must not link to 2024
    # an import of 2024 memories, in chronological order, with explicit created_at
    insert_at(conn, run, "m0", "2024-03-10 08:00+00")
    insert_at(conn, run, "m1", "2024-03-10 09:30+00")   # 1.5 h after m0
    insert_at(conn, run, "m1_c0", "2024-03-10 09:31+00", parent="m1")  # a chunk: ignored
    insert_at(conn, run, "m2", "2024-03-10 09:50+00")   # 20 min after m1, 1 h 50 after m0
    insert_at(conn, run, "m3", "2024-03-10 20:00+00")   # 10-12 h after m0..m2
    insert_at(conn, run, "m4", "2024-03-11 10:00+00")   # 14 h after m3, > 24 h after m0..m2

    assert run(links(conn)) == sorted([
        ("m1", "m0", "after", 0.7, 0.7),
        ("m2", "m0", "after", 0.7, 0.7),
        ("m2", "m1", "after", 0.9, 0.7),
        ("m3", "m0", "after", 0.5, 0.7),
        ("m3", "m1", "after", 0.5, 0.7),
        ("m3", "m2", "after", 0.5, 0.7),
        ("m4", "m3", "after", 0.5, 0.7),
    ])


def test_at_most_five_after_links_per_memory(migrated, run):
    for i in range(7):
        insert_at(migrated, run, f"e{i}", f"2024-05-01 10:0{i}+00")
    out = [l for l in run(links(migrated)) if l[0] == "e6"]
    assert [l[1] for l in out] == ["e1", "e2", "e3", "e4", "e5"]  # the five most recent
