"""Pure helpers of EnhancedPostgresStorage (src/storage/postgres_enhanced.py).

No database and no model: importing the module pulls in FlagEmbedding (a few
seconds), but the BGE-M3 weights are loaded lazily and never here.
`_preprocess_temporal_query` is the date parser the live search path uses.
"""
import asyncio
from datetime import datetime, timezone

import numpy as np
import pytest

import src.storage.postgres_enhanced as pe

UTC = timezone.utc
NOW = datetime(2026, 3, 4, 10, 0, tzinfo=UTC)  # a Wednesday


@pytest.fixture(scope="module")
def storage():
    return pe.EnhancedPostgresStorage()  # no pool, no encoder


def freeze(monkeypatch, tz):
    """Freeze 'now' at NOW and make the query time zone `tz`."""
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)

    monkeypatch.setattr(pe, "datetime", FrozenDateTime)
    monkeypatch.setattr(pe, "_query_timezone", lambda: tz)


@pytest.fixture
def frozen_now(monkeypatch):
    freeze(monkeypatch, UTC)


def utc(y, m, d, h=0, mi=0, s=0):
    return datetime(y, m, d, h, mi, s, tzinfo=UTC)


# --------------------------------------------------------------------------- dates
@pytest.mark.parametrize(
    "query, clean, flt",
    [
        ("October 2025 trip", "trip", {"$gte": utc(2025, 10, 1), "$lte": utc(2025, 10, 31, 23, 59, 59)}),
        ("октябрь 2025", "", {"$gte": utc(2025, 10, 1), "$lte": utc(2025, 10, 31, 23, 59, 59)}),
        ("встречи сентября 2024", "встречи", {"$gte": utc(2024, 9, 1), "$lte": utc(2024, 9, 30, 23, 59, 59)}),
        ("february 2024", "", {"$gte": utc(2024, 2, 1), "$lte": utc(2024, 2, 29, 23, 59, 59)}),
        ("21 августа", "", {"$gte": utc(2026, 8, 21), "$lt": utc(2026, 8, 22)}),
        ("notes 5 march", "notes", {"$gte": utc(2026, 3, 5), "$lt": utc(2026, 3, 6)}),
        ("Lisbon yesterday", "Lisbon", {"$gte": utc(2026, 3, 3), "$lt": utc(2026, 3, 4)}),
        ("что было вчера", "что было", {"$gte": utc(2026, 3, 3), "$lt": utc(2026, 3, 4)}),
        ("сегодня", "", {"$gte": utc(2026, 3, 4)}),
        ("this week", "", {"$gte": utc(2026, 3, 2)}),
        ("last week", "", {"$gte": utc(2026, 2, 23), "$lt": utc(2026, 3, 2)}),
        ("прошлая неделя", "", {"$gte": utc(2026, 2, 23), "$lt": utc(2026, 3, 2)}),
        ("this month", "", {"$gte": utc(2026, 3, 1)}),
        ("прошлый месяц", "", {"$gte": utc(2026, 2, 1), "$lt": utc(2026, 3, 1)}),
    ],
)
def test_temporal_markers_become_created_at_filters(storage, frozen_now, query, clean, flt):
    assert storage._preprocess_temporal_query(query) == (clean, {"created_at": flt})


@pytest.mark.parametrize("query", ["Lisbon trip", "рецепт супа", "pgvector index"])
def test_queries_without_dates_pass_through(storage, frozen_now, query):
    assert storage._preprocess_temporal_query(query) == (query, None)


@pytest.mark.parametrize("query", ["meeting 2025-10-03", "встреча 03.10.2025"])
def test_numeric_dates_filter_one_day(storage, frozen_now, query):
    _, flt = storage._preprocess_temporal_query(query)
    assert flt == {"created_at": {"$gte": utc(2025, 10, 3), "$lt": utc(2025, 10, 4)}}


def test_day_month_year_filters_one_day(storage, frozen_now):
    clean, flt = storage._preprocess_temporal_query("21 августа 2025")
    assert clean == ""
    assert flt == {"created_at": {"$gte": utc(2025, 8, 21), "$lt": utc(2025, 8, 22)}}


def test_nominative_january_is_a_month(storage, frozen_now):
    _, flt = storage._preprocess_temporal_query("январь 2025")
    assert flt == {"created_at": {"$gte": utc(2025, 1, 1), "$lte": utc(2025, 1, 31, 23, 59, 59)}}


def test_today_is_the_local_day(monkeypatch):
    # 2026-03-04 10:00 UTC is 19:00 in Tokyo: "today" starts at Tokyo midnight.
    from zoneinfo import ZoneInfo
    tokyo = ZoneInfo("Asia/Tokyo")
    freeze(monkeypatch, tokyo)
    storage = pe.EnhancedPostgresStorage()
    _, flt = storage._preprocess_temporal_query("today")
    assert flt == {"created_at": {"$gte": datetime(2026, 3, 4, tzinfo=tokyo)}}


@pytest.mark.parametrize("query, day", [
    ("позавчера", (2026, 3, 2)),
    ("August 21, 2025", (2025, 8, 21)),
])
def test_more_date_forms(storage, frozen_now, query, day):
    _, flt = storage._preprocess_temporal_query(query)
    start = utc(*day)
    assert flt["created_at"]["$gte"] == start


def test_impossible_date_is_not_a_filter(storage, frozen_now):
    assert storage._preprocess_temporal_query("ticket 31.02.2025") == ("ticket 31.02.2025", None)

# ----------------------------------------------------------- chunk -> parent resolution
class FakeConn:
    """Stands in for an asyncpg connection: serves parent rows by id."""

    def __init__(self, parents):
        self.parents = parents
        self.calls = []

    async def fetch(self, sql, ids):
        self.calls.append(sorted(ids))
        return [self.parents[i] for i in ids if i in self.parents]


def resolve(storage, rows, parents):
    conn = FakeConn(parents)
    out = asyncio.run(storage._resolve_to_parents(rows, conn))
    return out, conn


PARENT = {"id": "p1", "content": "full parent text", "metadata": {"tags": ["x"]},
          "sashf": {}, "created_at": NOW}


def test_resolve_empty(storage):
    assert resolve(storage, [], {})[0] == []


def test_standalone_rows_pass_through_unchanged(storage):
    rows = [{"id": "a", "parent_id": None, "similarity": 0.4, "content": "a"},
            {"id": "b", "parent_id": None, "similarity": 0.9, "content": "b"}]
    out, conn = resolve(storage, rows, {})
    assert [r["id"] for r in out] == ["b", "a"]
    assert out[1] == rows[0]
    assert conn.calls == []  # nothing to load


def test_chunks_collapse_to_their_parent_with_the_best_chunk_score(storage):
    rows = [
        {"id": "p1_c0", "parent_id": "p1", "similarity": 0.55, "content": "chunk 0"},
        {"id": "s", "parent_id": None, "similarity": 0.60, "content": "standalone"},
        {"id": "p1_c2", "parent_id": "p1", "similarity": 0.80, "content": "chunk 2"},
    ]
    out, conn = resolve(storage, rows, {"p1": PARENT})

    assert [r["id"] for r in out] == ["p1", "s"]
    parent = out[0]
    assert parent["content"] == "full parent text"
    assert parent["metadata"] == {"tags": ["x"]}
    assert parent["similarity"] == pytest.approx(0.80)
    assert parent["resolved_from_chunk"] is True
    assert "parent_id" not in parent
    assert conn.calls == [["p1"]]  # one batched lookup


def test_chunk_whose_parent_is_missing_passes_through(storage):
    rows = [{"id": "gone_c0", "parent_id": "gone", "similarity": 0.7, "content": "orphan"}]
    out, _ = resolve(storage, rows, {})
    assert out == rows


# ----------------------------------------------------------------- small helpers
@pytest.mark.parametrize(
    "sashf, key",
    [
        ({}, None),
        ({"actors": ["anna"]}, None),  # one component is not a pattern
        ({"actors": ["oleg", "anna", "boris"], "act": "гулять"}, "A:anna,boris|V:гулять"),
        ({"act": "писать", "manner": "быстро", "affect": {"bucket": "V+0.3A+0.0D+0.0"}},
         "V:писать|M:быстро|E:V+0.3A+0.0D+0.0"),
    ],
)
def test_pattern_key(storage, sashf, key):
    assert storage._generate_pattern_key(sashf) == key


def test_to_numpy():
    as_list = pe.EnhancedPostgresStorage._to_numpy([0.5, 1.0])
    as_array = pe.EnhancedPostgresStorage._to_numpy(np.array([0.5, 1.0], dtype=np.float64))
    assert as_list.dtype == np.float32 and as_array.dtype == np.float32
    assert as_list.tolist() == [0.5, 1.0]
    assert pe.EnhancedPostgresStorage._to_numpy(None) is None
    assert pe.EnhancedPostgresStorage._to_numpy("not a vector") is None
