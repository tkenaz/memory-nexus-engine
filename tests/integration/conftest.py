"""Fixtures for the integration layer.

Everything here runs against the throwaway server named by MEMORY_NEXUS_TEST_DSN
(see tests/dbsafety.py and tests/README.md). Each scenario gets its own freshly
created database on that server, named <dsn database>_<scenario>_<random>, and
drops it afterwards; the database named in the DSN is only used as the
maintenance connection. Every connect goes through dbsafety.assert_safe first.

Async code runs on one session-wide event loop (`run` fixture) instead of
pytest-asyncio, so session-scoped pools and the loaded model can be shared.
"""
import asyncio
import time

import asyncpg
import pytest

import dbsafety
from integration_helpers import ThrowawayServer


# --------------------------------------------------------------------------- loop
@pytest.fixture(scope="session")
def aio_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
    asyncio.set_event_loop(None)


@pytest.fixture(scope="session")
def run(aio_loop):
    """run(coro) -> result, on the shared session loop."""
    return aio_loop.run_until_complete


# ------------------------------------------------------------------------ server
@pytest.fixture(scope="session")
def server(run):
    target = dbsafety.ACTIVE_DSN
    if target is None:
        pytest.skip(dbsafety.SKIP_REASON)
    dbsafety.assert_safe(target)

    # config/settings.py must see the same target the gate approved.
    from config.settings import settings

    approved = (target.host, target.port, target.database)
    actual = (settings.PG_HOST, int(settings.PG_PORT), settings.PG_DB)
    if actual != approved:
        pytest.exit(f"config.settings points at {actual}, the safety gate approved {approved}", 2)

    async def connect_with_retry():
        last = None
        for _ in range(60):  # a freshly started container needs a few seconds
            try:
                return await asyncpg.connect(target.url, timeout=5)
            except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as exc:
                last = exc
                await asyncio.sleep(0.5)
        pytest.fail(f"cannot connect to the throwaway server {target.redacted}: {last!r}")

    conn = run(connect_with_retry())
    srv = ThrowawayServer(target, conn)
    yield srv
    for name in list(srv.created):
        run(srv.drop_database(target.with_database(name)))
    run(conn.close())


@pytest.fixture
def fresh_db(server, run):
    """An empty database (no extensions, no tables), dropped after the test."""
    dsn = run(server.create_database("t"))
    yield dsn
    run(server.drop_database(dsn))


@pytest.fixture
def db_conn(server, fresh_db, run):
    """A plain connection to fresh_db."""
    conn = run(server.connect(fresh_db))
    yield conn
    run(conn.close())


# ------------------------------------------------------------------------- model
def _require_cached_model():
    from config.settings import settings

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(settings.EMBEDDING_MODEL, local_files_only=True)
    except Exception as exc:  # LocalEntryNotFoundError and friends
        pytest.skip(
            f"{settings.EMBEDDING_MODEL} is not in the local Hugging Face cache ({exc.__class__.__name__}); "
            "download it once with network access, the tests never download"
        )
    return settings.EMBEDDING_MODEL


@pytest.fixture(scope="session")
def encoder(run):
    """BGE-M3 loaded exactly as the server loads it (EnhancedPostgresStorage._ensure_encoder
    -> src/core/encoder.load_bge_m3), on the device it picks, offline (HF_HUB_OFFLINE=1)."""
    _require_cached_model()
    import src.storage.postgres_enhanced as pe

    loader = pe.EnhancedPostgresStorage()
    started = time.perf_counter()
    run(loader._ensure_encoder())
    devices = getattr(loader.encoder, "target_devices", "?")
    print(f"\n[integration] BGE-M3 loaded on {devices} in {time.perf_counter() - started:.1f}s")
    return loader.encoder


@pytest.fixture(scope="session")
def cpu_reference():
    """BGE-M3 on the CPU in fp32: the reference the server-built encoder is compared to."""
    name = _require_cached_model()
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(name, use_fp16=False, devices="cpu")
