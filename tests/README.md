# Tests

Two layers. Everything the suite creates stays under `tests/`: there is no
`.pytest_cache`, and no `.pyc` files are written into `src/` or `config/`.

| Layer | Needs | Time (M-series Mac) |
|---|---|---|
| `tests/unit/` | Python deps only. No database, no model weights. | ~4 s |
| `tests/integration/` | A throwaway PostgreSQL 16 + pgvector, and BGE-M3 in the local Hugging Face cache | ~20 s |

Python 3.11 with `requirements-dev.txt` installed. Each command below can be run from
any directory: `-c tests/pytest.ini` points at the config, and `tests/conftest.py`
puts the repo root on `sys.path`.

## Unit layer

```bash
python -m pytest -c tests/pytest.ini tests/unit
```

What it covers:

- `test_rank_fusion.py`: `rrf_fuse`. Covers the 1/(k+rank) arithmetic, accumulation across
  engines, normalisation to (0, 1], `engines` in input order, ties, empty input,
  immutability of the inputs, and randomised invariants.
- `test_query_intent.py`, `test_chunker.py`, `test_sashf_parser.py`.
  These run with the clock frozen where dates matter, and use Russian and English phrases.
- `test_storage_helpers.py`: the pure parts of `EnhancedPostgresStorage`, with no database:
  - `_preprocess_temporal_query`, the date parser the search path actually uses;
  - chunk-to-parent resolution;
  - pattern keys.
- `test_encoder_loader.py`: `src/core/encoder.load_bge_m3` with FlagEmbedding faked. Checks
  eager attention on MPS, the CPU fallback, and that the patched runner is always restored.
- `test_safety_gate.py`: the integration gate itself.

## Integration layer

### Safety gate (read this first)

The integration tests create and drop databases, so they only run against a server you
can throw away:

- With no `MEMORY_NEXUS_TEST_DSN`, every integration test is **skipped** and the skip
  message says why.
- If the database name in the DSN does not contain `test`, the whole run is **aborted**
  (`pytest.exit`, exit code 2) before anything connects.
- `MEMORY_NEXUS_TEST_FORBID` adds local refusals, comma-separated: a number is a port,
  anything else a database name (for example `prod_db,5432` if a real server listens
  on your default port). A match aborts the run the same way.
- Every connection the fixtures open is checked again: scenario databases, the settings
  object the server code reads, and the `init_db.py` subprocess.

Each scenario gets its own database, `<dsn-db>_<scenario>_<random>`. It is created on the
throwaway server and dropped afterwards. The database named in the DSN is only used as the
maintenance connection.

### One command: Docker, run, clean up

```bash
PYTHON=/path/to/venv/bin/python tests/run_with_docker.sh          # unit + integration
PYTHON=/path/to/venv/bin/python tests/run_with_docker.sh tests/integration -x
```

The script does the following:

1. Starts `pgvector/pgvector:pg16` as `memory-nexus-test-<random hex>` on a free high port
   picked by Docker, bound to 127.0.0.1.
2. Waits until it accepts TCP connections.
3. Runs pytest with the DSN set.
4. Runs `docker rm -f` on exactly that container on exit, including on failure or Ctrl-C.

### By hand

```bash
docker run -d --name memory-nexus-test-$RANDOM -e POSTGRES_PASSWORD=test \
  -e POSTGRES_DB=memory_nexus_test -p 127.0.0.1:55432:5432 pgvector/pgvector:pg16

MEMORY_NEXUS_TEST_DSN=postgresql://postgres:test@127.0.0.1:55432/memory_nexus_test \
  python -m pytest -c tests/pytest.ini tests/integration

docker rm -f <that container name>
```

To apply the schema to that server yourself:

```bash
PG_HOST=127.0.0.1 PG_PORT=55432 PG_DB=memory_nexus_test PG_USER=postgres PG_PASSWORD=test \
  python init_db.py
```

`init_db.py` reads the same `PG_*` settings as the server (`config/settings.py`).

### The model

The search tests use the real BGE-M3, loaded exactly as the server loads it
(`EnhancedPostgresStorage._ensure_encoder` → `src/core/encoder.load_bge_m3`), on the
device that loader picks.

`HF_HUB_OFFLINE=1` is set, so nothing is downloaded. If the model is not in the local
Hugging Face cache, the model-dependent tests are skipped with a message, never mocked.

`test_embedding_device.py` compares the server-built encoder with a CPU fp32 reference.
On Apple Silicon it also checks that MPS runs with eager attention. Background: torch's
MPS scaled-dot-product attention returns wrong vectors for single inputs of 8 tokens or
fewer, which covers every tag and most short queries.

### What it covers

- `test_migrations.py`:
  - a fresh install gives `vector(1024)` and an HNSW cosine index;
  - `apply_migrations` and `init_db.py` can each run twice;
  - concurrent `apply_migrations` (advisory lock);
  - the upgrade from the pre-BGE-M3 schema (`fixtures/legacy_*.sql` are the migrations from
    git `e7e0935`, `vector(384)` + IVFFlat): migration 005 refuses when 384-d vectors exist,
    with a clear message, and leaves the data untouched; with no vectors it alters the column
    to 1024 + HNSW through `init_db.py`.
- `test_startup.py`: `initialize()` on an empty database; a restart; two servers starting at
  once; `AUTO_MIGRATE=false` on an empty database (no DDL) and on a migrated one; pool size
  from settings.
- `test_temporal_links.py`: the `create_temporal_links` trigger. Covers the
  newer → earlier direction, weights, the 5-link cap, chunks never linked, and imported
  2024 memories linking within their own day.
- `test_least_privilege.py`: runs the SQL from the README section *Running with least
  privilege* as written. The owner migrates; `memory_nexus_rw` with `AUTO_MIGRATE=false`
  stores (including the trigger insert into `memory_links` and the `tag_embeddings` upsert)
  and searches; DDL and `DELETE` are denied.
- `test_search.py`: 16 invented memories in English and Russian, one of them long enough
  to be chunked. A spy on `rrf_fuse` records what each engine handed to fusion. One probe
  per engine:
  - keyword: a rare token;
  - semantic: a Russian query for an English memory, and an English paraphrase with no
    shared stems;
  - tags: a tag that is absent from the text, and on text alone a different memory wins;
  - every engine at once (`semantic, keyword, tags, tag_exact, metadata`, in input order).

  Also covered: a metadata-only hit never outranks the direct match; `%` and `_` are taken
  literally; chunk hits come back as their parent; the temporal-only path; graph neighbours
  are marked, get recency, and are reached in both directions along a link; SASH-F
  `time_bucket` follows the local clock.

## Known issue (strict xfail)

One test encodes behaviour the code does not have yet, marked `xfail(strict=True)`: a
fix makes it XPASS, and strict mode then fails the run so the marker gets removed. Run
with `-rx` to see it.

| Test | Where |
|---|---|
| `integration/test_search.py::test_an_unrelated_tag_does_not_outrank_the_semantic_hit` | `config/settings.py` `TAG_MIN_SIMILARITY=0.5` |

With correct BGE-M3 vectors, short unrelated strings score 0.50-0.53 against tags, so the
current threshold lets unrelated tags vote. The threshold should be chosen from the
distribution of query-to-own-tag and query-to-foreign-tag similarities on real data,
not by eye; until then it stays as is and this test documents the gap.
