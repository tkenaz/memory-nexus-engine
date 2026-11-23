"""Root conftest: import path, side-effect hygiene, and the database safety gate.

Runs before any test module is collected, so the PG_* environment is in place
before config/settings.py builds its module-level `settings` object.
"""
import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

# Source packages (src/, config/) import from the repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# Do not write __pycache__/*.pyc into src/ or config/ (the suite only creates
# files under tests/; pytest's own rewritten test modules stay in tests/).
sys.dont_write_bytecode = True

# Never download a model or tokenizer during tests. The search tests need BGE-M3
# in the local Hugging Face cache; they skip with a clear message if it is missing.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import dbsafety  # noqa: E402  (needs TESTS_DIR on sys.path)


def pytest_configure(config):
    # The outcome is kept in dbsafety.ACTIVE_DSN / dbsafety.SKIP_REASON, which
    # tests/integration/conftest.py reads (conftest modules cannot import each other).
    raw = os.environ.get(dbsafety.DSN_ENV, "").strip()
    target = None
    if raw:
        try:
            target = dbsafety.parse_dsn(raw)
            dbsafety.assert_safe(target)
        except (ValueError, dbsafety.UnsafeDsn) as exc:
            pytest.exit(f"{dbsafety.DSN_ENV} rejected: {exc}", returncode=2)

    dbsafety.SKIP_REASON = dbsafety.skip_reason(target)
    if dbsafety.SKIP_REASON is None:
        dbsafety.ACTIVE_DSN = target
        # Environment beats .env for both python-dotenv (override=False) and
        # pydantic-settings, so a developer's own .env cannot redirect the tests.
        os.environ.update(target.pg_env())


def pytest_collection_modifyitems(config, items):
    integration_dir = TESTS_DIR / "integration"
    reason = dbsafety.SKIP_REASON
    skip = pytest.mark.skip(reason=reason) if reason else None
    for item in items:
        if integration_dir in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.integration)
            if skip is not None:
                item.add_marker(skip)
