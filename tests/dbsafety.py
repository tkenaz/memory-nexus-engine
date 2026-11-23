"""Safety gate for the integration layer.

The integration tests create and drop databases. They must only ever touch a
throwaway PostgreSQL server. The rules:

* no MEMORY_NEXUS_TEST_DSN                 -> integration tests are skipped
* database name without "test"              -> integration tests are skipped
* every connection, including the scenario databases the tests create, is
  checked first: a database name without "test" is refused
* MEMORY_NEXUS_TEST_FORBID (optional, comma-separated) adds local refusals:
  a number is a port, anything else a database name, e.g. "prod_db,5432".
  A matching DSN aborts the whole run before anything connects.
"""
import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import unquote, urlsplit

DSN_ENV = "MEMORY_NEXUS_TEST_DSN"
FORBID_ENV = "MEMORY_NEXUS_TEST_FORBID"
REQUIRED_DB_SUBSTRING = "test"


def forbidden_targets() -> tuple:
    """(database names, ports) refused on this machine, from FORBID_ENV."""
    names, ports = set(), set()
    for item in os.environ.get(FORBID_ENV, "").split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            ports.add(int(item))
        else:
            names.add(item.lower())
    return frozenset(names), frozenset(ports)


class UnsafeDsn(Exception):
    """The DSN points somewhere the tests must never connect to."""


@dataclass(frozen=True)
class TestDsn:
    __test__ = False  # not a test class, despite the name

    host: str
    port: int
    database: str
    user: str
    password: str

    def with_database(self, database: str) -> "TestDsn":
        return TestDsn(self.host, self.port, database, self.user, self.password)

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def redacted(self) -> str:
        return f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.database}"

    def pg_env(self) -> dict:
        """PG_* variables read by config/settings.py."""
        return {
            "PG_HOST": self.host,
            "PG_PORT": str(self.port),
            "PG_DB": self.database,
            "PG_USER": self.user,
            "PG_PASSWORD": self.password,
        }


def parse_dsn(dsn: str) -> TestDsn:
    parts = urlsplit(dsn)
    if parts.scheme not in ("postgresql", "postgres"):
        raise ValueError(f"{DSN_ENV} must start with postgresql://, got scheme {parts.scheme!r}")
    database = unquote(parts.path.lstrip("/"))
    if not database:
        raise ValueError(f"{DSN_ENV} must name a database")
    return TestDsn(
        host=parts.hostname or "localhost",
        port=parts.port or 5432,  # libpq default
        database=database,
        user=unquote(parts.username or "postgres"),
        password=unquote(parts.password or ""),
    )


def assert_safe(target: TestDsn) -> None:
    """Raise UnsafeDsn unless the target is a throwaway test database. Call before every connect."""
    if REQUIRED_DB_SUBSTRING not in target.database.lower():
        raise UnsafeDsn(
            f"refusing database {target.database!r}: only databases whose name contains "
            f"{REQUIRED_DB_SUBSTRING!r} are used"
        )
    names, ports = forbidden_targets()
    if target.database.lower() in names:
        raise UnsafeDsn(f"refusing database {target.database!r}: listed in {FORBID_ENV}")
    if target.port in ports:
        raise UnsafeDsn(f"refusing port {target.port} ({target.redacted}): listed in {FORBID_ENV}")


def skip_reason(target: Optional[TestDsn]) -> Optional[str]:
    """Why the integration layer cannot run, or None if it can."""
    if target is None:
        return (
            f"integration layer disabled: set {DSN_ENV}=postgresql://postgres:test@127.0.0.1:55432/"
            "memory_nexus_test (a throwaway pgvector server; see tests/README.md)"
        )
    if REQUIRED_DB_SUBSTRING not in target.database.lower():
        return (
            f"integration layer disabled: database {target.database!r} in {DSN_ENV} does not "
            f"contain {REQUIRED_DB_SUBSTRING!r}; only throwaway test databases are used"
        )
    return None


# Set once by tests/conftest.py (pytest_configure); read by the integration fixtures.
ACTIVE_DSN: Optional[TestDsn] = None
SKIP_REASON: Optional[str] = "safety gate not evaluated"
