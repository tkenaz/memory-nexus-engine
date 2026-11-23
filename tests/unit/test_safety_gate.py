"""The integration safety gate itself (tests/dbsafety.py)."""
import pytest

import dbsafety


@pytest.fixture(autouse=True)
def _no_local_forbid(monkeypatch):
    monkeypatch.delenv(dbsafety.FORBID_ENV, raising=False)


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://postgres:pw@127.0.0.1:55432/memory_nexus",  # no "test" in the name
        "postgresql://postgres:pw@127.0.0.1:55432/postgres",
        "postgres://postgres:pw@db.example:5432/production",
    ],
)
def test_databases_without_test_in_the_name_are_refused(dsn):
    with pytest.raises(dbsafety.UnsafeDsn):
        dbsafety.assert_safe(dbsafety.parse_dsn(dsn))


def test_throwaway_target_is_accepted_and_enables_the_layer():
    target = dbsafety.parse_dsn("postgresql://postgres:s%40cret@127.0.0.1:55432/memory_nexus_test")
    dbsafety.assert_safe(target)
    assert dbsafety.skip_reason(target) is None
    assert target.password == "s@cret"
    assert target.pg_env() == {
        "PG_HOST": "127.0.0.1", "PG_PORT": "55432", "PG_DB": "memory_nexus_test",
        "PG_USER": "postgres", "PG_PASSWORD": "s@cret",
    }
    assert "s@cret" not in target.redacted


def test_the_default_port_is_fine_without_a_local_rule():
    # A CI service container usually listens on 5432; that alone is not a reason to refuse.
    dbsafety.assert_safe(dbsafety.parse_dsn("postgresql://postgres:pw@localhost/ci_test"))


@pytest.mark.parametrize(
    "forbid, dsn",
    [
        ("5432", "postgresql://postgres:pw@localhost/memory_nexus_test"),  # port omitted = 5432
        ("5432", "postgresql://postgres:pw@10.0.0.5:5432/anything_test"),
        ("live_test", "postgresql://postgres:pw@127.0.0.1:55432/LIVE_TEST"),
        (" other , 6543 ", "postgresql://postgres:pw@127.0.0.1:6543/memory_nexus_test"),
    ],
)
def test_local_forbid_list_refuses_ports_and_databases(monkeypatch, forbid, dsn):
    monkeypatch.setenv(dbsafety.FORBID_ENV, forbid)
    with pytest.raises(dbsafety.UnsafeDsn):
        dbsafety.assert_safe(dbsafety.parse_dsn(dsn))


def test_missing_dsn_disables_the_layer_with_instructions():
    assert dbsafety.DSN_ENV in dbsafety.skip_reason(None)


def test_derived_database_names_stay_safe():
    target = dbsafety.parse_dsn("postgresql://postgres:pw@127.0.0.1:55432/memory_nexus_test")
    derived = target.with_database("memory_nexus_test_search_1a2b3c4d")
    dbsafety.assert_safe(derived)
    assert derived.port == 55432
    with pytest.raises(dbsafety.UnsafeDsn):
        dbsafety.assert_safe(target.with_database("postgres"))


@pytest.mark.parametrize("dsn", ["mysql://u:p@h:1/x_test", "postgresql://u:p@h:55432/"])
def test_malformed_dsn_is_rejected(dsn):
    with pytest.raises(ValueError):
        dbsafety.parse_dsn(dsn)
