"""Reciprocal Rank Fusion: src/search/rank_fusion.py (pure function, no I/O)."""
import copy
import random

import pytest

from src.search.rank_fusion import rrf_fuse


def rec(rid, **extra):
    return {"id": rid, **extra}


def ids(results):
    return [r["id"] for r in results]


def by_id(results):
    return {r["id"]: r for r in results}


def test_single_engine_scores_are_one_over_k_plus_rank_normalised_by_the_best():
    out = rrf_fuse([("semantic", [rec("a"), rec("b"), rec("c")])], k=60)

    assert ids(out) == ["a", "b", "c"]
    raw = {"a": 1 / 61, "b": 1 / 62, "c": 1 / 63}
    for r in out:
        assert r["similarity"] == pytest.approx(raw[r["id"]] / raw["a"])
    assert out[0]["similarity"] == 1.0
    assert all(r["engines"] == ["semantic"] for r in out)


def test_scores_accumulate_across_engines():
    out = rrf_fuse(
        [("semantic", [rec("x"), rec("y")]), ("keyword", [rec("y"), rec("z")])], k=60
    )
    raw = {"x": 1 / 61, "y": 1 / 62 + 1 / 61, "z": 1 / 62}
    best = raw["y"]

    assert ids(out) == ["y", "x", "z"]
    got = by_id(out)
    for rid, score in raw.items():
        assert got[rid]["similarity"] == pytest.approx(score / best)
    assert got["y"]["engines"] == ["semantic", "keyword"]
    assert got["x"]["engines"] == ["semantic"]
    assert got["z"]["engines"] == ["keyword"]


def test_document_found_by_all_three_engines_lists_them_in_input_order():
    out = rrf_fuse(
        [
            ("semantic", [rec("other"), rec("d")]),
            ("keyword", [rec("d")]),
            ("tags", [rec("d")]),
        ]
    )
    assert out[0]["id"] == "d"
    assert out[0]["engines"] == ["semantic", "keyword", "tags"]


def test_engines_follow_input_order_not_rank():
    out = rrf_fuse([("tags", [rec("a"), rec("b")]), ("semantic", [rec("b")])])
    assert by_id(out)["b"]["engines"] == ["tags", "semantic"]


@pytest.mark.parametrize("k", [0, 1, 60, 1000])
def test_normalised_scores_lie_in_zero_one_with_the_best_exactly_one(k):
    out = rrf_fuse(
        [("a", [rec(i) for i in "pqrst"]), ("b", [rec(i) for i in "tsrqp"]), ("c", [rec("q")])], k=k
    )
    sims = [r["similarity"] for r in out]
    assert max(sims) == 1.0
    assert all(0.0 < s <= 1.0 for s in sims)
    assert sims == sorted(sims, reverse=True)


def test_ties_keep_first_seen_order():
    # Both documents score 1/(k+1); Python's sort is stable, so input order decides.
    assert ids(rrf_fuse([("a", [rec("x")]), ("b", [rec("y")])])) == ["x", "y"]
    assert ids(rrf_fuse([("b", [rec("y")]), ("a", [rec("x")])])) == ["y", "x"]


def test_larger_k_flattens_the_advantage_of_top_ranks():
    lists = [("semantic", [rec(str(i)) for i in range(10)])]
    sharp = by_id(rrf_fuse(lists, k=1))["9"]["similarity"]
    flat = by_id(rrf_fuse(lists, k=600))["9"]["similarity"]
    assert sharp == pytest.approx(2 / 11)
    assert flat == pytest.approx(601 / 610)
    assert flat > sharp


@pytest.mark.parametrize(
    "engine_lists",
    [[], [("semantic", [])], [("semantic", []), ("keyword", [])], [("semantic", None)]],
)
def test_empty_input_gives_empty_output(engine_lists):
    assert rrf_fuse(engine_lists) == []


def test_empty_engine_lists_are_ignored():
    out = rrf_fuse([("semantic", []), ("keyword", [rec("a")])])
    assert ids(out) == ["a"]
    assert out[0]["engines"] == ["keyword"]
    assert out[0]["similarity"] == 1.0


def test_inputs_are_not_mutated_and_outputs_are_copies():
    semantic = [rec("a", content="A", metadata={"tags": ["t"]}, similarity=0.83)]
    keyword = [rec("a", content="A", metadata={"tags": ["t"]}, similarity=7.0), rec("b")]
    before = copy.deepcopy((semantic, keyword))

    out = rrf_fuse([("semantic", semantic), ("keyword", keyword)])

    assert (semantic, keyword) == before
    assert "engines" not in semantic[0]
    assert all(r is not semantic[0] and r is not keyword[0] for r in out)


def test_first_occurrence_supplies_the_fields_and_similarity_is_replaced():
    out = rrf_fuse(
        [
            ("semantic", [rec("a", content="from semantic", similarity=0.9)]),
            ("keyword", [rec("a", content="from keyword", similarity=12.5)]),
        ]
    )
    assert out[0]["content"] == "from semantic"
    assert out[0]["similarity"] == 1.0  # fused score, not the engine's own scale


def test_input_engines_key_is_overwritten_on_the_copy():
    out = rrf_fuse([("keyword", [rec("a", engines=["stale"])])])
    assert out[0]["engines"] == ["keyword"]


def test_random_lists_keep_the_invariants():
    rng = random.Random(1234)
    universe = [f"m{i}" for i in range(40)]
    for _ in range(200):
        lists = [
            (name, [rec(i) for i in rng.sample(universe, rng.randint(0, 15))])
            for name in ("semantic", "keyword", "tags")
        ]
        out = rrf_fuse(lists, k=rng.choice([0, 10, 60]))
        expected_ids = {r["id"] for _, lst in lists for r in lst}

        assert {r["id"] for r in out} == expected_ids
        assert len(out) == len(expected_ids)
        if out:
            sims = [r["similarity"] for r in out]
            assert sims == sorted(sims, reverse=True)
            assert max(sims) == 1.0 and min(sims) > 0
        for r in out:
            found_in = [name for name, lst in lists if any(x["id"] == r["id"] for x in lst)]
            assert r["engines"] == found_in
