"""Query intent detection: src/core/query_intent.py."""
import pytest

from src.core.query_intent import QueryIntent, QueryIntentDetector


@pytest.fixture
def detector():
    return QueryIntentDetector()


@pytest.mark.parametrize(
    "query, intent, boost",
    [
        ("notes from today", QueryIntent.TEMPORAL, {"recency": 2.0}),
        ("что было вчера", QueryIntent.TEMPORAL, {"recency": 2.0}),
        ("TODAY", QueryIntent.TEMPORAL, {"recency": 2.0}),
        ("memory-nexus roadmap", QueryIntent.PROJECT, {"metadata_match": 1.5}),
        ("a bug report", QueryIntent.TECHNICAL, {"tag_match": 1.5}),
        ("архитектура", QueryIntent.TECHNICAL, {"tag_match": 1.5}),
        ("he was so happy", QueryIntent.EMOTIONAL, {"affect_match": 1.5}),
        ("устал", QueryIntent.EMOTIONAL, {"affect_match": 1.5}),
        ("paid 40 for the tickets", QueryIntent.FACTUAL, {}),
        ("ticket price €", QueryIntent.FACTUAL, {}),
        ("Lisbon trip", QueryIntent.FACTUAL, {}),  # no markers: the default
    ],
)
def test_single_kind_of_marker_sets_intent_and_boost(detector, query, intent, boost):
    got, context = detector.detect_intent(query)
    assert got == intent
    assert context["boost_factors"] == boost


def test_markers_of_different_kinds_give_mixed_without_boosts(detector):
    intent, context = detector.detect_intent("bug yesterday")
    assert intent == QueryIntent.MIXED
    assert context["boost_factors"] == {}
    assert context["entities"]["technical"] == ["bug"]
    assert context["entities"]["temporal"] == ["yesterday"]


def test_context_has_all_entity_buckets(detector):
    _, context = detector.detect_intent("anything")
    assert set(context["entities"]) == {"persons", "projects", "technical", "temporal", "emotional"}
    assert all(v == [] for v in context["entities"].values())
    assert context["filters"] == {}


def test_project_intent_adds_metadata_filter(detector):
    _, context = detector.detect_intent("status of memory nexus")
    assert context["entities"]["projects"] == ["memory nexus"]
    assert context["filters"] == {"metadata": {"project": ["memory nexus"]}}


def test_technical_intent_adds_tag_filter(detector):
    _, context = detector.detect_intent("api")
    assert context["filters"] == {"tags": ["technical", "work"]}


def test_known_persons_are_configurable(detector):
    detector.persons = {"alice"}
    intent, context = detector.detect_intent("What did Alice say")
    assert intent == QueryIntent.PERSON
    assert context["entities"]["persons"] == ["alice"]
    assert context["boost_factors"] == {"actor_match": 1.5}


@pytest.mark.parametrize("intent", list(QueryIntent))
def test_disambiguate_query_is_a_pass_through(detector, intent):
    _, context = detector.detect_intent("memory-nexus bug yesterday")
    assert detector.disambiguate_query("memory-nexus bug yesterday", intent, context) == (
        "memory-nexus bug yesterday"
    )


@pytest.mark.parametrize(
    "query, expected",
    [
        ("bug in the code", QueryIntent.TECHNICAL),
        ("recent notes from today", QueryIntent.TEMPORAL),
        ("устала", QueryIntent.EMOTIONAL),
    ],
)
def test_several_markers_of_one_kind_are_not_mixed(detector, query, expected):
    assert detector.detect_intent(query)[0] == expected


def test_markers_match_whole_words_only(detector):
    intent, context = detector.detect_intent("latest notes")
    assert context["entities"]["technical"] == []
    assert intent == QueryIntent.TEMPORAL
