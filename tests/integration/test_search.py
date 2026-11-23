"""search_with_graph on a small synthetic corpus, with the real BGE-M3 model.

Each engine gets a probe that only it can answer. A spy around rrf_fuse records
the ranked list every engine handed to fusion, so the tests check both what
each engine found and what the fused result says (`engines`).

Engines: semantic (pgvector cosine, chunks resolved to parents), keyword
(PostgreSQL FTS, OR semantics), tags (query vs tag_embeddings catalog),
tag_exact (query equals a metadata tag), metadata (query is a substring of the
metadata JSON). The corpus is invented and neutral.
"""
import re
from datetime import datetime, timezone

import pytest

import src.storage.postgres_enhanced as pe
from integration_helpers import open_storage, store_like_server
from src.core import chunker

SOUP = """## Red lentil soup
A thick, cheap weeknight soup made from red lentils, carrots and onions. It is ready in about forty minutes, needs one pot, and gives six generous bowls.

## Ingredients
Two cups of red lentils rinsed until the water runs clear, one large onion, two carrots, three cloves of garlic, a tin of chopped tomatoes, a teaspoon of ground cumin, half a teaspoon of smoked paprika, a pinch of chilli flakes, one and a half litres of vegetable stock, two tablespoons of olive oil, salt, black pepper and the juice of half a lemon. A handful of chopped parsley is optional.

## Method
Soften the chopped onion and carrots in the olive oil over a medium flame for ten minutes, until the onion is translucent. Add the garlic, cumin, paprika and chilli and stir for one minute. Tip in the lentils, tomatoes and stock, bring everything to a boil, then lower the flame and simmer for twenty five minutes, stirring now and then so nothing catches on the bottom of the pot. When the lentils have collapsed, blend half of the pot for a creamy texture and leave the rest chunky. Season with salt and pepper and finish with the lemon juice just before serving.

## Variations
Swap the carrots for diced sweet potato, stir in a handful of spinach at the end, or top each bowl with a spoon of yoghurt and toasted seeds. A spoon of harissa instead of chilli flakes gives a smokier, warmer bowl.

## Storing leftovers
Cool the pot quickly, then ladle portions into airtight boxes and label them with the date. They keep three days in the fridge and up to three months in the freezer. Thaw overnight and reheat gently until steaming, adding a splash of water if the soup has thickened.
"""

# name -> (content, metadata). Order matters: it is the insertion order, and the
# temporal trigger links each memory to up to five earlier ones.
CORPUS = {
    "lisbon": ("Spent four days in Lisbon: rode tram 28 up to the castle, ate custard tarts in Belém "
               "and watched the sunset from a viewpoint over the river.",
               {"tags": ["lisbon", "travel"]}),
    "falcon": ("At dawn a small falcon hovered over the meadow, wings beating fast, then dropped onto "
               "a mouse; we watched it through binoculars from behind the hedge.", {}),
    "kestrel": ("Rehearsed the PostgreSQL major-version upgrade on a staging copy: pg_upgrade took eleven "
                "minutes, then we rebuilt indexes and compared row counts table by table.",
                {"tags": ["Project-Kestrel"]}),
    "cat_vet": ("Took the cat to the vet clinic for her annual vaccination; she weighed 4.8 kg, got a "
                "booster shot and sulked in the carrier all the way home.", {}),
    "bike": ("Repaired a flat bicycle tyre: found the thorn, patched the inner tube and pumped it back "
             "up to four bar.", {}),
    "run": ("Утром пробежала пять километров вдоль реки, потом растяжка и холодный душ.", {}),
    "vorbulax": ("The plumber replaced the vorbulax fitting under the sink and left it under pressure "
                 "overnight to check for drips.", {}),
    "umbrella": ("Bought a new umbrella after the old one turned inside out in the wind.",
                 {"source": "notes-from-lisbon.txt"}),
    "soup": (SOUP, {"tags": ["cooking"]}),
    "tap": ("Починили протекающий кран на кухне, мастер поменял прокладку и проверил давление.", {}),
    "lighthouse": ("Читали вслух главу из романа про маяк и смотрителя, который вёл дневник погоды.",
                   {"tags": ["книги"]}),
    "bread": ("Испекли хлеб на закваске, корочка получилась тёмной и хрустящей.", {"tags": ["выпечка"]}),
    "tomatoes": ("Planted tomato seedlings on the balcony and rigged a drip line from a recycled bottle.",
                 {"tags": ["garden"]}),
    "budget": ("Quarterly budget review: moved the cloud storage line from monthly to annual billing to "
               "save about twelve percent.", {"tags": ["finance"]}),
    "ukulele": ("Learned three chords on the ukulele and played a slow version of an old folk song.",
                {"tags": ["music"]}),
    "museum": ("Сходили в музей современного искусства, долго стояли у большой синей инсталляции.", {}),
}


class Corpus:
    def __init__(self, storage, ids, run):
        self.storage, self.ids, self.run = storage, ids, run
        self.name_of = {v: k for k, v in ids.items()}

    def search(self, monkeypatch, query, **kwargs):
        """Run search_with_graph; return (results, {engine: [names in rank order]})."""
        captured = {}
        real = pe.rrf_fuse

        def spy(engine_lists, k=60):
            captured.update({name: [r["id"] for r in lst] for name, lst in engine_lists})
            return real(engine_lists, k=k)

        monkeypatch.setattr(pe, "rrf_fuse", spy)
        kwargs.setdefault("limit", 10)
        kwargs.setdefault("use_graph", False)
        results = self.run(self.storage.search_with_graph(query, **kwargs))
        monkeypatch.setattr(pe, "rrf_fuse", real)
        named = {engine: [self.name_of.get(i, i) for i in ids] for engine, ids in captured.items()}
        return results, named

    def names(self, results):
        return [self.name_of.get(r["id"], r["id"]) for r in results]

    def semantic_rank(self, query):
        """Pure cosine ranking of parents/standalone memories (chunks folded into parents)."""
        async def go():
            import numpy as np
            vec = np.array(await self.storage.encode_text(query), dtype=np.float32)
            async with self.storage.pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT coalesce(parent_id, id) AS pid, max(1 - (embedding <=> $1)) AS sim
                       FROM memories WHERE embedding IS NOT NULL GROUP BY 1 ORDER BY 2 DESC""", vec)
            return [(self.name_of[r["pid"]], float(r["sim"])) for r in rows]
        return self.run(go())


@pytest.fixture(scope="module")
def corpus(server, encoder, run):
    dsn = run(server.create_database("search"))
    storage = run(open_storage(dsn, encoder))  # AUTO_MIGRATE=true: the server's own startup path
    ids = {}
    for name, (content, metadata) in CORPUS.items():
        ids[name] = run(store_like_server(storage, content, metadata))
    yield Corpus(storage, ids, run)
    run(storage.close())
    run(server.drop_database(dsn))


def fetch(corpus, sql, *args):
    async def go():
        async with corpus.storage.pool.acquire() as conn:
            return await conn.fetch(sql, *args)
    return corpus.run(go())


# ------------------------------------------------------------------ the corpus itself
def test_long_memory_is_stored_as_parent_plus_embedded_chunks(corpus):
    soup = corpus.ids["soup"]
    expected = len(chunker.chunk_text(SOUP))
    parent = fetch(corpus, "SELECT embedding IS NULL AS no_vec FROM memories WHERE id = $1", soup)
    chunks = fetch(corpus, "SELECT id, vector_dims(embedding) AS d FROM memories WHERE parent_id = $1", soup)
    assert expected >= 3
    assert parent[0]["no_vec"]
    assert len(chunks) == expected and all(c["d"] == 1024 for c in chunks)
    assert {c["id"] for c in chunks} == {f"{soup}_c{i}" for i in range(expected)}


def test_tag_catalog_is_built_from_metadata_tags_lowercased(corpus):
    rows = fetch(corpus, "SELECT tag, memory_count, vector_dims(embedding) AS d FROM tag_embeddings")
    expected = {t.lower() for _, meta in CORPUS.values() for t in meta.get("tags", [])}
    assert {r["tag"] for r in rows} == expected
    assert all(r["memory_count"] == 1 and r["d"] == 1024 for r in rows)


def test_time_bucket_follows_the_local_clock(corpus):
    def bucket(hour):
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "day"
        if 17 <= hour < 22:
            return "evening"
        return "night"

    rows = fetch(corpus, "SELECT created_at, sashf->>'time_bucket' AS b FROM memories WHERE parent_id IS NULL")
    assert len(rows) == len(CORPUS)
    for r in rows:
        assert r["b"] == bucket(r["created_at"].astimezone().hour)  # machine local time


# ---------------------------------------------------------------- one probe per engine
def test_keyword_engine_finds_a_rare_exact_token(corpus, monkeypatch):
    results, lists = corpus.search(monkeypatch, "vorbulax")

    assert lists["keyword"] == ["vorbulax"]              # FTS found exactly one memory
    assert "tags" not in lists                            # precondition: no tag looks like it
    assert corpus.names(results)[0] == "vorbulax"
    assert results[0]["engines"][:2] == ["semantic", "keyword"]
    assert not any("keyword" in r.get("engines", []) for r in results[1:])


@pytest.mark.parametrize(
    "query, target",
    [
        ("кошку возили к ветеринару на прививку", "cat_vet"),  # Russian query, English memory
        ("a bird of prey hunting a rodent in a field", "falcon"),  # paraphrase, no shared stems
    ],
)
def test_semantic_engine_alone_finds_a_paraphrase(corpus, monkeypatch, query, target):
    results, lists = corpus.search(monkeypatch, query)

    # preconditions: the query shares no word with the corpus and matches no tag
    assert set(lists) == {"semantic"}, f"other engines fired: {lists}"
    assert lists["semantic"][0] == target
    assert corpus.names(results)[0] == target
    assert results[0]["engines"] == ["semantic"]


def test_tag_engine_finds_a_memory_by_a_tag_absent_from_its_text(corpus, monkeypatch):
    query = "project kestrel"
    by_cosine = corpus.semantic_rank(query)
    results, lists = corpus.search(monkeypatch, query)

    # precondition: on text alone the bird memory wins, the tagged one does not
    assert by_cosine[0][0] != "kestrel", by_cosine[:3]
    assert "kestrel" not in CORPUS["kestrel"][0].lower()
    assert lists["tags"][0] == "kestrel"
    assert "kestrel" not in lists.get("keyword", [])
    # the tag engine is what lifts it to the top of the fused list
    assert corpus.names(results)[0] == "kestrel"
    assert "tags" in results[0]["engines"] and "keyword" not in results[0]["engines"]


def test_query_equal_to_a_tag_fires_every_engine_in_input_order(corpus, monkeypatch):
    results, lists = corpus.search(monkeypatch, "lisbon")

    top = results[0]
    assert corpus.names(results)[0] == "lisbon"
    assert top["engines"] == ["semantic", "keyword", "tags", "tag_exact", "metadata"]
    assert lists["tag_exact"] == ["lisbon"]
    assert set(lists["metadata"]) == {"lisbon", "umbrella"}


def test_metadata_only_hit_does_not_outrank_a_direct_match(corpus, monkeypatch):
    # 'umbrella' mentions lisbon only in metadata.source; the old fixed 0.5/0.8
    # scores for metadata hits could put such records above direct matches.
    results, _ = corpus.search(monkeypatch, "lisbon")
    names = corpus.names(results)
    umbrella = results[names.index("umbrella")]

    assert names.index("lisbon") == 0 < names.index("umbrella")
    assert "metadata" in umbrella["engines"]
    assert not {"keyword", "tags", "tag_exact"} & set(umbrella["engines"])
    assert umbrella["similarity"] < results[0]["similarity"]


@pytest.mark.parametrize("query", ["%", "_", "lis_on", "%lisbon%"])
def test_like_wildcards_in_the_query_are_literal(corpus, monkeypatch, query):
    _, lists = corpus.search(monkeypatch, query)
    assert "metadata" not in lists, lists.get("metadata")


def test_chunk_hits_are_returned_as_their_parent(corpus, monkeypatch):
    results, lists = corpus.search(monkeypatch, "how long do the portions keep in the freezer")

    assert corpus.names(results)[0] == "soup"
    assert results[0]["id"] == corpus.ids["soup"]
    assert results[0].get("resolved_from_chunk") is True
    assert lists["semantic"][0] == "soup"
    chunk_like = re.compile(r"_c\d+$")
    assert not any(chunk_like.search(r["id"]) for r in results)
    assert not any(chunk_like.search(i) for engine in lists.values() for i in engine)


# ------------------------------------------------------------------ temporal + graph
def test_temporal_only_query_returns_parents_created_today(corpus, monkeypatch):
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    expected = {r["id"] for r in fetch(
        corpus, "SELECT id FROM memories WHERE parent_id IS NULL AND created_at >= $1", today)}

    results, lists = corpus.search(monkeypatch, "today", limit=50)

    assert lists == {}  # no engine ran: the temporal-only path is a plain date query
    assert {r["id"] for r in results} == expected
    assert len(expected) == len(CORPUS)


def test_graph_neighbours_are_marked_and_get_recency(corpus, monkeypatch):
    # min_similarity 0.99 silences the semantic engine, so only the keyword hit is
    # direct and everything else must come from graph expansion over 'after' links.
    results, lists = corpus.search(monkeypatch, "vorbulax", use_graph=True, min_similarity=0.99)
    links = {r["to_id"] for r in fetch(
        corpus, "SELECT to_id FROM memory_links WHERE from_id = $1 AND link_type = 'after'",
        corpus.ids["vorbulax"])}

    assert "semantic" not in lists
    assert corpus.names(results)[0] == "vorbulax"
    graph = [r for r in results if r.get("source") == "graph"]
    assert graph, "expected graph-expanded neighbours"
    assert links <= {r["id"] for r in results}
    for r in graph:
        assert r["created_at"] is not None and r["days_old"] == 0
        assert r["recency_boost"] == pytest.approx(1.0)
        assert r["similarity"] < results[0]["similarity"]


def test_graph_walks_links_in_both_directions(corpus, monkeypatch):
    # 'after' links are stored newer -> earlier. The first memory of the corpus has no
    # earlier neighbour, so anything the graph adds to it must be reached by walking a
    # link backwards, towards what came after it. 'falcon' was stored right after
    # 'lisbon' and has no tags and no words in common with the query.
    results, lists = corpus.search(monkeypatch, "Belém", use_graph=True, min_similarity=0.99)

    assert corpus.names(results)[0] == "lisbon"
    assert not any(corpus.ids["falcon"] in ids for ids in lists.values())
    graph = {r["id"] for r in results if r.get("source") == "graph"}
    assert corpus.ids["falcon"] in graph


def test_graph_expansion_does_not_displace_the_top_direct_hit(corpus, monkeypatch):
    without, _ = corpus.search(monkeypatch, "кошку возили к ветеринару на прививку")
    with_graph, _ = corpus.search(monkeypatch, "кошку возили к ветеринару на прививку", use_graph=True)
    assert with_graph[0]["id"] == without[0]["id"] == corpus.ids["cat_vet"]


# -------------------------------------------------------------------- known defects
@pytest.mark.xfail(
    strict=True,
    reason=(
        "RANKING DEFECT config/settings.py TAG_MIN_SIMILARITY=0.5 + tag engine in "
        "src/storage/postgres_enhanced.py (search_with_graph, STEP 2.1): with correct BGE-M3 "
        "vectors, short unrelated strings routinely score 0.50-0.53 against tags (here "
        "'fixing a punctured wheel' vs tag 'cooking' = 0.52). The tag engine then puts the "
        "cooking memory at rank 1 of its list, RRF counts that as a full vote, and the soup "
        "recipe outranks the bicycle memory that the semantic engine ranked first."
    ),
)
def test_an_unrelated_tag_does_not_outrank_the_semantic_hit(corpus, monkeypatch):
    results, lists = corpus.search(monkeypatch, "fixing a punctured wheel")
    assert lists["semantic"][0] == "bike"  # the semantic engine gets it right
    assert corpus.names(results)[0] == "bike"
