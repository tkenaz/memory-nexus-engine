"""Reciprocal Rank Fusion (RRF) for the search engines in EnhancedPostgresStorage.

The engines score on incomparable scales: ts_rank_cd for keywords, cosine for
dense vectors, query-to-tag cosine for the tag catalog. RRF ignores the scores and
fuses ranks only:

    score(doc) = sum over engines that returned doc of 1 / (k + rank)

with rank starting at 1. Reference: Cormack, Clarke, Buettcher (2009),
"Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods".
"""

from typing import Dict, List, Sequence, Tuple

RankedList = Sequence[Dict]
EngineList = Tuple[str, RankedList]


def rrf_fuse(engine_lists: Sequence[EngineList], k: int = 60) -> List[Dict]:
    """Fuse ranked result lists from several engines.

    Args:
        engine_lists: (engine_name, ranked_records) pairs. Each record is a dict
            with an ``id`` key; order within a list is the engine's ranking.
            Empty lists are ignored.
        k: RRF constant. Larger k flattens the advantage of top ranks.

    Returns:
        One record per distinct id, sorted by fused score (descending). Each
        record is a shallow copy of the first occurrence, with two added keys:
        ``similarity`` - fused score normalised to (0, 1] by the best score, so
        downstream scoring keeps its scale; ``engines`` - names of the engines
        that returned it, in input order.
    """
    scores: Dict[str, float] = {}
    records: Dict[str, Dict] = {}

    for engine, ranked in engine_lists:
        if not ranked:
            continue
        for rank, record in enumerate(ranked, start=1):
            rid = record["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank)
            if rid in records:
                if engine not in records[rid]["engines"]:
                    records[rid]["engines"].append(engine)
            else:
                fused = dict(record)
                fused["engines"] = [engine]
                records[rid] = fused

    if not scores:
        return []

    best = max(scores.values())
    results = []
    for rid, score in scores.items():
        record = records[rid]
        record["similarity"] = score / best
        results.append(record)
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results
