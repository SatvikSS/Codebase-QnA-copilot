"""Search chunks by meaning (vectors), by exact words (BM25), or both merged."""

from . import keyword, store
from .embedder import embed

RRF_K = 60        # standard reciprocal-rank-fusion constant
CANDIDATES = 30   # how many results each search contributes before merging


def search_chunks(
    question: str, k: int = 5, mode: str = "hybrid", project: str | None = None
) -> list[dict]:
    """mode: "vector", "keyword" or "hybrid". project limits the search to one project."""
    vector_hits = (
        store.search(embed([question])[0], CANDIDATES, project) if mode != "keyword" else []
    )
    keyword_hits = keyword.search(question, CANDIDATES, project) if mode != "vector" else []

    scores: dict[str, float] = {}
    hits_by_id: dict[str, dict] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, hit in enumerate(hits, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1 / (RRF_K + rank)
            hits_by_id.setdefault(hit["id"], hit)

    best = sorted(scores, key=scores.get, reverse=True)[:k]
    return [{**hits_by_id[chunk]} | {"score": scores[chunk]} for chunk in best]
