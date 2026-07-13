"""Reciprocal Rank Fusion: combine ranked lists from BM25 and dense search using rank POSITION,
not raw score. BM25 scores are unbounded and dense scores are cosine similarity in [-1, 1] -- the
two scales aren't comparable, so fusing on raw scores would let whichever retriever happens to
produce larger numbers dominate. RRF sidesteps that entirely by scoring only on where a chunk
lands in each list.
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """ranked_lists: each a list of (chunk_id, score) already sorted best-first.
    k=60 is the standard damping constant from the original RRF paper (Cormack et al., 2009) --
    large enough that the exact rank of a low-ranked hit barely matters, so fusion isn't jumpy."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _score) in enumerate(ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])
