"""Cross-encoder reranker: scores (query, chunk_text) pairs jointly through one transformer pass,
rather than comparing two independently-computed embeddings. Much more precise than bi-encoder
cosine similarity, but too slow to run over an entire corpus -- so it only ever sees the fused
top-k candidates, not the full index.
"""
from sentence_transformers import CrossEncoder

from src import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(query: str, candidates: list[tuple[str, str]], top_k: int = config.FINAL_K) -> list[tuple[str, float]]:
    """candidates: list of (chunk_id, chunk_text). Returns top_k (chunk_id, score), best first."""
    if not candidates:
        return []
    pairs = [(query, text) for _, text in candidates]
    scores = get_model().predict(pairs)
    ranked = sorted(zip([cid for cid, _ in candidates], scores), key=lambda x: -x[1])
    return [(cid, float(s)) for cid, s in ranked[:top_k]]
