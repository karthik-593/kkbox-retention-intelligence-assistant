"""Sparse retrieval: BM25Okapi over lowercased word/underscore tokens. A deliberately simple
tokenizer -- BM25's value in this hybrid is exact keyword recall (feature names like
`is_auto_renew`, offer IDs like `OFR-301`) that a dense embedding can blur, not linguistic nuance.
"""
import pickle
import re

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class SparseIndex:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []

    def build(self, texts: list[str], chunk_ids: list[str]):
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.chunk_ids = list(chunk_ids)

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: -x[1])[:k]
        return [(cid, float(s)) for cid, s in ranked if s > 0]

    def save(self, path):
        with open(str(path) + ".bm25.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(str(path) + ".bm25.pkl", "rb") as f:
            d = pickle.load(f)
        obj.bm25, obj.chunk_ids = d["bm25"], d["chunk_ids"]
        return obj
