"""Dense index: FAISS IndexFlatIP over L2-normalized embeddings, so inner product == cosine
similarity (an un-normalized index would make the "cosine" ranking meaningless). IndexFlatIP is
exact, brute-force search -- the honest choice at this corpus's scale (hundreds of chunks); an ANN
index would trade accuracy for a speed advantage this project doesn't need.
"""
import pickle

import faiss
import numpy as np


class DenseIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids: list[str] = []

    def build(self, embeddings: np.ndarray, chunk_ids: list[str]):
        assert embeddings.shape[0] == len(chunk_ids)
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        self.chunk_ids = list(chunk_ids)

    def search(self, query_vec: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        scores, idx = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [(self.chunk_ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

    def save(self, path):
        faiss.write_index(self.index, str(path) + ".faiss")
        with open(str(path) + ".ids.pkl", "wb") as f:
            pickle.dump(self.chunk_ids, f)

    @classmethod
    def load(cls, path):
        index = faiss.read_index(str(path) + ".faiss")
        obj = cls(index.d)
        obj.index = index
        with open(str(path) + ".ids.pkl", "rb") as f:
            obj.chunk_ids = pickle.load(f)
        return obj
