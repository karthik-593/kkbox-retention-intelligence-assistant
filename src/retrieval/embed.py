"""Embedding wrapper around bge-small. bge-small-en-v1.5 was trained asymmetrically: queries need
an instruction prefix, passages do not. Omitting the query prefix is a silent quality killer -- no
error, just a worse ranking, since the query embedding then lands outside the subspace the model
was tuned to match against.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_passages(texts: list[str]) -> np.ndarray:
    return get_model().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


def embed_query(text: str) -> np.ndarray:
    return get_model().encode(QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False)
