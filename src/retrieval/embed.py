"""Embedding wrapper, default model bge-small. bge-small-en-v1.5 was trained asymmetrically:
queries need an instruction prefix, passages do not. Omitting the query prefix is a silent quality
killer -- no error, just a worse ranking, since the query embedding then lands outside the subspace
the model was tuned to match passages against.

Also supports swapping in all-MiniLM-L6-v2 (a symmetric model, no query prefix) via an explicit
model_name argument, purely so experiments/mlflow_sweep.py can compare the two -- every other
caller (build_index.py, hybrid.py, retrieval_eval.py) uses the default and is unaffected.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

from src import config

# query-side instruction prefix per model; "" for symmetric models that don't use one.
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
}

_models: dict[str, SentenceTransformer] = {}


def get_model(model_name: str = config.EMBEDDING_MODEL) -> SentenceTransformer:
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def embed_passages(texts: list[str], model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    return get_model(model_name).encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


def embed_query(text: str, model_name: str = config.EMBEDDING_MODEL) -> np.ndarray:
    prefix = QUERY_PREFIXES.get(model_name, "")
    return get_model(model_name).encode(prefix + text, normalize_embeddings=True, show_progress_bar=False)
