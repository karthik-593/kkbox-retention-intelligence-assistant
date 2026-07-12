"""One-command pipeline: ingest corpus -> chunk -> embed -> build dense + sparse indices.

Run: python scripts/build_index.py [--chunk-size 400] [--overlap 50]
Writes data/processed/chunks.jsonl, dense_index.{faiss,ids.pkl}, sparse_index.bm25.pkl
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `src.*` imports resolve

import numpy as np

from src import config
from src.ingest.chunker import chunk_records, write_jsonl
from src.ingest.loaders import load_csv, load_pdf
from src.retrieval import embed
from src.retrieval.dense import DenseIndex
from src.retrieval.sparse import SparseIndex


def load_corpus() -> list[dict]:
    records = []
    for fname, doc_type in config.CORPUS_MANIFEST.items():
        path = config.CORPUS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run scripts/generate_corpus.py first")
        if path.suffix == ".pdf":
            records += load_pdf(path, doc_type)
        elif path.suffix == ".csv":
            records += load_csv(path, doc_type)
    return records


def main(chunk_size: int, overlap: int):
    t0 = time.time()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = load_corpus()
    print(f"loaded {len(records)} page/row records from {len(config.CORPUS_MANIFEST)} files")

    chunks = chunk_records(records, chunk_size=chunk_size, overlap=overlap)
    write_jsonl(chunks, config.PROCESSED_DIR / "chunks.jsonl")
    print(f"chunked into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    passage_vecs = embed.embed_passages(texts)
    dense = DenseIndex(dim=passage_vecs.shape[1])
    dense.build(np.asarray(passage_vecs), chunk_ids)
    dense.save(config.PROCESSED_DIR / "dense_index")
    print(f"built dense index: {passage_vecs.shape[0]} vectors, dim={passage_vecs.shape[1]}")

    sparse = SparseIndex()
    sparse.build(texts, chunk_ids)
    sparse.save(config.PROCESSED_DIR / "sparse_index")
    print("built sparse (BM25) index")

    print(f"done in {time.time() - t0:.1f}s -> {config.PROCESSED_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = ap.parse_args()
    main(args.chunk_size, args.overlap)
