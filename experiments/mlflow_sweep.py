"""MLflow retrieval sweep (Week 3, Days 15-16): chunk_size x overlap x embedding_model x top_k x
rerank, each combination logged as one MLflow run with Recall@k / MRR / latency. Bounded to
retrieval only -- generation isn't swept, on purpose (a much bigger, noisier search space that
wouldn't fit "sweep it and log it" the same way).

Builds indices in memory per (chunk_size, overlap, embedding_model) combo; never touches
data/processed/ (the live app's index), so this can be re-run any time without rebuilding the
production index.

Run: python experiments/mlflow_sweep.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import numpy as np

from src import config
from src.ingest.chunker import chunk_records, read_jsonl
from src.ingest.loaders import load_csv, load_pdf
from src.retrieval import embed
from src.retrieval.dense import DenseIndex
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import rerank as ce_rerank
from src.retrieval.sparse import SparseIndex

# sweep grid -- these define what THIS experiment scans over, not the app's runtime defaults, so
# they stay local rather than moving to config.py (config.CHUNK_SIZE/CHUNK_OVERLAP/FUSE_K/
# EMBEDDING_MODELS hold the single-value defaults everything else uses).
CHUNK_SIZES = [200, 400, 600]
OVERLAPS = [25, 50]
TOP_KS = [5, 10, 20]
RERANK_OPTIONS = [False, True]
FUSE_K = config.FUSE_K


def load_corpus_records() -> list[dict]:
    records = []
    for fname, doc_type in config.CORPUS_MANIFEST.items():
        path = config.CORPUS_DIR / fname
        if path.suffix == ".pdf":
            records += load_pdf(path, doc_type)
        elif path.suffix == ".csv":
            records += load_csv(path, doc_type)
    return records


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    hit = len(set(ranked_ids[:k]) & gold_ids)
    return hit / len(gold_ids) if gold_ids else 0.0


def mrr_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


def evaluate_config(chunk_lookup, dense, sparse, queries, embedding_model, top_k, use_rerank):
    dense.search(embed.embed_query(queries[0]["query"], model_name=embedding_model), k=FUSE_K)  # warm-up

    recalls, mrrs, latencies = [], [], []
    for q in queries:
        gold = set(q["gold_chunk_ids"])
        t0 = time.perf_counter()
        d = dense.search(embed.embed_query(q["query"], model_name=embedding_model), k=FUSE_K)
        s = sparse.search(q["query"], k=FUSE_K)
        fused = reciprocal_rank_fusion([d, s])[:FUSE_K]
        if use_rerank:
            candidates = [(cid, chunk_lookup[cid]["text"]) for cid, _ in fused]
            ranked_ids = [cid for cid, _ in ce_rerank(q["query"], candidates, top_k=top_k)]
        else:
            ranked_ids = [cid for cid, _ in fused[:top_k]]
        latencies.append((time.perf_counter() - t0) * 1000)
        recalls.append(recall_at_k(ranked_ids, gold, top_k))
        mrrs.append(mrr_at_k(ranked_ids, gold, top_k))

    n = len(queries)
    return sum(recalls) / n, sum(mrrs) / n, sum(latencies) / n


def main():
    mlflow.set_tracking_uri(f"sqlite:///{(config.REPO_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("retrieval-sweep")

    records = load_corpus_records()
    queries = read_jsonl(config.QUERIES_PATH)
    print(f"corpus: {len(records)} page/row records, {len(queries)} eval queries\n")

    results = []
    run_n = 0
    total = len(CHUNK_SIZES) * len(OVERLAPS) * len(config.EMBEDDING_MODELS) * len(TOP_KS) * len(RERANK_OPTIONS)

    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            chunks = chunk_records(records, chunk_size=chunk_size, overlap=overlap)
            chunk_lookup = {c["chunk_id"]: c for c in chunks}
            texts = [c["text"] for c in chunks]
            chunk_ids = [c["chunk_id"] for c in chunks]

            sparse = SparseIndex()
            sparse.build(texts, chunk_ids)

            for embedding_key, model_name in config.EMBEDDING_MODELS.items():
                vecs = np.asarray(embed.embed_passages(texts, model_name=model_name))
                dense = DenseIndex(dim=vecs.shape[1])
                dense.build(vecs, chunk_ids)

                for top_k in TOP_KS:
                    for use_rerank in RERANK_OPTIONS:
                        recall, mrr, latency = evaluate_config(
                            chunk_lookup, dense, sparse, queries, model_name, top_k, use_rerank
                        )
                        with mlflow.start_run(
                            run_name=f"cs{chunk_size}_ov{overlap}_{embedding_key}_k{top_k}_rr{use_rerank}"
                        ):
                            mlflow.log_params({
                                "chunk_size": chunk_size, "overlap": overlap,
                                "embedding_model": embedding_key, "top_k": top_k,
                                "rerank": use_rerank, "n_chunks": len(chunks),
                            })
                            mlflow.log_metrics({
                                "recall_at_k": recall, "mrr": mrr, "latency_ms": latency,
                            })
                        run_n += 1
                        print(f"[{run_n}/{total}] cs={chunk_size:<3} ov={overlap:<2} emb={embedding_key:<9} "
                              f"k={top_k:<2} rerank={str(use_rerank):<5} -> "
                              f"recall={recall:.3f} mrr={mrr:.3f} lat={latency:6.1f}ms")
                        results.append({
                            "chunk_size": chunk_size, "overlap": overlap, "embedding_model": embedding_key,
                            "top_k": top_k, "rerank": use_rerank, "recall": recall, "mrr": mrr,
                            "latency_ms": latency,
                        })

    at5 = [r for r in results if r["top_k"] == 5]
    best = max(at5, key=lambda r: (round(r["recall"], 4), round(r["mrr"], 4)))
    print(f"\nBest at k=5 (by recall, tie-broken by MRR):")
    print(f"  chunk_size={best['chunk_size']} overlap={best['overlap']} "
          f"embedding={best['embedding_model']} rerank={best['rerank']} "
          f"-> recall={best['recall']:.3f} mrr={best['mrr']:.3f} latency={best['latency_ms']:.1f}ms")
    print(f"\n{run_n} runs logged to MLflow (sqlite:///mlflow.db, experiment 'retrieval-sweep'). "
          f"Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to browse.")


if __name__ == "__main__":
    main()
