"""Retrieval ablation: BM25 only vs Dense only vs Hybrid (RRF) vs Hybrid + CrossEncoder rerank,
all measured at the same k, on the same hand-labeled queries, with latency logged consistently
(same machine, warm model, averaged over the eval set). This isolation is what lets each retrieval
choice (why BM25, why hybrid, why rerank) be defended with a number instead of an adjective.

Run: python eval/retrieval_eval.py
Reads data/eval/queries.jsonl + the index built by scripts/build_index.py.
Writes data/eval/ablation_table.md
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.ingest.chunker import read_jsonl
from src.retrieval import embed
from src.retrieval.dense import DenseIndex
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import rerank as ce_rerank
from src.retrieval.sparse import SparseIndex

K = config.FINAL_K
FUSE_K = config.FUSE_K  # candidate pool fed into RRF / the reranker, before cutting to K


def recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    hit = len(set(ranked_ids[:k]) & gold_ids)
    return hit / len(gold_ids) if gold_ids else 0.0


def mrr_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    for rank, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


def run_system(name, fn, queries):
    """fn(query) -> list[chunk_id], best first. Times each call, warms up once first."""
    fn(queries[0]["query"])  # warm-up: model already loaded module-wide, this also warms lazy caches

    recalls, mrrs, latencies = [], [], []
    for q in queries:
        gold = set(q["gold_chunk_ids"])
        t0 = time.perf_counter()
        ranked_ids = fn(q["query"])
        latencies.append((time.perf_counter() - t0) * 1000)
        recalls.append(recall_at_k(ranked_ids, gold, K))
        mrrs.append(mrr_at_k(ranked_ids, gold, K))

    n = len(queries)
    return {
        "system": name,
        "recall@5": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "latency_ms": sum(latencies) / n,
    }


def main():
    queries = read_jsonl(config.QUERIES_PATH)
    chunks = {c["chunk_id"]: c for c in read_jsonl(config.PROCESSED_DIR / "chunks.jsonl")}
    dense = DenseIndex.load(config.PROCESSED_DIR / "dense_index")
    sparse = SparseIndex.load(config.PROCESSED_DIR / "sparse_index")

    def bm25_only(query):
        return [cid for cid, _ in sparse.search(query, k=K)]

    def dense_only(query):
        return [cid for cid, _ in dense.search(embed.embed_query(query), k=K)]

    def hybrid(query):
        d = dense.search(embed.embed_query(query), k=FUSE_K)
        s = sparse.search(query, k=FUSE_K)
        return [cid for cid, _ in reciprocal_rank_fusion([d, s])[:K]]

    def hybrid_rerank(query):
        d = dense.search(embed.embed_query(query), k=FUSE_K)
        s = sparse.search(query, k=FUSE_K)
        fused = reciprocal_rank_fusion([d, s])[:FUSE_K]
        candidates = [(cid, chunks[cid]["text"]) for cid, _ in fused]
        return [cid for cid, _ in ce_rerank(query, candidates, top_k=K)]

    systems = [
        ("BM25 only", bm25_only),
        ("Dense only", dense_only),
        ("Hybrid (BM25+Dense+RRF)", hybrid),
        ("Hybrid + CrossEncoder rerank", hybrid_rerank),
    ]

    results = [run_system(name, fn, queries) for name, fn in systems]

    lines = [
        f"Retrieval ablation — {len(queries)} hand-labeled queries, k={K} (fuse_k={FUSE_K})\n",
        f"{'System':<32}{'Recall@5':>10}{'MRR':>8}{'Latency (ms)':>16}",
    ]
    for r in results:
        lines.append(f"{r['system']:<32}{r['recall@5']:>10.3f}{r['mrr']:>8.3f}{r['latency_ms']:>16.1f}")
    table = "\n".join(lines)
    print(table)

    md = ["| System | Recall@5 | MRR | Latency (ms) |", "|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['system']} | {r['recall@5']:.3f} | {r['mrr']:.3f} | {r['latency_ms']:.1f} |")
    out_path = config.EVAL_DIR / "ablation_table.md"
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
