"""Wires embed + dense + sparse + fusion + rerank into one retriever. This is pure orchestration --
each piece it calls is independently testable/swappable (that's the point of the ablation table in
eval/retrieval_eval.py), this module just holds the pipeline order the plan settled on:
BM25 + dense -> RRF fuse -> cross-encoder rerank.
"""
from pathlib import Path

from src import config
from src.ingest.chunker import read_jsonl
from src.retrieval import embed
from src.retrieval.dense import DenseIndex
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.rerank import rerank as ce_rerank
from src.retrieval.sparse import SparseIndex


class HybridRetriever:
    def __init__(self, index_dir: Path = config.PROCESSED_DIR):
        index_dir = Path(index_dir)
        self.chunks = {c["chunk_id"]: c for c in read_jsonl(index_dir / "chunks.jsonl")}
        self.dense = DenseIndex.load(index_dir / "dense_index")
        self.sparse = SparseIndex.load(index_dir / "sparse_index")

    def retrieve(
        self,
        query: str,
        fuse_k: int = config.FUSE_K,
        final_k: int = config.FINAL_K,
        use_rerank: bool = True,
    ) -> list[dict]:
        """Returns final_k chunk dicts (chunk_id, text, source, doc_type, page, score), best first."""
        dense_hits = self.dense.search(embed.embed_query(query), k=fuse_k)
        sparse_hits = self.sparse.search(query, k=fuse_k)
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits])[:fuse_k]

        if use_rerank:
            candidates = [(cid, self.chunks[cid]["text"]) for cid, _ in fused]
            ranked = ce_rerank(query, candidates, top_k=final_k)
        else:
            ranked = fused[:final_k]

        out = []
        for cid, score in ranked:
            rec = dict(self.chunks[cid])
            rec["score"] = score
            out.append(rec)
        return out
