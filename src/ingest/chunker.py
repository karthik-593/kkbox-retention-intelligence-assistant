"""Fixed-size, overlapping chunker, token-counted with tiktoken so chunk_size is comparable across
documents regardless of average word length. Overlap keeps a sentence at a chunk boundary from
losing the context on either side of it.
"""
import json

import tiktoken

from src import config

ENC = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    tokens = ENC.encode(text)
    if not tokens:
        return []
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(ENC.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap  # step back by the overlap, not forward by the full chunk_size
    return chunks


def chunk_records(
    records: list[dict], chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP
) -> list[dict]:
    """records: loader output ({text, source, doc_type, page}).
    Returns chunk dicts with a stable chunk_id (source::page::chunk-index), so eval gold labels
    and citations can reference a chunk deterministically across rebuilds."""
    out = []
    for rec in records:
        for i, piece in enumerate(chunk_text(rec["text"], chunk_size, overlap)):
            out.append({
                "chunk_id": f"{rec['source']}::p{rec['page']}::c{i}",
                "text": piece,
                "source": rec["source"],
                "doc_type": rec["doc_type"],
                "page": rec["page"],
            })
    return out


def write_jsonl(chunks: list[dict], path):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
