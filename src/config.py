"""Paths + tunable constants, one place so nothing downstream retypes them from memory. Grows as
new modules need new constants, rather than each module hardcoding its own copy of the same value.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- chunking ------------------------------------------------------------------
CHUNK_SIZE = 400     # tokens
CHUNK_OVERLAP = 50   # tokens

# --- retrieval -------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
