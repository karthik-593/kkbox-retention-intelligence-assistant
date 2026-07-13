"""Paths + tunable constants, one place so nothing downstream retypes them from memory. Grows as
new modules need new constants, rather than each module hardcoding its own copy of the same value.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- data paths --------------------------------------------------------------
CORPUS_DIR = REPO_ROOT / "data" / "raw" / "corpus"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"      # chunks.jsonl, dense/sparse indices

# filename -> doc_type (section 5 of the build plan). One manifest, so build_index.py doesn't
# guess doc_type from a filename pattern -- a new corpus doc means adding one line here.
CORPUS_MANIFEST = {
    "playbook_p1_auto_renew_reenablement.pdf": "playbook",
    "playbook_p2_plan_commitment_upsell.pdf": "playbook",
    "playbook_p3_discount_expiry_price_sensitivity.pdf": "playbook",
    "playbook_p4_dormant_winback.pdf": "playbook",
    "playbook_p5_low_engagement_activation.pdf": "playbook",
    "playbook_p6_recommendation_quality.pdf": "playbook",
    "policy_subscription_renewal.pdf": "policy",
    "campaign_manual.pdf": "campaign_manual",
    "support_faq.pdf": "faq",
    "historical_retention_notes.pdf": "case_notes",
    "offer_eligibility_rules.csv": "eligibility_rules",
}

# --- chunking ------------------------------------------------------------------
CHUNK_SIZE = 400     # tokens
CHUNK_OVERLAP = 50   # tokens

# --- retrieval -------------------------------------------------------------
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FUSE_K = 20   # candidate pool fed into RRF fusion / the reranker, before cutting to FINAL_K
FINAL_K = 5   # final number of chunks returned to the caller
