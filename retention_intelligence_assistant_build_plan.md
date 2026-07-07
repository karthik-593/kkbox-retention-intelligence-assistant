# KKBox Retention Intelligence Assistant — Full Build Plan

**A hybrid-RAG system that talks to your deployed KKBox churn model.**
Not "another RAG chatbot" — a retrieval system conditioned on a live churn model's SHAP explanations.

Timeline: 3 weeks · Each week ends with something demoable · Weeks 1–2 alone = complete, defensible project.

---

## 0. The one-sentence pitch (memorize this)

> A retention assistant for a music-streaming service that answers policy questions with hybrid RAG, answers subscriber-risk questions by calling a deployed KKBox churn model, and — for combined questions — conditions retrieval on the model's SHAP drivers, so the retention playbook it returns matches *why that specific subscriber is at risk*.

The last clause is the whole differentiator. Everything below serves it.

---

## 1. Your real churn model (source of truth)

**Model training features (12) — these are the ONLY things that can appear as SHAP drivers:**

```
is_auto_renew        payment_plan_days    actual_amount_paid   plan_list_price
discount             has_activity_60d     recency_days         active_days_30
secs_30              unq_30               completion_ratio     activity_trend
```

**Monitoring-only features (NOT in training) — never appear as SHAP drivers:**

```
tenure_days          n_prior_cycles
```

**Critical rule:** the driver→playbook mapping keys off the 12 training features only. `tenure_days` / `n_prior_cycles` are used as *segmentation context* in answers and in the logging layer — never as churn drivers. (This distinction is also a clean interview point: you understand what the model can and can't explain.)

---

## 2. Driver → Playbook taxonomy (the audit that keeps the `both` path from breaking)

Every training feature maps to a retention playbook, grouped into 6 playbook clusters so retrieval has distinct, non-duplicate targets:

| SHAP driver (feature) | Churn meaning when it's a top driver | Playbook cluster |
|---|---|---|
| `is_auto_renew` (off) | Will lapse at cycle end, no auto-continuation | **P1 — Auto-renew re-enablement** |
| `payment_plan_days` (short) | Low commitment, short cycles churn more | **P2 — Plan-commitment upsell** |
| `discount` (high) / `actual_amount_paid` ≪ `plan_list_price` | Discount-dependent; churns when the offer ends | **P3 — Discount-expiry / price-sensitivity** |
| `plan_list_price` | Price tier context (pairs with P3) | **P3** |
| `has_activity_60d` (no) | Dormant — not listening at all | **P4 — Dormant win-back** |
| `recency_days` (high) | Lapsing — long gap since last listen | **P4** |
| `active_days_30` (low) | Thin engagement across the month | **P5 — Low-engagement activation** |
| `secs_30` (low) | Low total listening volume | **P5** |
| `unq_30` (low) | Narrow catalog use / declining variety | **P5 (discovery)** |
| `completion_ratio` (low) | High skip rate → recommendation dissatisfaction | **P6 — Recommendation-quality** |
| `activity_trend` (negative) | Downward trajectory, early warning | **P6 (early intervention)** |

**`src/churn/driver_to_query.py`** — the conditioning step, the actual differentiator:
```python
DRIVER_TO_QUERY = {
    "is_auto_renew":      "auto-renew re-enablement retention for non-renewing subscribers",
    "payment_plan_days":  "plan-commitment upsell converting short-cycle subscribers to longer plans",
    "discount":           "discount-expiry retention for price-sensitive discount-dependent subscribers",
    "actual_amount_paid": "discount-expiry retention for price-sensitive discount-dependent subscribers",
    "plan_list_price":    "discount-expiry retention for price-sensitive discount-dependent subscribers",
    "has_activity_60d":   "dormant listener win-back re-engagement campaign",
    "recency_days":       "dormant listener win-back re-engagement campaign",
    "active_days_30":     "low-engagement activation building listening habits",
    "secs_30":            "low-engagement activation building listening habits",
    "unq_30":             "content discovery and catalog exploration for low-variety listeners",
    "completion_ratio":   "recommendation quality improvement for high-skip listeners",
    "activity_trend":     "declining-engagement early intervention for downward activity trend",
}

def drivers_to_query(top_drivers) -> str:
    # top_drivers: SHAP output, already restricted to the 12 training features
    return " ; ".join(DRIVER_TO_QUERY[d.feature] for d in top_drivers[:2])
```

**Audit rule for Day 1:** every feature your SHAP can emit must have a key here. It does (all 12 covered). If you ever add a feature to the model, add a key here in the same commit — that's how the `both` path stays unbreakable.

---

## 3. Architecture (final)

```
                          User query
                              │
                        FastAPI Backend
                              │
                     Intent Classification
                     (policy | customer | both)
              ┌───────────────┼───────────────┐
              │               │               │
          POLICY          CUSTOMER          BOTH
              │               │               │
              │        get_churn_risk(id)  1. get_churn_risk(id)
              │        → risk + SHAP          → risk + top SHAP drivers
              │               │            2. drivers_to_query(drivers)
              │               │               → churn-conditioned query
              │               │            3. hybrid RAG on THAT query
        Hybrid RAG            │               │
        (BM25+Dense+RRF)      │           Hybrid RAG (conditioned)
              │               │               │
        CrossEncoder          │           CrossEncoder
         rerank               │            rerank
              │               │               │
              └───────────────┼───────────────┘
                              │
                       Context assembly
              (retrieved chunks + SHAP drivers + segment
               context from tenure_days / n_prior_cycles)
                              │
                        LLM Generator  (Llama 3.3 / Qwen2.5)
                   (grounded, cited answer)
                              │
                    Groundedness Check
                              │
                     Confidence Score ──low──► "Insufficient evidence"
                              │
                          Response
                              │
                       SQLite Logging
              (query, route, chunks, drivers, confidence,
               latency, cost, segment)
                              │
              ┌───────────────┴───────────────┐
        MLflow (retrieval sweeps)     Metrics notebook (matplotlib)
                                      [Streamlit dashboard = bonus]
```

---

## 4. Tech stack — every tool, and why

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11 | Stable |
| API | FastAPI + uvicorn | Async, auto OpenAPI docs |
| PDF ingestion | PyMuPDF (`fitz`) | Fast text+metadata |
| Tabular ingestion | pandas | Support/policy CSVs |
| Token counting | tiktoken | Chunk sizing + cost |
| Embeddings | sentence-transformers `BAAI/bge-small-en-v1.5` | Strong small model; `all-MiniLM-L6-v2` = lighter fallback |
| Dense index | FAISS (`faiss-cpu`) | Standard ANN, no server |
| Sparse | `rank_bm25` (BM25Okapi) | Keyword recall |
| Fusion | Reciprocal Rank Fusion (hand-written) | ~15 lines; explain it in interview |
| Reranking | sentence-transformers `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) | Best "I know retrieval" signal |
| LLM | **Groq** (Llama 3.3 70B) primary; **Ollama** (Qwen2.5 7B) offline fallback | Fast+free+good / fully-local story |
| Churn tool | Your deployed KKBox model + SHAP (already done) | The differentiator |
| Intent routing | LLM structured-output (JSON) | Robust, 1 call |
| Groundedness | LLM-as-judge prompt | MVP standard; caveat self-judging |
| **Experiment tracking** | **MLflow** (you already use it) | Turns "why chunk 400?" into a logged sweep |
| Retrieval eval | Hand-rolled Recall@k / MRR (+ optional Ragas) | The comparison table |
| Logging | SQLite | Zero-setup observability proof |
| Metrics viz | matplotlib notebook | Hours not days |
| Dashboard (bonus) | Streamlit | Only if week 3 has slack |

**Deliberately NOT used (say so in README):** LangChain/LlamaIndex for core pipeline, agent frameworks, MCP, LangGraph, long-term memory, web search, multimodal RAG, Kubernetes, Prometheus/Grafana, GPT-from-scratch training. These raise scope more than interview value.

---

## 5. The document corpus (heterogeneous, honestly synthetic)

There are no real KKBox retention playbooks in the world (it's a Kaggle dataset), so supporting docs will be synthetic — but make them **heterogeneous and realistically structured**, not 20 near-identical "playbook" files. Distinct doc types give BM25 and dense retrieval something real to differentiate and make routing meaningful:

| Doc type | Purpose | Feeds which route |
|---|---|---|
| Retention playbooks (P1–P6 above) | The strategies the `both` path retrieves | both, policy |
| Subscription & renewal policy | Auto-renew mechanics, plan lengths, cancellation | policy |
| Offer-eligibility rules | Who qualifies for which discount/offer | policy, both |
| Campaign manual | How re-engagement campaigns run operationally | policy |
| Customer support FAQ | Common subscriber questions | policy |
| Historical retention notes | Case-note style ("dormant + no auto-renew → offer X worked") | both |

**README honesty line:** "Supporting documents were synthesized to emulate enterprise retention documentation; the churn model and its SHAP explanations are real, trained on the KKBox dataset." Defensible and accurate.

---

## 6. Repository structure

```
kkbox-retention-intelligence/
├── README.md
├── DECISIONS.md                  # architectural choices + trade-offs (like your churn project)
├── requirements.txt
├── data/{raw,processed,eval}/
├── src/
│   ├── ingest/{loaders.py,chunker.py}
│   ├── retrieval/{embed.py,dense.py,sparse.py,fusion.py,rerank.py}
│   ├── churn/{model_api.py,driver_to_query.py}
│   ├── generation/{router.py,generator.py,groundedness.py,confidence.py}
│   ├── logging_db.py
│   └── app.py
├── eval/{retrieval_eval.py,faithfulness_eval.py,metrics_report.ipynb}
├── experiments/mlflow_sweep.py   # chunk/topk/model sweeps logged to MLflow
└── scripts/{build_index.py,demo_queries.py}
```

---

## 7. Week 1 — Data + Retrieval Core

**Goal:** working hybrid retriever + real Recall@k / MRR comparison table.

### Days 1–2 — Data & ingestion
- Build the heterogeneous corpus (section 5). Align every playbook to a driver cluster from section 2.
- **Day-1 audit:** run SHAP on a handful of your real KKBox rows, list the top drivers that actually appear, confirm each has a `DRIVER_TO_QUERY` key and a matching playbook. Fix gaps now, not during the demo.
- `src/ingest/loaders.py`: `load_pdf` (fitz) + `load_csv` (pandas) → `{text, source, doc_type, page}`.

### Days 3–4 — Chunking + hybrid retrieval
- `chunker.py`: fixed-size + overlap (~300–400 tokens, ~50 overlap, tiktoken-counted) → `chunks.jsonl`.
- `embed.py`: bge-small, **remember the query prefix** ("Represent this sentence for retrieval: ") — silent quality killer if omitted.
- `dense.py`: FAISS `IndexFlatIP` + **normalized** embeddings (or cosine is meaningless).
- `sparse.py`: BM25Okapi over tokenized chunks.
- `fusion.py`: reciprocal rank fusion (hand-written, ~15 lines).
- `scripts/build_index.py`: ingest → chunk → embed → index in one command.

### Days 5–7 — Reranking + eval harness
- `rerank.py`: CrossEncoder on hybrid top-20 → top-5.
- Hand-label `data/eval/queries.jsonl`: 15–20 queries + gold chunk_ids.
- `eval/retrieval_eval.py` → **full ablation (each row isolates one component) — the table that beats 80% of resume RAG projects:**

| System | Recall@5 | MRR | Latency (ms) |
|---|---|---|---|
| BM25 only | _ | _ | _ |
| Dense only | _ | _ | _ |
| Hybrid (BM25+Dense+RRF) | _ | _ | _ |
| Hybrid + CrossEncoder rerank | _ | _ | _ |

Measure all rows at the **same k**, log latency **consistently** (same machine, warm model, averaged over the eval queries). The rerank row is the key trade-off: it should lift Recall/MRR *and* add latency — so the table tells an accuracy-vs-latency story. Interview answer becomes: *"rerank added X MRR for Y ms — worth it here, might not be at production scale."* That isolation lets you defend **why BM25, why Hybrid, why Reranker** each separately, with numbers.

**Checkpoint:** retrieval works; you can prove hybrid+rerank beats dense-only with numbers.

---

## 8. Week 2 — Generation, Grounding, Churn Integration

**Goal:** all three routes working, including the churn-conditioned `both` path + follow-up explanation.

### Days 8–9 — Grounded generation
- `generator.py`: answer only from chunks; cite `[source, chunk_id]` inline; Groq client (swap to Ollama via env var).
- `groundedness.py`: LLM-judge — are all claims supported by context? (0–1 + unsupported list).
- `confidence.py`: combine top rerank score + groundedness → confidence; below threshold → **"I don't have enough evidence to answer that confidently."**

### Days 10–12 — Churn tool + the `both` path (your best engineering time)
- `model_api.py`: `get_churn_risk(id)` → `{risk, top_drivers (from the 12 features), segment (tenure_days, n_prior_cycles)}`. Reuse your deployed model + existing SHAP.
- `router.py`: LLM returns `policy | customer | both`.
  - `policy` → hybrid RAG on raw query.
  - `customer` → churn tool only → explain risk from SHAP.
  - `both` → **churn tool first** → `drivers_to_query()` → hybrid RAG on the *driver-derived* query → context = SHAP drivers + retrieved playbook + segment context → generate.
- **Follow-up explanation beat** (folds in the "explainability retrieval" idea cheaply): support a follow-up like *"Why are you recommending that?"* by re-surfacing the same SHAP drivers + playbook chunks and having the LLM justify the recommendation with citation. No new subsystem — reuses the `both` context. Great demo moment.

**Canonical demo queries** (`scripts/demo_queries.py`):
1. *Policy:* "What re-engagement offers can a lapsed subscriber get before renewal?" → RAG only.
2. *Customer:* "Why is subscriber msno=XXptr… at high churn risk?" → churn tool only.
3. *Both:* "This subscriber is at risk — why, and what should we do?" → SHAP surfaces e.g. `has_activity_60d=0` + `is_auto_renew=0` → retrieves Dormant win-back + Auto-renew re-enablement playbooks → cites both drivers and playbooks.
4. *Follow-up:* "Why that offer and not a discount?" → re-justifies from the same context. **Practice narrating 3→4; it's the centerpiece.**

### Days 13–14 — Buffer (not optional)
bge prefix, FAISS normalization, SHAP shape, LLM JSON parsing — something overruns. If not, start week 3 early.

**Checkpoint:** three routes work; `both` visibly conditions retrieval on churn output; follow-up justification works.

---

## 9. Week 3 — Eval Depth, MLflow, Documentation

**Goal:** measured, tracked, logged, documented, interview-ready.

### Days 15–16 — MLflow retrieval sweep
- `experiments/mlflow_sweep.py`: sweep **chunk_size {200,400,600}**, **overlap {25,50}**, **top_k {5,10,20}**, **embedding model {bge-small, MiniLM}**, **rerank {on,off}** — log Recall@5, MRR, latency per run to MLflow.
- Now "why chunk size 400?" → *"I swept 200/400/600 and logged retrieval metrics in MLflow; 400 won on Recall@5 without a latency penalty."* Keep the sweep bounded to retrieval — don't track everything.

### Day 17 — Faithfulness + cost
- `faithfulness_eval.py`: LLM-judge hallucination rate over ~20 held-out Qs (honestly caveated).
- Add latency + cost estimate (tiktoken tokens × price, or "$0 local").

### Days 18–19 — Documentation (treat like your churn project)
- `DECISIONS.md`: key choices + trade-offs (why hybrid, why RRF, why CrossEncoder, why no agent framework, model-vs-monitoring-features).
- `README.md`: architecture diagram, **both eval tables + MLflow screenshots**, honest synthetic-docs line, **"What's excluded and why"** section, setup/run.
- `metrics_report.ipynb`: matplotlib over the SQLite log (latency, confidence, route breakdown) — your "dashboard" for now.

### Days 20–21 — Resume + interview prep + demo
- **Resume bullet:**
  > Built a KKBox Retention Intelligence Assistant combining hybrid RAG (BM25 + dense + cross-encoder reranking) over enterprise-style retention docs with live tool-calling into a deployed churn model; conditioned retrieval on the model's SHAP drivers so recommended playbooks matched each subscriber's risk cause. Swept retrieval configs in MLflow, improving Recall@5 by [X]% over dense-only; grounded all answers with citation + confidence-gated fallback.
- **Prep a 10–15 min demo** + anticipated-questions list: why hybrid not dense; how you prevent hallucination; how you evaluate retrieval; what's novel (SHAP-conditioned retrieval); model vs monitoring features; why not fine-tune everything (knowledge changes, retrieval cheaper, factual grounding, no retrain for new docs — answer conceptually, no GPT-training needed).

**Bonus (only if slack):** Streamlit dashboard, Dockerfile, LangGraph comparison of your hand-built orchestration.

---

## 10. Sequencing rules (so it never collapses)

1. **Never skip the Week-2 buffer.** Difference between "done" and "80% done, undemoable."
2. **Weeks 1–2 stand alone.** Three working routes + a retrieval eval table = complete project. Week 3 is depth.
3. **The `both` path is the point.** If short on time, cut dashboard, cut faithfulness eval, cut Docker, cut MLflow before you cut churn-conditioned retrieval.
4. **Every number is real.** One honest Recall@5 table + MLflow sweep beats ten unmeasured claims.
5. **Locked scope.** No multi-agent, MCP, LangGraph, long-term memory, GPT-from-scratch. Polish over breadth.

---

## 11. Setup checklist (Day 1, first hour)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn pymupdf pandas tiktoken \
    sentence-transformers faiss-cpu rank-bm25 \
    shap scikit-learn mlflow matplotlib jupyter python-dotenv
# LLM: Groq (pip install groq, set GROQ_API_KEY) OR Ollama (install + `ollama pull qwen2.5`)
```
Pin versions into `requirements.txt`. Get one thin end-to-end slice working (ingest 1 doc → embed → dense search → print) before building breadth. Then run the Day-1 SHAP driver audit against your real model before writing any playbooks.
