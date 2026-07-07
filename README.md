# KKBox Retention Intelligence Assistant

A retention assistant for a music-streaming service that will answer policy questions with hybrid
RAG, answer subscriber-risk questions by calling a deployed KKBox churn model, and — for combined
questions — condition retrieval on the model's SHAP drivers, so the retention playbook it returns
matches *why that specific subscriber is at risk*.

This is not "another RAG chatbot." The differentiator is a retrieval system conditioned on a live
churn model's SHAP explanations, not document search alone. The full rationale for that claim, the
architecture, and the phased build plan are in `retention_intelligence_assistant_build_plan.md`.

## Status

Day 1 — repository scaffolding. Nothing is implemented yet; this commit is the skeleton the rest of
the project builds on. See `DECISIONS.md` for the architectural choices locked in before writing any
retrieval or generation code.

## Planned architecture

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
                        LLM Generator
                   (grounded, cited answer)
                              │
                    Groundedness check
                              │
                     Confidence score ──low──► "Insufficient evidence"
                              │
                          Response
                              │
                       SQLite logging
```

## Planned tech stack

| Layer | Tool |
|---|---|
| API | FastAPI + uvicorn |
| PDF ingestion | PyMuPDF (`fitz`) |
| Tabular ingestion | pandas |
| Embeddings | sentence-transformers (`bge-small-en-v1.5`) |
| Dense index | FAISS |
| Sparse | `rank_bm25` (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion (hand-written) |
| Reranking | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | Groq (Llama 3.3 70B) primary, Ollama (local) fallback |
| Churn tool | A deployed KKBox churn model + SHAP (separate project, reused here) |
| Experiment tracking | MLflow |
| Logging | SQLite |

Deliberately not used: LangChain/LlamaIndex, agent frameworks, MCP, LangGraph, long-term memory,
Kubernetes, Prometheus/Grafana. See `DECISIONS.md` for why.

## Repository layout

```
├── README.md
├── DECISIONS.md
├── retention_intelligence_assistant_build_plan.md
├── requirements.txt
├── .env.example
├── data/{raw,processed,eval}/
├── src/
│   ├── ingest/         (loaders, chunker)
│   ├── retrieval/      (embed, dense, sparse, fusion, rerank)
│   ├── churn/          (churn model API, driver-to-query conditioning)
│   ├── generation/     (router, generator, groundedness, confidence)
│   ├── logging_db.py
│   └── app.py
├── eval/               (retrieval eval, faithfulness eval, metrics notebook)
├── experiments/        (MLflow sweeps)
└── scripts/            (build_index, demo_queries)
```

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in GROQ_API_KEY, or set LLM_BACKEND=ollama
```

Nothing is runnable yet — ingestion, retrieval, churn integration, and generation land over the
following days per the build plan.
