# DECISIONS

Architectural choices made before writing any implementation code, and the reasoning behind each.
Updated as the project progresses; see `retention_intelligence_assistant_build_plan.md` for the
full day-by-day plan these decisions come from.

## 1. Why SHAP-conditioned retrieval is the differentiator, not just RAG + a tool call

A generic RAG-over-policy-docs project is a well-worn portfolio piece. The design choice that makes
this one different: for a combined "why is this subscriber at risk, and what should we do"
question, the churn model's own SHAP output is turned into the retrieval query, so the playbook
that comes back is conditioned on *why that specific subscriber* is at risk, not a generic answer.
This means retrieval quality and churn-model explainability are coupled by design, not two
independent demos bolted together.

## 2. Why hybrid retrieval (BM25 + dense + RRF), not dense-only

BM25 catches exact-token matches a bi-encoder can blur (feature names, offer IDs); dense catches
paraphrase/semantic matches BM25 misses. Reciprocal Rank Fusion combines the two ranked lists on
rank position rather than raw score, because BM25 scores are unbounded and cosine similarity is
bounded in [-1, 1] — fusing on raw scores would let whichever retriever produces larger numbers
dominate regardless of actual relevance. This is a real trade-off to measure, not assume — the
retrieval eval (Week 1) will produce an ablation table isolating BM25-only, dense-only, hybrid, and
hybrid+rerank so the choice can be defended with numbers.

## 3. Why a cross-encoder reranker is in the pipeline at all

A cross-encoder scores (query, chunk) pairs jointly and is more precise than bi-encoder cosine
similarity, but too slow to run over a whole corpus — so it only ever sees the fused top-k
candidates. Whether it's worth the added latency at this project's corpus scale is an empirical
question the Week 1 ablation table will answer, not an assumption baked in upfront.

## 4. Why the corpus will be honestly synthetic, and heterogeneous on purpose

There are no real KKBox retention playbooks in the world — it's a Kaggle dataset with no
accompanying CRM/support data. The supporting documents will be synthesized, but as distinct
document types (playbooks, subscription policy, campaign manual, support FAQ, historical case
notes, an offer-eligibility table) rather than twenty near-identical "playbook" files, so BM25 and
dense retrieval have something structurally real to differentiate. The README will state this
honestly rather than implying the documents are real enterprise data.

## 5. Why the churn model is reused from a separate project, not retrained here

A deployed, calibrated KKBox churn model (XGBoost, isotonic-calibrated, SHAP-explained) already
exists from prior work. This project's job is to build the retrieval/generation system around it
and condition retrieval on its real SHAP output — not to re-solve churn prediction. The 12 features
that model was trained on are the only features that can ever appear as a SHAP driver here; two
additional fields it tracks (tenure, prior-cycle count) are carried as descriptive segment context
in answers and logs, but are never eligible to be treated as a driver, since the model was never
trained to attribute risk to them. Every feature the model can attribute SHAP to must map to a
retention playbook — that mapping is audited against the real model on Day 1, before any playbook
content is written, so a coverage gap is caught immediately rather than during a later demo.

## 6. Why Python 3.12 is pinned, and why `requirements.txt` mirrors the churn model's own versions

The churn model artifact this project loads was built and pickled under Python 3.12 with specific
`xgboost`/`scikit-learn`/`shap` versions. Unpickling a model artifact under a different minor version
of these libraries is a known source of silent or loud breakage, so this project's environment
pins to match that build environment exactly, rather than picking the newest available versions.

## 7. What's deliberately excluded, and why

No LangChain/LlamaIndex for the core pipeline, no agent framework, no MCP, no LangGraph, no
long-term memory, no web search, no Kubernetes, no Prometheus/Grafana. Each of these would raise
scope without raising interview value for a project of this size — the retrieval fusion, the
churn-conditioning logic, and the grounding/confidence gate are hand-rolled specifically so every
line is defensible, the same restraint principle the plan is built around.

## 8. Repository structure

`src/` is organized by pipeline stage (`ingest`, `retrieval`, `churn`, `generation`) rather than by
layer-agnostic utility folders, so the three routes in the architecture diagram map directly onto
which modules a request touches. `scripts/` holds one-command entry points (build the index, run
the demo queries); `eval/` and `experiments/` are kept separate because retrieval evaluation and the
MLflow parameter sweep answer different questions (is retrieval good vs. which config is best) and
shouldn't share one script.
