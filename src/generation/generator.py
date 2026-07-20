"""Grounded answer generation, plus the shared LLM client both other generation modules reuse.

Backend switches on the LLM_BACKEND env var: "groq" (Llama 3.3 70B, default, needs GROQ_API_KEY) or
"ollama" (Qwen2.5, local, fully offline). Nothing downstream needs to know which one is active.

The prompt builds ONE context block (retrieved chunks, cited by [source, chunk_id]; SHAP drivers +
segment context when a churn lookup ran) and asks the LLM to answer only from what is given, never
from its own background knowledge about retention strategy -- that's what keeps the answer citable
and checkable by groundedness.py.
"""
import os

from dotenv import load_dotenv

from src import config

load_dotenv()

BACKEND = os.environ.get("LLM_BACKEND", config.LLM_BACKEND_DEFAULT).lower()
GROQ_MODEL = os.environ.get("GROQ_MODEL", config.GROQ_MODEL_DEFAULT)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", config.OLLAMA_MODEL_DEFAULT)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", config.OLLAMA_HOST_DEFAULT)

_groq_client = None

SYSTEM_PROMPT = """You are a retention assistant for a music-streaming subscription service.
Answer ONLY using the CONTEXT provided below -- retrieved playbook/policy chunks and, where present,
a subscriber's churn-risk drivers. Do not use outside knowledge about retention strategy.
Cite every claim that comes from a retrieved chunk inline as [source, chunk_id], using the exact
source and chunk_id given in the context. If the context does not contain enough information to
answer, say so explicitly instead of guessing."""


def _groq_chat(system: str, user: str, temperature: float) -> str:
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set (checked .env and the environment). Set it, or set "
                "LLM_BACKEND=ollama in .env to use a local Ollama model instead."
            )
        _groq_client = Groq(api_key=api_key)
    resp = _groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def _ollama_chat(system: str, user: str, temperature: float) -> str:
    import requests
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": temperature},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_llm(system: str, user: str, temperature: float = 0.2) -> str:
    if BACKEND == "ollama":
        return _ollama_chat(system, user, temperature)
    return _groq_chat(system, user, temperature)


def _format_chunks(chunks: list[dict]) -> str:
    parts = [f"[{c['source']}, {c['chunk_id']}]\n{c['text']}" for c in chunks]
    return "RETRIEVED CONTEXT:\n" + "\n\n".join(parts)


def _format_risk(risk) -> str:
    drivers = ", ".join(
        f"{d.feature}={d.feature_value:g} (shap={d.shap_value:+.2f})" for d in risk.top_drivers
    )
    return (
        f"SUBSCRIBER CHURN RISK:\nmsno={risk.msno}, calibrated P(churn)={risk.risk:.2f}\n"
        f"top SHAP drivers: {drivers}\n"
        f"segment context (descriptive only, NOT a churn driver): "
        f"tenure_days={risk.segment['tenure_days']}, n_prior_cycles={risk.segment['n_prior_cycles']}"
    )


def build_context(chunks: list[dict] | None = None, risk=None) -> str:
    """Shared with groundedness.py so the judge checks the answer against the exact same context
    the generator saw -- not a re-derived approximation of it."""
    sections = []
    if risk is not None:
        sections.append(_format_risk(risk))
    if chunks:
        sections.append(_format_chunks(chunks))
    return "\n\n".join(sections) if sections else "(no context retrieved)"


def generate_answer(query: str, chunks: list[dict] | None = None, risk=None) -> str:
    context = build_context(chunks, risk)
    user = f"{context}\n\nQUESTION: {query}"
    return call_llm(SYSTEM_PROMPT, user)
