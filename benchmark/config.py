"""
Provider, model, price, and condition configuration for the benchmark.

Everything can be overridden via environment variables so you can sweep models
without touching code. Default model IDs are sensible placeholders: adjust them
to whatever you actually use on Groq / OpenRouter.

Prices are in USD per 1M tokens. VERIFY them against the provider's pricing
page before drawing cost conclusions.
"""

from __future__ import annotations

import os

# Load keys from a .env file if present (does not depend on the shell).
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# --- Endpoints (both are OpenAI-compatible -> backend="openai") ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# --- Model registry ---
# Each logical model defines: provider, real id, context window (tokens),
# and price (in/out per 1M tokens).
MODELS: dict[str, dict] = {
    # Small model on Groq (sub-queries and "all-small" case). Do not use llama.
    "small": {
        "provider": "groq",
        "base_url": GROQ_BASE_URL,
        "api_key_env": "GROQ_API_KEY",
        "model_id": os.environ.get("GROQ_SMALL_MODEL", "qwen/qwen3.6-27b"),
        "window": int(os.environ.get("GROQ_SMALL_WINDOW", "131072")),
        # Groq qwen3.6-27b prices: VERIFY on Groq's pricing page.
        "price_in": float(os.environ.get("GROQ_SMALL_PRICE_IN", "0.29")),
        "price_out": float(os.environ.get("GROQ_SMALL_PRICE_OUT", "0.59")),
    },
    # Orchestrator via OpenRouter (RLM root and strong vanilla baseline).
    # kimi-k2.6 (general variant, not the code-agentic one). Pinned to a single
    # endpoint via provider_only to avoid quantization/window roulette (e.g.
    # Nebius serves only 8K context). Routing tag: "moonshotai/int4".
    "strong": {
        "provider": "openrouter",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": os.environ.get("RLM_STRONG_MODEL", "moonshotai/kimi-k2.6"),
        # Pin to a single OpenRouter endpoint (empty = automatic routing).
        "provider_only": os.environ.get("RLM_STRONG_PROVIDER", "moonshotai/int4"),
        "window": int(os.environ.get("RLM_STRONG_WINDOW", "262144")),
        # Moonshot AI int4 prices (per 1M tokens), verified via API.
        "price_in": float(os.environ.get("RLM_STRONG_PRICE_IN", "0.95")),
        "price_out": float(os.environ.get("RLM_STRONG_PRICE_OUT", "4.0")),
    },
    # Judge for synthesis questions (LLM-as-judge). Independent from the
    # experiment models to avoid bias. gpt-5-mini returns clean JSON (~80 tok)
    # and is cheap.
    "judge": {
        "provider": "openrouter",
        "base_url": OPENROUTER_BASE_URL,
        "api_key_env": "OPENROUTER_API_KEY",
        "model_id": os.environ.get("BENCH_JUDGE_MODEL_ID", "openai/gpt-5-mini"),
        "window": 400000,
        "price_in": float(os.environ.get("BENCH_JUDGE_PRICE_IN", "0.25")),
        "price_out": float(os.environ.get("BENCH_JUDGE_PRICE_OUT", "2.0")),
    },
}


# --- Experiment conditions ---
# mode: "vanilla" (single call, truncated to window) | "rlm" (recursive)
# root: logical model that orchestrates / answers
# subs: logical model for RLM sub-queries (only mode="rlm")
CONDITIONS: dict[str, dict] = {
    # Cheap baseline: small model, context truncated to its window -> hits the wall.
    "vanilla-small": {"mode": "vanilla", "root": "small"},
    # Quality ceiling: strong long-context model (also truncates if corpus > window).
    "vanilla-strong": {"mode": "vanilla", "root": "strong"},
    # Canonical RLM pattern: strong orchestrator + cheap subs.
    "rlm-strong-subs": {"mode": "rlm", "root": "strong", "subs": "small"},
    # Extreme cheap case: everything with the small model.
    "rlm-all-small": {"mode": "rlm", "root": "small", "subs": "small"},
}

# Model used as judge for synthesis questions (LLM-as-judge).
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "judge")

# Judge output token cap. IMPORTANT: keep it generous; models that "reason"
# (kimi, etc.) consume tokens before the JSON and returned empty with 300.
JUDGE_MAX_OUTPUT_TOKENS = int(os.environ.get("JUDGE_MAX_OUTPUT_TOKENS", "1500"))

# How many iterations we give the RLM root.
RLM_MAX_ITERATIONS = int(os.environ.get("RLM_MAX_ITERATIONS", "30"))

# Output token cap for vanilla calls.
VANILLA_MAX_OUTPUT_TOKENS = int(os.environ.get("VANILLA_MAX_OUTPUT_TOKENS", "4000"))


def provider_body(model_key: str) -> dict:
    """OpenRouter extra_body to pin to a single provider/quantization.

    Returns {} if the model has no provider_only (automatic routing).
    """
    tag = MODELS.get(model_key, {}).get("provider_only")
    if not tag:
        return {}
    return {"provider": {"only": [tag], "allow_fallbacks": False}}


def resolve_api_key(model_key: str) -> str:
    env = MODELS[model_key]["api_key_env"]
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"Missing environment variable {env} for model '{model_key}'.")
    return key


def cost_usd(model_key: str, input_tokens: int, output_tokens: int) -> float:
    m = MODELS[model_key]
    return (input_tokens * m["price_in"] + output_tokens * m["price_out"]) / 1_000_000
