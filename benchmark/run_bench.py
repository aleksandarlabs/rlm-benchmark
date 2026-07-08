#!/usr/bin/env python3
"""
RLM vs Vanilla benchmark runner over a codebase that exceeds the model context window.

Usage:
    python benchmark/run_bench.py --condition vanilla-small
    python benchmark/run_bench.py --condition rlm-strong-subs --set needles
    python benchmark/run_bench.py --condition rlm-strong-subs --set all --limit 3

Conditions (see config.py): vanilla-small, vanilla-strong, rlm-strong-subs, rlm-all-small.

Measures, per condition: needle accuracy, judge score on synthesis, token/cost
breakdown per model, and latency. Detects RLM convergence failures (when it does
not produce a final answer and returns a REPL block instead).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import openai

import config as C
import corpus
import scoring

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------- Llamadas a modelos ----------

def _client(model_key: str) -> tuple[openai.OpenAI, str]:
    m = C.MODELS[model_key]
    client = openai.OpenAI(api_key=C.resolve_api_key(model_key), base_url=m["base_url"])
    return client, m["model_id"]


def call_plain(model_key: str, prompt: str, max_output: int) -> dict:
    """Una sola llamada chat. Devuelve texto + usage."""
    client, model_id = _client(model_key)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_output,
        extra_body=C.provider_body(model_key),  # pin de proveedor OpenRouter (si aplica)
    )
    u = resp.usage
    return {
        "text": resp.choices[0].message.content or "",
        "input_tokens": u.prompt_tokens,
        "output_tokens": u.completion_tokens,
        "model_id": model_id,
    }


# ---------- Vanilla ----------

def run_vanilla(condition: dict, question: str, flat_context: str) -> dict:
    model_key = condition["root"]
    window = C.MODELS[model_key]["window"]
    max_out = C.VANILLA_MAX_OUTPUT_TOKENS
    # Conservative input budget: leaves room for prompt scaffolding + output.
    budget = int((window - max_out - 4000) * 0.9)

    t0 = time.time()
    truncated = False
    out = None
    for attempt in range(4):
        ctx, truncated = corpus.truncate_to_tokens(flat_context, budget)
        prompt = (
            "Below is the source code of a repository (possibly truncated).\n\n"
            f"{ctx}\n\n---\nQUESTION:\n{question}\n\n"
            "Answer concisely and briefly, citing files when applicable."
        )
        try:
            out = call_plain(model_key, prompt, max_out)
            break
        except openai.BadRequestError as e:
            if "context length" in str(e).lower() and attempt < 3:
                budget = int(budget * 0.8)  # trim and retry
                print(f"  ⚠️  context exceeded the window, retrying with budget={budget:,} tok")
                continue
            raise
    elapsed = time.time() - t0

    cost = C.cost_usd(model_key, out["input_tokens"], out["output_tokens"])
    return {
        "text": out["text"],
        "elapsed_s": round(elapsed, 1),
        "context_truncated": truncated,
        "cost_usd": cost,
        "usage_by_model": {out["model_id"]: {
            "input_tokens": out["input_tokens"],
            "output_tokens": out["output_tokens"],
            "cost_usd": cost,
        }},
        "failed": False,
    }


# ---------- RLM ----------

def _looks_unfinished(text: str) -> bool:
    """RLM 'failed' if it never produced a final answer (returned a repl block or empty)."""
    if not text or not text.strip():
        return True
    return "```repl" in text


def _model_id_to_key(model_id: str) -> str | None:
    for k, m in C.MODELS.items():
        if m["model_id"] == model_id:
            return k
    return None


def _install_openrouter_provider_pin():
    """The rlm library does not expose `provider`; we inject the pin on OpenRouter calls.

    Patches Completions.create (sync and async) to add the provider of model
    'strong' (the only OpenRouter model) whenever the base_url is openrouter.
    """
    if getattr(openai, "_rlm_provider_patched", False):
        return
    pin = C.provider_body("strong").get("provider")
    if not pin:
        return
    from openai.resources.chat import completions as _c

    def wrap(orig):
        def create(self, *args, **kwargs):
            base = str(getattr(self._client, "base_url", ""))
            if "openrouter.ai" in base:
                eb = dict(kwargs.get("extra_body") or {})
                eb.setdefault("provider", pin)
                kwargs["extra_body"] = eb
            return orig(self, *args, **kwargs)
        return create

    _c.Completions.create = wrap(_c.Completions.create)
    _c.AsyncCompletions.create = wrap(_c.AsyncCompletions.create)
    openai._rlm_provider_patched = True


def run_rlm(condition: dict, question: str, context_dict: dict) -> dict:
    from rlm import RLM

    _install_openrouter_provider_pin()

    root_key = condition["root"]
    sub_key = condition["subs"]
    root_m, sub_m = C.MODELS[root_key], C.MODELS[sub_key]

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": root_m["model_id"],
            "base_url": root_m["base_url"],
            "api_key": C.resolve_api_key(root_key),
        },
        other_backends=["openai"],
        other_backend_kwargs=[{
            "model_name": sub_m["model_id"],
            "base_url": sub_m["base_url"],
            "api_key": C.resolve_api_key(sub_key),
        }],
        max_iterations=C.RLM_MAX_ITERATIONS,
        verbose=True,
    )

    root_prompt = (
        f"{question}\n\n"
        "The source code of the repository is in the variable `context`, a dict "
        "{file_path: content}. Explore it with the REPL and use llm_query / "
        "llm_query_batched to analyse the files. When you have the answer, "
        "assign it to a variable and call FINAL_VAR('var_name')."
    )

    t0 = time.time()
    result = rlm.completion(prompt=context_dict, root_prompt=root_prompt)
    elapsed = time.time() - t0

    text = result.response or ""

    # Cost breakdown per model from the library's usage_summary.
    usage_by_model: dict[str, dict] = {}
    total_cost = 0.0
    summary = getattr(result, "usage_summary", None)
    model_usages = getattr(summary, "model_usage_summaries", {}) if summary else {}
    for model_id, mu in model_usages.items():
        key = _model_id_to_key(model_id)
        it = getattr(mu, "total_input_tokens", 0)
        ot = getattr(mu, "total_output_tokens", 0)
        c = C.cost_usd(key, it, ot) if key else 0.0
        total_cost += c
        usage_by_model[model_id] = {
            "input_tokens": it, "output_tokens": ot,
            "calls": getattr(mu, "total_calls", None), "cost_usd": c,
        }

    return {
        "text": text,
        "elapsed_s": round(elapsed, 1),
        "context_truncated": False,
        "cost_usd": total_cost,
        "usage_by_model": usage_by_model,
        "failed": _looks_unfinished(text),
    }


# ---------- Orchestration ----------

def make_judge_call():
    def judge_call(prompt: str) -> str:
        out = call_plain(C.JUDGE_MODEL, prompt, C.JUDGE_MAX_OUTPUT_TOKENS)
        return out["text"]
    return judge_call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=list(C.CONDITIONS.keys()))
    ap.add_argument("--set", dest="qset", default="all", choices=["needles", "synthesis", "all"])
    ap.add_argument("--limit", type=int, default=0, help="max questions per set (0 = all)")
    ap.add_argument("--no-judge", action="store_true", help="do not score synthesis with LLM-judge")
    ap.add_argument("--force-context", action="store_true", help="rebuild context cache")
    ap.add_argument("--corpus-dir", type=Path, default=corpus.DEFAULT_CORPUS_DIR,
                    help="path to codebase to benchmark (default: ./corpus)")
    ap.add_argument("--cache-dir", type=Path, default=corpus.DEFAULT_CACHE_DIR,
                    help="directory for context cache")
    ap.add_argument("--questions", type=Path,
                    default=Path(__file__).resolve().parent.parent / "questions" / "questions.json",
                    help="path to questions JSON file")
    args = ap.parse_args()

    condition = C.CONDITIONS[args.condition]
    print(f"\n{'='*70}\n  CONDITION: {args.condition}  ({condition['mode']})\n{'='*70}")

    print("Loading context...")
    files = corpus.build_context(
        repo_dir=args.corpus_dir,
        cache_dir=args.cache_dir,
        force=args.force_context,
    )
    st = corpus.stats(files)
    print(f"  {st['files']} files · {st['est_tokens']:,} estimated tokens")
    flat = corpus.as_flat_string(files) if condition["mode"] == "vanilla" else ""

    qs = json.loads(args.questions.read_text())
    needles = qs["needles"]
    synthesis = qs["synthesis"]
    if args.limit:
        needles, synthesis = needles[:args.limit], synthesis[:args.limit]

    judge_call = None if args.no_judge else make_judge_call()

    results = {
        "condition": args.condition,
        "mode": condition["mode"],
        "timestamp": datetime.now().isoformat(),
        "context_tokens_est": st["est_tokens"],
        "models": {k: C.MODELS[condition[k]] for k in ("root", "subs") if k in condition},
        "needles": [],
        "synthesis": [],
    }

    def run_one(question: str) -> dict:
        if condition["mode"] == "vanilla":
            return run_vanilla(condition, question, flat)
        return run_rlm(condition, question, files)

    # --- Needles ---
    if args.qset in ("needles", "all"):
        print(f"\n--- NEEDLES ({len(needles)}) ---")
        for q in needles:
            print(f"\n[{q['id']}] {q['question']}")
            r = run_one(q["question"])
            ok = (not r["failed"]) and scoring.score_needle(r["text"], q["accept"])
            mark = "✅" if ok else ("💥 RLM FAILED" if r["failed"] else "❌")
            print(f"  {mark}  {r['elapsed_s']}s · ${r['cost_usd']:.4f} · trunc={r['context_truncated']}")
            results["needles"].append({**q, "correct": ok, "failed": r["failed"],
                                       "elapsed_s": r["elapsed_s"], "cost_usd": r["cost_usd"],
                                       "usage_by_model": r["usage_by_model"],
                                       "response": r["text"][:1500]})

    # --- Synthesis ---
    if args.qset in ("synthesis", "all"):
        print(f"\n--- SYNTHESIS ({len(synthesis)}) ---")
        for q in synthesis:
            print(f"\n[{q['id']}] {q['question'][:80]}...")
            r = run_one(q["question"])
            verdict = {"score": None, "reason": "no judge"}
            if judge_call and not r["failed"]:
                verdict = scoring.judge_synthesis(q["question"], q["rubric"], r["text"], judge_call)
            sc = "💥 RLM FAILED" if r["failed"] else f"score={verdict['score']}/10"
            print(f"  {sc} · {r['elapsed_s']}s · ${r['cost_usd']:.4f}")
            results["synthesis"].append({**q, "judge": verdict, "failed": r["failed"],
                                         "elapsed_s": r["elapsed_s"], "cost_usd": r["cost_usd"],
                                         "usage_by_model": r["usage_by_model"],
                                         "response": r["text"][:2500]})

    # --- Summary ---
    n_ok = sum(1 for x in results["needles"] if x["correct"])
    n_tot = len(results["needles"])
    s_scores = [x["judge"]["score"] for x in results["synthesis"] if x["judge"]["score"] is not None]
    total_cost = sum(x["cost_usd"] for x in results["needles"] + results["synthesis"])
    total_time = sum(x["elapsed_s"] for x in results["needles"] + results["synthesis"])
    fails = sum(1 for x in results["needles"] + results["synthesis"] if x["failed"])

    results["summary"] = {
        "needles_correct": n_ok, "needles_total": n_tot,
        "synthesis_avg_score": round(sum(s_scores) / len(s_scores), 2) if s_scores else None,
        "rlm_failures": fails, "total_cost_usd": round(total_cost, 4),
        "total_time_s": round(total_time, 1),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"{args.condition}_{int(time.time())}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    print(f"\n{'='*70}\n  SUMMARY — {args.condition}\n{'='*70}")
    print(f"  Needles:   {n_ok}/{n_tot} correct")
    print(f"  Synthesis: {results['summary']['synthesis_avg_score']} /10 (avg)")
    print(f"  RLM failures (no convergence): {fails}")
    print(f"  Total cost:   ${total_cost:.4f}")
    print(f"  Total time:  {total_time:.1f}s")
    print(f"  Saved to:   {out_file}")


if __name__ == "__main__":
    main()
