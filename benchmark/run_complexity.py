#!/usr/bin/env python3
"""
Experimento 2 — curva de crossover por complejidad de tarea.

Corre 3 paradigmas (vanilla-truncado / Pi agent / RLM) sobre el suite de tareas
en tres tiers (constante / lineal / cuadratica), con el MISMO modelo en los tres
brazos (fairness: comparamos paradigmas, no modelos). Puntua con F1 set-based y
produce la tabla de crossover (tier x brazo).

Uso:
    # sonda barata: 1 tarea por tier, los 3 brazos
    python benchmark/run_complexity.py --limit-per-tier 1

    # solo un brazo / tier
    python benchmark/run_complexity.py --arms pi --tiers quadratic

    # matriz completa
    python benchmark/run_complexity.py

Modelo comun: config.MODELS['strong'] (kimi-k2.6) para vanilla y RLM;
Pi usa el mismo via --provider/--model.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import config as C
import corpus
import pi_agent
import run_bench
import scoring_complexity as SC

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TIERS = ["constant", "linear", "quadratic"]
ARMS = ["vanilla", "pi", "rlm"]

# LLM arm configurations (same 'strong' model = kimi-k2.6).
_ARM_CONDITION = {
    "vanilla": {"mode": "vanilla", "root": "strong"},
    "rlm": {"mode": "rlm", "root": "strong", "subs": "strong"},
}

# Pi: model/provider for the agent arm (same model as the other arms).
PI_PROVIDER = "openrouter"
PI_MODEL = "moonshotai/kimi-k2.6"
PI_THINKING = "medium"


def run_one_arm(arm: str, question: str, flat: str, files: dict, repo_dir: str) -> dict:
    if arm == "pi":
        r = pi_agent.run_pi_agent(question, repo_dir, provider=PI_PROVIDER,
                                  model_id=PI_MODEL, thinking=PI_THINKING)
        return {"text": r["text"], "cost_usd": r["cost_usd"], "elapsed_s": r["elapsed_s"],
                "failed": r["failed"], "turns": r["assistant_turns"], "tool_calls": r["tool_calls"]}
    cond = _ARM_CONDITION[arm]
    if cond["mode"] == "vanilla":
        r = run_bench.run_vanilla(cond, question, flat)
    else:
        r = run_bench.run_rlm(cond, question, files)
    return {"text": r["text"], "cost_usd": r["cost_usd"], "elapsed_s": r["elapsed_s"],
            "failed": r.get("failed", False), "turns": None, "tool_calls": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated: vanilla,pi,rlm")
    ap.add_argument("--tiers", default=",".join(TIERS), help="comma-separated: constant,linear,quadratic")
    ap.add_argument("--limit-per-tier", type=int, default=0, help="max tasks per tier (0=all)")
    ap.add_argument("--corpus-dir", type=Path, default=corpus.DEFAULT_CORPUS_DIR,
                    help="path to codebase to benchmark (default: ./corpus)")
    ap.add_argument("--cache-dir", type=Path, default=corpus.DEFAULT_CACHE_DIR,
                    help="directory for context cache")
    ap.add_argument("--questions", type=Path,
                    default=Path(__file__).resolve().parent.parent / "questions" / "questions-complexity.json",
                    help="path to complexity questions JSON file")
    ap.add_argument("--force-context", action="store_true", help="rebuild context cache")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a in ARMS]
    tiers = [t for t in args.tiers.split(",") if t in TIERS]
    repo_dir = str(args.corpus_dir.resolve())

    q = json.loads(args.questions.read_text())
    tasks = []
    for t in tiers:
        items = q[t][: args.limit_per_tier] if args.limit_per_tier else q[t]
        tasks.extend(items)

    print(f"Common model: {C.MODELS['strong']['model_id']} | Pi: {PI_PROVIDER}/{PI_MODEL}")
    print(f"Arms: {arms} | Tiers: {tiers} | Tasks: {len(tasks)}")
    print("Loading context...")
    files = corpus.build_context(
        repo_dir=args.corpus_dir,
        cache_dir=args.cache_dir,
        force=args.force_context,
    )
    flat = corpus.as_flat_string(files)
    print(f"  {len(files)} files · {corpus.stats(files)['est_tokens']:,} tokens\n")

    results = {"timestamp": datetime.now().isoformat(),
               "model": C.MODELS["strong"]["model_id"], "tasks": []}

    for task in tasks:
        print(f"\n=== [{task['tier']}] {task['id']}: {task['question'][:70]}...")
        row = {"id": task["id"], "tier": task["tier"], "arms": {}}
        for arm in arms:
            print(f"  · {arm} ...", end="", flush=True)
            out = run_one_arm(arm, task["question"], flat, files, repo_dir)
            score = {"f1": 0.0} if out["failed"] else SC.score_task(task, out["text"])
            f1 = score.get("f1", 0.0)
            fail_mark = "  💥" if out["failed"] else ""
            turns_mark = f"  turns={out['turns']}" if out["turns"] is not None else ""
            print(f" F1={f1}  ${out['cost_usd']:.4f}  {out['elapsed_s']}s{fail_mark}{turns_mark}")
            row["arms"][arm] = {**out, "score": score, "f1": f1,
                                "response": out["text"][:1200]}
        results["tasks"].append(row)

    # --- Crossover table (tier x arm -> average F1) ---
    print(f"\n{'='*64}\n  CROSSOVER — average F1 by tier x arm\n{'='*64}")
    header = f"{'tier':<12}" + "".join(f"{a:>12}" for a in arms)
    print(header)
    crossover = {}
    for t in tiers:
        cells = []
        for a in arms:
            f1s = [r["arms"][a]["f1"] for r in results["tasks"] if r["tier"] == t and a in r["arms"]]
            avg = round(sum(f1s) / len(f1s), 3) if f1s else None
            crossover.setdefault(t, {})[a] = avg
            cells.append(f"{avg if avg is not None else '-':>12}")
        print(f"{t:<12}" + "".join(cells))

    # total cost/time per arm
    print(f"\n  Total cost and time per arm:")
    for a in arms:
        cost = sum(r["arms"][a]["cost_usd"] for r in results["tasks"] if a in r["arms"])
        tsec = sum(r["arms"][a]["elapsed_s"] for r in results["tasks"] if a in r["arms"])
        print(f"    {a:<10} ${cost:.4f}   {tsec:.0f}s")

    results["crossover"] = crossover
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"complexity_{int(time.time())}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n  Saved to: {out_file}")


if __name__ == "__main__":
    main()
