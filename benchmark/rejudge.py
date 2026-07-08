#!/usr/bin/env python3
"""
Re-score synthesis questions from already saved results WITHOUT regenerating
answers (only calls the judge). Useful when the judge failed during the original
run. Updates JSON files in-place and recomputes synthesis_avg_score.

    python benchmark/rejudge.py                # re-judge all results/*.json
    python benchmark/rejudge.py rlm-all-small  # only one condition

Judge: config.JUDGE_MODEL (default gpt-5-mini, independent from experiment models).

Note: re-judges the saved response (truncated to ~2500 chars on save), which is
enough for the coverage rubric.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import openai

import config as C
import scoring

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def judge_call(prompt: str) -> str:
    m = C.MODELS[C.JUDGE_MODEL]
    client = openai.OpenAI(api_key=C.resolve_api_key(C.JUDGE_MODEL), base_url=m["base_url"])
    resp = client.chat.completions.create(
        model=m["model_id"],
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=C.JUDGE_MAX_OUTPUT_TOKENS,
        extra_body=C.provider_body(C.JUDGE_MODEL),
    )
    return resp.choices[0].message.content or ""


def rejudge_file(path: Path) -> None:
    data = json.loads(path.read_text())
    syn = data.get("synthesis", [])
    if not syn:
        return
    print(f"\n{path.name}  ({data['condition']})")
    for item in syn:
        if item.get("failed"):
            print(f"  {item['id']}: RLM failed, skipping")
            continue
        verdict = scoring.judge_synthesis(
            item["question"], item["rubric"], item.get("response", ""), judge_call
        )
        if verdict["score"] is None:  # retry on transient judge glitch
            verdict = scoring.judge_synthesis(
                item["question"], item["rubric"], item.get("response", ""), judge_call
            )
        item["judge"] = verdict
        print(f"  {item['id']}: score={verdict['score']}/10 — {verdict['reason'][:90]}")

    scores = [s["judge"]["score"] for s in syn if s.get("judge", {}).get("score") is not None]
    avg = round(sum(scores) / len(scores), 2) if scores else None
    data.setdefault("summary", {})["synthesis_avg_score"] = avg
    data["summary"]["judge_model"] = C.MODELS[C.JUDGE_MODEL]["model_id"]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    filt = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(glob.glob(str(RESULTS_DIR / "*.json")))
    if filt:
        files = [f for f in files if filt in Path(f).name]
    if not files:
        print("No results to re-judge.")
        return
    print(f"Judge: {C.MODELS[C.JUDGE_MODEL]['model_id']}")
    for f in files:
        rejudge_file(Path(f))
    print("\nDone. Run  .venv/bin/python benchmark/compare.py  to see the table.")


if __name__ == "__main__":
    main()
