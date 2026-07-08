#!/usr/bin/env python3
"""
Aggregate the latest result of each condition into a comparison table.

    python benchmark/compare.py

Reads results/*.json and shows the most recent run per condition.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def latest_per_condition() -> dict[str, dict]:
    latest: dict[str, tuple[float, dict]] = {}
    for f in RESULTS_DIR.glob("*.json"):
        data = json.loads(f.read_text())
        cond = data.get("condition", f.stem)
        ts = f.stat().st_mtime
        if cond not in latest or ts > latest[cond][0]:
            latest[cond] = (ts, data)
    return {k: v[1] for k, v in latest.items()}


def main():
    rows = latest_per_condition()
    if not rows:
        print("No results in", RESULTS_DIR)
        return

    order = ["vanilla-small", "vanilla-strong", "rlm-strong-subs", "rlm-all-small"]
    conds = [c for c in order if c in rows] + [c for c in rows if c not in order]

    header = f"{'condition':<18} {'needles':>9} {'synthesis':>9} {'failures':>8} {'cost $':>10} {'time s':>8}"
    print("\n" + header)
    print("-" * len(header))
    for c in conds:
        s = rows[c].get("summary", {})
        needles = f"{s.get('needles_correct','?')}/{s.get('needles_total','?')}"
        syn = s.get("synthesis_avg_score")
        syn = f"{syn}/10" if syn is not None else "-"
        print(f"{c:<18} {needles:>9} {syn:>9} {s.get('rlm_failures','?'):>8} "
              f"{s.get('total_cost_usd',0):>10.4f} {s.get('total_time_s',0):>8.1f}")
    print()

    # Cost breakdown per model for RLM conditions.
    for c in conds:
        if not rows[c].get("mode") == "rlm":
            continue
        agg: dict[str, dict] = {}
        for item in rows[c]["needles"] + rows[c]["synthesis"]:
            for model_id, u in item.get("usage_by_model", {}).items():
                a = agg.setdefault(model_id, {"in": 0, "out": 0, "cost": 0.0})
                a["in"] += u.get("input_tokens", 0)
                a["out"] += u.get("output_tokens", 0)
                a["cost"] += u.get("cost_usd", 0.0)
        print(f"[{c}] cost breakdown by model:")
        for model_id, a in agg.items():
            print(f"   {model_id:<40} in={a['in']:>10,} out={a['out']:>8,} ${a['cost']:.4f}")
        print()


if __name__ == "__main__":
    main()
