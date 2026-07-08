"""
Set-based scoring for the linear/quadratic tasks of Experiment 2.

- CLI flags and file paths are EXTRACTED from the response text with regex and
  compared against a frozen reference_set -> precision / recall / F1.
- Config defaults: coverage = fraction of (key, value) pairs from reference_map
  that appear correctly in the response.

Set F1 is objective and not gameable: the model's prose does not matter,
only which items it got right / missed / hallucinated.
"""

from __future__ import annotations

import re


def _f1(pred: set[str], ref: set[str]) -> dict:
    tp = len(pred & ref)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(ref) if ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "f1": round(f1, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "tp": tp,
        "pred_n": len(pred),
        "ref_n": len(ref),
        "missing": sorted(ref - pred)[:10],   # sample of omitted items
        "extra": sorted(pred - ref)[:10],     # sample of hallucinated items
    }


# --- extractors ---

def extract_flags(text: str) -> set[str]:
    return {f.lower() for f in re.findall(r"--[a-zA-Z][a-zA-Z0-9-]+", text or "")}


def _norm_path(p: str, strip_prefix: str | None) -> str:
    p = p.strip().lstrip("./").strip("`'\"")
    if strip_prefix and p.startswith(strip_prefix):
        p = p[len(strip_prefix):]
    return p


def extract_paths(text: str, strip_prefix: str | None = None) -> set[str]:
    raw = re.findall(r"[\w./-]+\.ts", text or "")
    return {_norm_path(p, strip_prefix) for p in raw if not p.endswith(".d.ts")}


# --- scorers by type ---

def score_set_flags(response: str, reference: list[str]) -> dict:
    return _f1(extract_flags(response), set(reference))


def score_set_paths(response: str, reference: list[str], strip_prefix: str | None = None) -> dict:
    # Reference is already normalized (relative to src/ or tests/).
    return _f1(extract_paths(response, strip_prefix), set(reference))


def _value_str(v) -> list[str]:
    if isinstance(v, bool):
        return ["true" if v else "false"]
    if isinstance(v, (int, float)):
        # accept the raw number and common variants (e.g. 50MB for maxFileSize)
        s = str(v)
        out = [s]
        if v == 52428800:
            out += ["50mb", "50 mb", "50*1024*1024", "50 * 1024 * 1024"]
        return out
    return [str(v)]


def score_coverage_defaults(response: str, reference_map: dict) -> dict:
    text = (response or "").lower()
    correct = 0
    missing = []
    for dotted, val in reference_map.items():
        key = dotted.split(".")[-1].lower()  # models usually cite the leaf, not the full path
        kpos = text.find(key)
        ok = False
        if kpos != -1:
            window = text[kpos: kpos + 120]
            ok = any(vs.lower() in window for vs in _value_str(val))
        if ok:
            correct += 1
        else:
            missing.append(dotted)
    total = len(reference_map)
    return {
        "f1": round(correct / total, 3) if total else 0.0,  # coverage as the main metric
        "coverage": round(correct / total, 3) if total else 0.0,
        "correct": correct,
        "ref_n": total,
        "missing": missing[:10],
    }


def score_task(task: dict, response: str) -> dict:
    """Dispatch based on task['score'] or task['tier']/'accept'."""
    kind = task.get("score")
    if task.get("accept") is not None:  # constant
        from scoring import score_needle
        ok = score_needle(response, task["accept"])
        return {"f1": 1.0 if ok else 0.0, "correct": ok}
    if kind == "set_f1_flags":
        return score_set_flags(response, task["reference_set"])
    if kind == "set_f1_paths":
        prefix = "src/" if task["id"].startswith("Q1") else ("tests/" if task["id"].startswith("Q2") else None)
        return score_set_paths(response, task["reference_set"], prefix)
    if kind == "coverage_defaults":
        return score_coverage_defaults(response, task["reference_map"])
    raise ValueError(f"unknown score for {task.get('id')}: {kind}")
