"""
Answer scoring.

- Needles: deterministic substring match (case-insensitive against the `accept` list).
  Not gameable, does not depend on an LLM.
- Synthesis: LLM-as-judge using a rubric, returns a 0-10 score.
"""

from __future__ import annotations

import json
import re


def score_needle(response: str, accept: list[str]) -> bool:
    if not response:
        return False
    hay = response.lower()
    return any(a.lower() in hay for a in accept)


JUDGE_PROMPT = """You are a strict evaluator. A system was asked:

QUESTION:
{question}

EVALUATION RUBRIC:
{rubric}

SYSTEM RESPONSE:
{response}

Evaluate the response ONLY according to the rubric. Return EXCLUSIVELY a valid JSON object with this shape:
{{"score": <integer 0-10>, "reason": "<one sentence>"}}
Do not add any text outside the JSON."""


def judge_synthesis(question: str, rubric: str, response: str, judge_call) -> dict:
    """judge_call(prompt:str)->str is a function that calls the judge model."""
    prompt = JUDGE_PROMPT.format(question=question, rubric=rubric, response=response or "(empty)")
    raw = judge_call(prompt)
    return _parse_judge(raw)


def _parse_judge(raw: str) -> dict:
    if not raw:
        return {"score": None, "reason": "judge returned no response"}
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            sc = d.get("score")
            return {"score": int(sc) if sc is not None else None, "reason": d.get("reason", "")}
        except Exception:
            pass
    # Fallback: models that emit malformed JSON (e.g. {"score"::int 8, ...}).
    sm = re.search(r'score"?\s*:+\s*(?:int\s+)?(\d+)', raw)
    rm = re.search(r'reason"?\s*:\s*"([^"]*)"', raw, re.DOTALL)
    if sm:
        return {"score": int(sm.group(1)), "reason": (rm.group(1) if rm else "(parse fallback)")}
    return {"score": None, "reason": f"invalid JSON: {raw[:120]}"}
