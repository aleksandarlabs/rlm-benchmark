"""
Agent arm (paradigm 2) using the Pi CLI.

Pi runs in the target repo directory with navigation-only tools
(`bash` for grep/rg/find + `read`), solves the question by exploring the
filesystem, and emits JSONL (`--mode json`) from which we extract answer + cost.

No context is pre-loaded: it navigates the real codebase on demand, like a
production coding agent.
"""

from __future__ import annotations

import json
import re
import subprocess
import time


def _parse_pi_jsonl(stdout: str) -> dict:
    assistant_msgs = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") == "message_end":
            m = d.get("message", {})
            if m.get("role") == "assistant":
                assistant_msgs.append(m)

    def msg_text(m: dict) -> str:
        return "".join(c.get("text", "") for c in m.get("content", []) if c.get("type") == "text")

    # Respuesta = texto del ultimo mensaje del asistente con contenido (tras usar tools).
    answer = ""
    for m in assistant_msgs:
        t = msg_text(m)
        if t.strip():
            answer = t
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    # Uso/coste: sumado por mensaje del asistente (cada turno tiene su usage).
    inp = out = 0
    cost = 0.0
    per = []
    for m in assistant_msgs:
        u = m.get("usage") or {}
        i, o = u.get("input", 0), u.get("output", 0)
        c = (u.get("cost") or {}).get("total", 0.0)
        inp += i
        out += o
        cost += c
        per.append({"in": i, "out": o, "cost": c})

    # Nº de tool-calls (proxy de exploracion / turnos).
    tool_calls = sum(
        1
        for m in assistant_msgs
        for c in m.get("content", [])
        if "tool" in str(c.get("type", "")).lower()
    )

    return {
        "text": answer,
        "input_tokens": inp,
        "output_tokens": out,
        "cost_usd": round(cost, 6),
        "assistant_turns": len(assistant_msgs),
        "tool_calls": tool_calls,
        "per_message": per,
    }


def run_pi_agent(
    question: str,
    repo_dir: str,
    provider: str = "openrouter",
    model_id: str = "moonshotai/kimi-k2.6",
    thinking: str = "medium",
    tools: str = "bash,read",
    timeout: int = 900,
) -> dict:
    cmd = [
        "pi", "-p", "--mode", "json",
        "--thinking", thinking,
        "--no-session", "--no-context-files", "--no-extensions", "--no-skills",
        "--tools", tools,
        "--provider", provider, "--model", model_id,
        question,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=repo_dir, capture_output=True, text=True, timeout=timeout)
        stdout, rc, timed_out = proc.stdout, proc.returncode, False
    except subprocess.TimeoutExpired as e:
        stdout, rc, timed_out = (e.stdout or ""), -1, True
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "ignore")
    elapsed = time.time() - t0

    res = _parse_pi_jsonl(stdout)
    res["elapsed_s"] = round(elapsed, 1)
    res["returncode"] = rc
    res["timed_out"] = timed_out
    res["failed"] = timed_out or not res["text"]
    return res


if __name__ == "__main__":
    # Smoke test: a constant (lookup) task on the default corpus.
    import sys
    from pathlib import Path

    repo = str(Path(__file__).resolve().parent.parent / "corpus")
    q = sys.argv[1] if len(sys.argv) > 1 else (
        "What is the default value of output.git.sortByChangesMaxCommits in the "
        "configuration schema? Search the code and answer with the number."
    )
    print(f"repo: {repo}\nquestion: {q}\n")
    r = run_pi_agent(q, repo)
    print(f"--- answer ---\n{r['text'][:600]}\n")
    print(f"turns={r['assistant_turns']} tool_calls={r['tool_calls']} "
          f"tokens={r['input_tokens']}in/{r['output_tokens']}out "
          f"cost=${r['cost_usd']} time={r['elapsed_s']}s failed={r['failed']}")
