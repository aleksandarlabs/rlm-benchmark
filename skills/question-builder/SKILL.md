---
name: question-builder
description: Generate benchmark questions and ground truth for the RLM vs. terminal-agent benchmark from any codebase placed in ./corpus.
version: 1.0.0
author: AleksandarLabs
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [rlm, benchmark, question-generation, ground-truth, coding-agents]
    related_skills: []
---

# RLM Benchmark Question Builder

## Overview

This skill turns any codebase into a benchmark dataset for comparing three context-retrieval paradigms:

1. **Vanilla** — dump as much code as fits into the prompt.
2. **Terminal agent** — give the model `bash`/`read` tools to explore the repo.
3. **RLM** — load the context as a REPL variable and let the model navigate it programmatically.

Your job is to explore the codebase in `./corpus`, identify verifiable facts at three complexity levels, and write two JSON files:

- `questions/questions.json` — for **Experiment 1** (needles + synthesis).
- `questions/questions-complexity.json` — for **Experiment 2** (constant / linear / quadratic tiers).

## When to use

- A user has placed a codebase in `./corpus` and wants to benchmark RLM, vanilla, and a terminal agent on it.
- The user asks you to prepare questions, ground truth, or a judge pipeline for the benchmark.

## Output files

### 1. `questions/questions.json`

Top-level shape:

```json
{
  "needles": [...],
  "synthesis": [...]
}
```

**Needles** (constant complexity) — one isolated fact per question.

```json
{
  "id": "n01-default-sort-by",
  "question": "What is the default value of output.git.sortByChangesMaxCommits in the configuration schema?",
  "accept": ["10"]
}
```

Rules:

- The answer must be a single value or a small set of values.
- `accept` contains exact strings the model answer must include to be marked correct.
- The fact must be locatable in 1-2 files (needle in a haystack).

**Synthesis** (linear / light quadratic) — a question that requires summarising or comparing a bounded set of files.

```json
{
  "id": "s01-cli-flags",
  "question": "List all CLI flags accepted by the main entry point.",
  "rubric": "Score 0-10. Award 10 only if every flag from src/cli.ts is listed with no hallucinations. Deduct 1 point per missing flag and 2 points per invented flag."
}
```

Rules:

- The rubric must be concrete enough for an LLM-as-judge to apply deterministically.
- Prefer questions whose ideal answer is a list or a short paragraph, not open-ended prose.

### 2. `questions/questions-complexity.json`

Top-level shape:

```json
{
  "constant": [...],
  "linear": [...],
  "quadratic": [...]
}
```

Each task:

```json
{
  "id": "c01-where-is-x-defined",
  "tier": "constant",
  "question": "In which file is function `parseConfig` defined?",
  "ground_truth": ["src/config.ts"],
  "verifier": "set-equality"
}
```

| Tier | Definition | Verifier |
|---|---|---|
| `constant` | One isolated fact. | `set-equality` or `contains` |
| `linear` | Requires scanning most files of one type. | `set-equality` with a list |
| `quadratic` | Requires crossing two sets against each other. | `set-equality` with computed ground truth |

For quadratic tasks, include a `script_ground_truth` field with a tiny shell/Python snippet that computes the correct answer mechanically. Example:

```json
{
  "id": "q01-files-without-tests",
  "tier": "quadratic",
  "question": "Which .py files under src/ do not have a corresponding test file under tests/?",
  "ground_truth": ["src/a.py", "src/b.py"],
  "verifier": "set-equality",
  "script_ground_truth": "comm -23 <(find src -name '*.py' | sort) <(find tests -name 'test_*.py' | sed 's#tests/test_#src/#' | sort)"
}
```

## Workflow

1. **Read the corpus.**
   - Run `python benchmark/corpus.py` to see file/token counts.
   - Skim the top-level `README.md`, `package.json`, `pyproject.toml`, or equivalent to understand the project.

2. **Identify candidate facts.**
   - Look for configuration defaults, exported function names, CLI flags, dependency lists, error codes, etc.
   - For each candidate, decide its tier: constant (one file), linear (scan many files), quadratic (cross two sets).

3. **Write ground truth mechanically.**
   - For `constant` and `linear`, use `grep`, `find`, or a short script to build the exact answer.
   - For `quadratic`, always include a `script_ground_truth` snippet.

4. **Create the JSON files.**
   - Aim for at least 5 needles, 2 synthesis questions, and 2 tasks per complexity tier.
   - Save to `questions/questions.json` and `questions/questions-complexity.json`.

5. **Validate.**
   - Run `python benchmark/corpus.py`.
   - Run a cheap smoke test: `python benchmark/run_bench.py --condition vanilla-small --set needles --limit 1`.
   - Run a complexity smoke test: `python benchmark/run_complexity.py --tiers constant --limit-per-tier 1`.

## Complexity-tier guidelines

### Constant

- The answer lives in one or two files.
- Example: "What is the default port in `src/server.ts`?"

### Linear

- The answer is a list built by scanning many files of the same kind.
- Example: "List all middleware registered in the Express app."

### Quadratic

- The answer requires comparing two sets.
- Example: "Which public functions in `src/` are never called in `tests/`?"
- Always provide a mechanical verification script.

## Judge role

If the user asks you to act as a judge for synthesis answers:

1. Read the question and its rubric.
2. Read the model response.
3. Score 0-10 and explain every deduction in one sentence.
4. Return JSON:

```json
{
  "score": 8,
  "reason": "Missing --verbose flag (-1); invented --dry-run flag (-2)."
}
```

## Common pitfalls

1. **Questions that require external knowledge.** Every answer must be derivable from `./corpus` alone.
2. **Vague rubrics.** "Good answer" is not a rubric. Use countable criteria.
3. **Answers that change with every commit.** Pick stable architectural facts, not transient values.
4. **Quadratic tasks that are actually linear.** If the answer can be found by grepping one keyword, it is constant or linear, not quadratic.
5. **Ground truth written by eye.** Always verify with a script or a precise grep.

## Verification checklist

- [ ] `questions/questions.json` exists and contains at least 5 needles and 2 synthesis questions.
- [ ] `questions/questions-complexity.json` exists and contains at least 2 tasks per tier.
- [ ] Every needle has a non-empty `accept` list.
- [ ] Every synthesis question has a concrete rubric.
- [ ] Every quadratic task has a `script_ground_truth` snippet.
- [ ] `python benchmark/corpus.py` runs without errors.
- [ ] A `--limit 1` smoke test of `run_bench.py` and `run_complexity.py` completes.
