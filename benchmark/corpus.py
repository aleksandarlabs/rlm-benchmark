"""
Context collector for the target codebase.

Produces two representations of the same content:
  - dict {relative_path: content}  -> passed to RLM as `context`
  - concatenated string with markers -> passed to vanilla in a single prompt

Caches the result to avoid re-reading large codebases on every run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def _ext_set(env_val: str | None, default: set[str]) -> set[str]:
    if not env_val:
        return default
    return {e if e.startswith(".") else f".{e}" for e in env_val.split(",") if e.strip()}


# Files included in the context. Override with BENCH_INCLUDE_EXT="py,md,json,jsonl".
INCLUDE_EXT = _ext_set(
    os.environ.get("BENCH_INCLUDE_EXT"),
    {
        ".ts",
        ".js",
        ".tsx",
        ".jsx",
        ".mjs",
        ".cjs",
        ".py",
        ".md",
        ".json",
        ".yml",
        ".yaml",
        ".sh",
        ".toml",
        ".cfg",
    },
)

# Noisy directories that do not help understand the codebase.
SKIP_DIRS = {
    ".git",
    "node_modules",
    "dist",
    "lib",
    "coverage",
    ".venv",
    "venv",
    "__pycache__",
    ".cache",
    "build",
    "out",
    ".turbo",
    ".next",
}

# Individual files to skip (huge lockfiles with little semantic value).
SKIP_FILES = {"pnpm-lock.yaml", "package-lock.json", "yarn.lock", "bun.lockb"}

MAX_FILE_CHARS = int(os.environ.get("BENCH_MAX_FILE_CHARS", "200000"))  # truncate giant files

# Total context token cap (0 = unlimited). Useful for capping huge repos.
MAX_TOTAL_TOKENS = int(os.environ.get("BENCH_MAX_TOTAL_TOKENS", "0"))

CHARS_PER_TOKEN_TRUNCATE = 3.4


def est_tokens(chars: int) -> int:
    """Cheap estimate: ~4 chars per token."""
    return chars // 4


def _cache_file(repo_dir: Path, cache_dir: Path) -> Path:
    repo_dir = repo_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{repo_dir.name}_context.json"


def build_context(
    repo_dir: Path | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict[str, str]:
    """Return {path: content}. Uses cache unless force=True."""
    repo_dir = Path(repo_dir or os.environ.get("BENCH_REPO_DIR") or DEFAULT_CORPUS_DIR)
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_file = _cache_file(repo_dir, cache_dir)

    if cache_file.exists() and not force:
        return json.loads(cache_file.read_text())

    if not repo_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {repo_dir}\n"
            "Place a codebase in ./corpus/ or pass --corpus-dir."
        )

    files: dict[str, str] = {}
    total_chars = 0
    cap_chars = MAX_TOTAL_TOKENS * 4 if MAX_TOTAL_TOKENS else 0
    for path in sorted(repo_dir.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.name in SKIP_FILES:
            continue
        if path.suffix not in INCLUDE_EXT:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n... [TRUNCATED]"
        rel = str(path.relative_to(repo_dir))
        files[rel] = content
        total_chars += len(content)
        if cap_chars and total_chars >= cap_chars:
            break

    cache_file.write_text(json.dumps(files, ensure_ascii=False))
    return files


def as_flat_string(files: dict[str, str]) -> str:
    """Concatenate the dict into a single string with markers (for vanilla)."""
    return "\n\n".join(f"=== FILE: {rel} ===\n{content}" for rel, content in files.items())


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    """Truncate `text` to fit in `max_tokens`. Returns (text, was_truncated).

    Uses a conservative chars/token ratio to avoid exceeding the real window.
    """
    max_chars = int(max_tokens * CHARS_PER_TOKEN_TRUNCATE)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n\n... [CONTEXT TRUNCATED TO FIT WINDOW]", True


def stats(files: dict[str, str]) -> dict:
    total_chars = sum(len(c) for c in files.values())
    return {
        "files": len(files),
        "chars": total_chars,
        "est_tokens": est_tokens(total_chars),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the token size of a codebase corpus.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Path to the codebase to benchmark (default: ./corpus).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for the context cache (default: benchmark/.cache).",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild the context cache.")
    args = parser.parse_args()

    files = build_context(repo_dir=args.corpus_dir, cache_dir=args.cache_dir, force=args.force)
    s = stats(files)
    print(f"Repo: {args.corpus_dir.resolve()}")
    print(f"Files: {s['files']}")
    print(f"Chars: {s['chars']:,}")
    print(f"Estimated tokens: {s['est_tokens']:,}")
    print(f"Cache: {_cache_file(args.corpus_dir, args.cache_dir)}")
    top = sorted(files.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
    print("\nTop 10 files by size:")
    for rel, content in top:
        print(f"  {est_tokens(len(content)):>7,} tok  {rel}")


if __name__ == "__main__":
    main()
