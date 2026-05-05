"""
Build a larger retrieval eval JSONL from the local MSOE corpus.

For each sampled document, uses the first line of cleaned text (usually
"Title | MSOE") as the query and sets expected_doc_ids to that document's id.
This is a **self-retrieval** check: "does the vector index return this page when
asked with its own title line?"

Usage:
  python scripts/generate_retrieval_eval_questions.py --n 50
  python scripts/generate_retrieval_eval_questions.py --n 100 --out data_clean/eval/retrieval_questions_many.jsonl
  python scripts/eval_retrieval.py --data data_clean/eval/retrieval_questions_many.jsonl --k 1 3 5 10

Env (for eval only): NEO4J_URI, NEO4J_PASSWORD
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DOCS = _ROOT / "data_clean/msoe/documents.jsonl"
_DEFAULT_OUT = _ROOT / "data_clean/eval/retrieval_questions_many.jsonl"

_WS = re.compile(r"\s+")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def first_line_query(text: str, max_len: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    line = t.splitlines()[0].strip()
    line = _WS.sub(" ", line)
    if len(line) > max_len:
        line = line[:max_len].rsplit(" ", 1)[0] or line[:max_len]
    return line


def reservoir_sample(rng: random.Random, path: Path, k: int, *, min_query_len: int, max_query_len: int):
    """Sample up to k documents that yield a usable query string (reservoir over eligible rows only)."""
    pool: list[dict] = []
    seen_eligible = 0
    for rec in iter_jsonl(path):
        q = first_line_query(rec.get("text") or "", max_query_len)
        if len(q) < min_query_len:
            continue
        seen_eligible += 1
        item = {"rec": rec, "query": q}
        if len(pool) < k:
            pool.append(item)
        else:
            j = rng.randint(0, seen_eligible - 1)
            if j < k:
                pool[j] = item
    return pool


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate retrieval_questions JSONL from documents.jsonl")
    ap.add_argument("--docs", type=Path, default=_DEFAULT_DOCS, help="Source corpus JSONL")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="Output eval JSONL path")
    ap.add_argument("--n", type=int, default=50, help="How many questions to generate")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    ap.add_argument("--min-query-len", type=int, default=24, help="Skip docs whose first line is shorter")
    ap.add_argument("--max-query-len", type=int, default=220, help="Trim first line to this length")
    args = ap.parse_args()

    if not args.docs.exists():
        print(f"ERROR: missing {args.docs}")
        return 1

    rng = random.Random(int(args.seed))
    picked = reservoir_sample(
        rng,
        args.docs,
        int(args.n),
        min_query_len=int(args.min_query_len),
        max_query_len=int(args.max_query_len),
    )
    if len(picked) < int(args.n):
        print(
            f"Warning: only {len(picked)} usable docs after filters "
            f"(wanted {args.n}). Try lowering --min-query-len or increasing corpus size."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for item in picked:
            rec = item["rec"]
            q = item["query"]
            did = rec.get("id")
            if not did:
                continue
            row = {
                "query": q,
                "expected_doc_ids": [str(did)],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows -> {args.out}")
    rel = args.out
    try:
        rel = args.out.relative_to(_ROOT)
    except ValueError:
        pass
    print("Next:")
    print(f"  python scripts/eval_retrieval.py --data {rel} --k 1 3 5 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
