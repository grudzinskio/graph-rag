"""
Offline helper: suggest `Document.id` values for eval questions.

If you don't have Neo4j access, you can still build `expected_doc_ids`
from the local source of truth `data_clean/msoe/documents.jsonl`.

Scoring is simple token overlap (fast, dependency-free). It’s not a model;
it’s just a practical way to find likely supporting docs.

Usage:
  python scripts/suggest_expected_docs.py --query "mechanical engineering program" --top 10
  python scripts/suggest_expected_docs.py --data data_clean/eval/retrieval_questions.jsonl --top 5
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DOCS_PATH = Path("data_clean/msoe/documents.jsonl")

_TOK_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "is",
    "are",
    "be",
    "as",
    "that",
    "this",
    "it",
    "you",
    "your",
}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in _TOK_RE.findall(text or "")]
    return [t for t in toks if t not in _STOP and len(t) > 1]


def score_query_doc(q_toks: Counter[str], doc_text: str) -> float:
    d = Counter(tokenize(doc_text))
    if not d:
        return 0.0
    overlap = sum(min(q_toks[t], d.get(t, 0)) for t in q_toks.keys())
    # Light length normalization so huge pages don't always win.
    denom = (sum(d.values()) ** 0.5) or 1.0
    return overlap / denom


def top_docs_for_query(query: str, top_n: int) -> list[dict]:
    q_toks = Counter(tokenize(query))
    if not q_toks:
        return []
    best: list[tuple[float, dict]] = []
    for rec in iter_jsonl(DOCS_PATH):
        sc = score_query_doc(q_toks, rec.get("text") or "")
        if sc <= 0:
            continue
        best.append((sc, rec))
    best.sort(key=lambda x: x[0], reverse=True)
    out = []
    for sc, rec in best[:top_n]:
        out.append(
            {
                "score": round(float(sc), 6),
                "id": rec.get("id"),
                "source_path": rec.get("source_path"),
                "title_preview": (rec.get("text") or "").splitlines()[0][:120],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Suggest expected_doc_ids from local documents.jsonl")
    ap.add_argument("--query", type=str, default=None, help="Single query string")
    ap.add_argument("--data", type=Path, default=None, help="JSONL with {query,...} rows; prints suggestions per row")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    if not DOCS_PATH.exists():
        print(f"ERROR: missing {DOCS_PATH}")
        return 1

    if args.query:
        hits = top_docs_for_query(args.query, args.top)
        print(json.dumps({"query": args.query, "suggested": hits}, ensure_ascii=False, indent=2))
        return 0

    if args.data:
        for row in iter_jsonl(args.data):
            q = (row.get("query") or "").strip()
            if not q:
                continue
            hits = top_docs_for_query(q, args.top)
            print(json.dumps({"query": q, "suggested": hits}, ensure_ascii=False))
        return 0

    print("Provide --query or --data")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

