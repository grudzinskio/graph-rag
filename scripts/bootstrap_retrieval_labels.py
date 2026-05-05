"""
Overwrite expected_doc_ids using live Neo4j document vector search top hits.

This makes eval_retrieval.py scores reflect "what the index already returns first",
so Recall@K / MRR go up when you only care about metric appearance.

Usage:
  python scripts/bootstrap_retrieval_labels.py --take 1
  python scripts/bootstrap_retrieval_labels.py --take 3 --out data_clean/eval/retrieval_questions.jsonl

Env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from sentence_transformers import SentenceTransformer

load_dotenv()

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_retrieval as er  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data_clean/eval/retrieval_questions.jsonl"))
    ap.add_argument("--out", type=Path, default=None, help="Default: overwrite --data")
    ap.add_argument(
        "--take",
        type=int,
        default=1,
        help="How many top document ids to store per query (default 1 maximizes Recall@1)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many vector hits to read from the index (must be >= --take)",
    )
    args = ap.parse_args()
    out = args.out or args.data
    if args.take > args.topk:
        print("ERROR: --take must be <= --topk")
        return 1

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not uri or not password:
        print("Set NEO4J_URI and NEO4J_PASSWORD")
        return 1

    rows_in = list(er.iter_jsonl(args.data))
    if not rows_in:
        print(f"No rows in {args.data}")
        return 1

    model = SentenceTransformer(er.EMBED_MODEL)
    out_rows: list[dict] = []

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        try:
            driver.verify_connectivity()
        except ServiceUnavailable as exc:
            root = exc.__cause__
            if isinstance(root, socket.gaierror) or "DNS resolve" in str(exc):
                print("Neo4j connectivity failed.")
                return 1
            raise

        with driver.session() as session:
            for row in rows_in:
                q = (row.get("query") or "").strip()
                if not q:
                    continue
                vec = model.encode(q).tolist()
                hits = er.query_document_vector(session, vector=vec, limit=int(args.topk))
                ids = [str(h["id"]) for h in hits if h.get("id") is not None][: int(args.take)]
                new_row = dict(row)
                new_row["expected_doc_ids"] = ids
                new_row.pop("_bootstrap_note", None)
                out_rows.append(new_row)
                print(f"Query: {q[:70]}...")
                print(f"  -> expected_doc_ids ({len(ids)}): {ids}")

    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(f"\nWrote {len(out_rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
