"""
GraphRAG retrieval evaluation (document-level) for your MSOE Neo4j index.

This measures whether the vector search over `document_embeddings` returns
the expected document ids for a set of queries.

Usage:
  python scripts/eval_retrieval.py --data data_clean/eval/retrieval_questions.jsonl --k 1 3 5 10

Input JSONL format (one per line):
  {
    "query": "...",
    "expected_doc_ids": ["123", "456"]
  }
Optionally:
  {
    "query": "...",
    "expected_doc_ids": ["123"],
    "expected_entities": ["mechanical engineering", "capstone"]
  }
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
VECTOR_DIMS = 384


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def recall_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    if not expected:
        return 0.0
    return 1.0 if any(x in expected for x in ranked[:k]) else 0.0


def mrr(ranked: list[str], expected: set[str]) -> float:
    for i, x in enumerate(ranked, start=1):
        if x in expected:
            return 1.0 / i
    return 0.0


def query_document_vector(session, *, vector: list[float], limit: int) -> list[dict]:
    """
    Prefer Neo4j SEARCH clause (newer); fall back to deprecated procedure.
    Tries multiple SEARCH forms because Neo4j versions differ in how they accept parameters.
    """
    search_variants = [
        # Variant A: allow LIST query vector directly
        (
            """
            MATCH (d:Document)
              SEARCH d IN (
                VECTOR INDEX document_embeddings
                FOR $vector
                LIMIT $limit
              ) SCORE AS score
            RETURN d.id AS id, score
            ORDER BY score DESC
            """,
            {"vector": vector, "limit": limit},
        ),
        # Variant B: typed vector() wrapper (newer Cypher)
        (
            """
            MATCH (d:Document)
              SEARCH d IN (
                VECTOR INDEX document_embeddings
                FOR vector($vector, $dims, FLOAT)
                LIMIT $limit
              ) SCORE AS score
            RETURN d.id AS id, score
            ORDER BY score DESC
            """,
            {"vector": vector, "dims": VECTOR_DIMS, "limit": limit},
        ),
    ]

    last_exc: Exception | None = None
    for q, params in search_variants:
        try:
            return session.run(q, **params).data()
        except Exception as exc:  # pragma: no cover
            last_exc = exc

    # Backward-compatible fallback (deprecated in newer Neo4j)
    try:
        return session.run(
            """
            CALL db.index.vector.queryNodes('document_embeddings', $k, $vector)
            YIELD node AS d, score
            RETURN d.id AS id, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            vector=vector,
            k=limit,
            limit=limit,
        ).data()
    except Exception as exc:  # pragma: no cover
        if last_exc is not None:
            raise last_exc
        raise exc


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Neo4j document vector retrieval")
    ap.add_argument("--data", type=Path, default=Path("data_clean/eval/retrieval_questions.jsonl"))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--topk", type=int, default=10, help="How many docs to retrieve per query")
    ap.add_argument(
        "--show-misses",
        type=int,
        default=5,
        help="Print top hits for the first N queries with Recall@maxK == 0",
    )
    args = ap.parse_args()

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not uri or not password:
        print("Set NEO4J_URI and NEO4J_PASSWORD in the environment or .env")
        return 1

    if not args.data.exists():
        print(f"ERROR: eval dataset not found: {args.data}")
        return 1

    print(f"Loading embedding model ({EMBED_MODEL})...")
    model = SentenceTransformer(EMBED_MODEL)

    rows = list(iter_jsonl(args.data))
    if not rows:
        print(f"ERROR: no rows in {args.data}")
        return 1

    ks = sorted(set(int(x) for x in args.k if int(x) > 0))
    topk = max(args.topk, max(ks))

    r_at = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    n = 0
    shown = 0

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            for row in rows:
                q = (row.get("query") or "").strip()
                exp = set(str(x) for x in (row.get("expected_doc_ids") or []))
                if not q or not exp:
                    continue
                vec = model.encode(q).tolist()
                hits = query_document_vector(session, vector=vec, limit=topk)
                ranked = [str(h["id"]) for h in hits if h.get("id") is not None]
                if not ranked:
                    continue
                n += 1
                for k in ks:
                    r_at[k] += recall_at_k(ranked, exp, k)
                mrr_sum += mrr(ranked, exp)

                if shown < int(args.show_misses):
                    maxk = max(ks)
                    if recall_at_k(ranked, exp, maxk) == 0.0:
                        shown += 1
                        print("\n--- MISS ---")
                        print("Query:", q)
                        print("Expected doc ids:", sorted(exp))
                        print("Top hits:", ranked[: min(10, len(ranked))])

    if n == 0:
        print("No valid evaluation rows (need query + expected_doc_ids).")
        return 1

    print("\n=== Retrieval Eval (Document Vector Search) ===")
    print(f"Queries evaluated: {n}")
    for k in ks:
        print(f"Recall@{k}: {r_at[k] / n:.4f}")
    print(f"MRR: {mrr_sum / n:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

