"""
GraphRAG retrieval evaluation for your MSOE Neo4j index.

Modes:
  - document: `document_embeddings` on :Document (default; same as before)
  - chunk_doc: `chunk_embeddings` on :Chunk, then rank documents by best chunk
    score per doc_id (closer to chunk-first GraphRAG retrieval)
  - both: run and print both blocks

Usage:
  python scripts/eval_retrieval.py --data data_clean/eval/retrieval_questions.jsonl --k 1 3 5 10
  python scripts/eval_retrieval.py --data data_clean/eval/retrieval_questions_many.jsonl --mode both

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
import socket
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
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


def query_chunk_vector(session, *, vector: list[float], limit: int) -> list[dict]:
    """Search chunk_embeddings; same fallbacks as graph_rag_query / probe_retrieval_gold."""
    search_variants = [
        (
            """
            MATCH (c:Chunk)
              SEARCH c IN (
                VECTOR INDEX chunk_embeddings
                FOR $vector
                LIMIT $limit
              ) SCORE AS score
            RETURN c.doc_id AS doc_id, score
            ORDER BY score DESC
            """,
            {"vector": vector, "limit": limit},
        ),
        (
            """
            MATCH (c:Chunk)
              SEARCH c IN (
                VECTOR INDEX chunk_embeddings
                FOR vector($vector, $dims, FLOAT)
                LIMIT $limit
              ) SCORE AS score
            RETURN c.doc_id AS doc_id, score
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
    try:
        return session.run(
            """
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $vector)
            YIELD node AS c, score
            RETURN c.doc_id AS doc_id, score
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


def rank_docs_from_chunk_hits(rows: list[dict]) -> list[str]:
    """Best chunk score per doc_id; sort by score DESC, then doc_id for stability."""
    best: dict[str, float] = {}
    for r in rows:
        did = r.get("doc_id")
        if did is None:
            continue
        did = str(did)
        sc = float(r.get("score") or 0.0)
        if did not in best or sc > best[did]:
            best[did] = sc
    return sorted(best.keys(), key=lambda d: (-best[d], d))


def eval_pass(
    session,
    model,
    rows: list[dict],
    ks: list[int],
    ranked_limit: int,
    show_misses: int,
    title: str,
    get_ranked,
) -> tuple[int, dict[int, float], float]:
    """get_ranked(session, query_vector) -> ordered doc id list."""
    r_at = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    n = 0
    shown = 0
    max_k = max(ks)
    for row in rows:
        q = (row.get("query") or "").strip()
        exp = set(str(x) for x in (row.get("expected_doc_ids") or []))
        if not q or not exp:
            continue
        vec = model.encode(q).tolist()
        ranked = get_ranked(session, vec)
        ranked = ranked[:ranked_limit]
        if not ranked:
            continue
        n += 1
        for k in ks:
            r_at[k] += recall_at_k(ranked, exp, k)
        mrr_sum += mrr(ranked, exp)

        if shown < int(show_misses) and recall_at_k(ranked, exp, max_k) == 0.0:
            shown += 1
            print("\n--- MISS ---")
            print(f"[{title}]")
            print("Query:", q)
            print("Expected doc ids:", sorted(exp))
            print("Top hits:", ranked[: min(10, len(ranked))])

    return n, r_at, mrr_sum


def print_block(title: str, n: int, ks: list[int], r_at: dict[int, float], mrr_sum: float) -> None:
    print(f"\n=== {title} ===")
    print(f"Queries evaluated: {n}")
    for k in ks:
        print(f"Recall@{k}: {r_at[k] / n:.4f}")
    print(f"MRR: {mrr_sum / n:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Neo4j document / chunk retrieval")
    ap.add_argument("--data", type=Path, default=Path("data_clean/eval/retrieval_questions.jsonl"))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--topk", type=int, default=10, help="How many ranked docs to keep per query (for misses display)")
    ap.add_argument(
        "--mode",
        choices=("document", "chunk_doc", "both"),
        default="document",
        help="document=Document index (default); chunk_doc=Chunk index aggregated by doc; both=print two blocks",
    )
    ap.add_argument(
        "--chunk-scan",
        type=int,
        default=2500,
        help="How many chunk hits to pull before aggregating by doc_id (chunk_doc / both)",
    )
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
    ranked_limit = max(args.topk, max(ks))
    chunk_scan = int(args.chunk_scan)

    def get_ranked_document(session, vec: list[float]) -> list[str]:
        hits = query_document_vector(session, vector=vec, limit=ranked_limit)
        return [str(h["id"]) for h in hits if h.get("id") is not None]

    def get_ranked_chunk_doc(session, vec: list[float]) -> list[str]:
        ch = query_chunk_vector(session, vector=vec, limit=chunk_scan)
        return rank_docs_from_chunk_hits(ch)[:ranked_limit]

    runs: list[tuple[str, object]] = []
    if args.mode in ("document", "both"):
        runs.append(("Retrieval Eval (Document vector index)", get_ranked_document))
    if args.mode in ("chunk_doc", "both"):
        runs.append(
            (
                f"Retrieval Eval (Chunk->doc: best chunk per doc, scan {chunk_scan} chunks)",
                get_ranked_chunk_doc,
            )
        )

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        try:
            driver.verify_connectivity()
        except ServiceUnavailable as exc:
            # Common on Windows when DNS/VPN/campus network blocks Aura resolution.
            root = exc.__cause__
            if isinstance(root, socket.gaierror) or "DNS resolve" in str(exc):
                print("\nERROR: Neo4j connection failed due to DNS resolution.")
                print(f"NEO4J_URI={uri!r}")
                print("Fixes to try:")
                print("- Ensure you have internet access and DNS is working.")
                print("- If on a restricted network, try a VPN or different network.")
                print("- For Neo4j Aura, use a TLS URI, e.g.:")
                print("  - neo4j+s://<id>.databases.neo4j.io")
                print("  - bolt+s://<id>.databases.neo4j.io:7687")
                print("- Verify the hostname resolves (PowerShell): nslookup <id>.databases.neo4j.io")
                return 1
            raise

        any_n = 0
        for title, get_ranked in runs:
            with driver.session() as session:
                n, r_at, mrr_sum = eval_pass(
                    session,
                    model,
                    rows,
                    ks,
                    ranked_limit,
                    int(args.show_misses),
                    title,
                    get_ranked,
                )
            if n == 0:
                print(f"No valid evaluation rows for [{title}] (need query + expected_doc_ids, or empty retrieval).")
                continue
            any_n = max(any_n, n)
            print_block(title, n, ks, r_at, mrr_sum)

    if any_n == 0:
        print("No valid evaluation rows (need query + expected_doc_ids).")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

