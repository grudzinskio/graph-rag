"""
Probe Neo4j retrieval vs gold doc ids in retrieval_questions.jsonl.

Compares:
  - Document-level vector index (`document_embeddings`) — same as eval_retrieval.py
  - Chunk-level index (`chunk_embeddings`) with best score per doc_id (matches GraphRAG chunk path)

For each expected Document id, prints:
  - Whether the node exists and size(d.text) (explains chunk-truncated vs full-text mismatch)
  - Rank among top-K document hits (if found)
  - Rank when aggregating chunk hits by doc_id (if found)

Usage (from project root):
  python scripts/probe_retrieval_gold.py
  python scripts/probe_retrieval_gold.py --data data_clean/eval/retrieval_questions.jsonl --topk 200 --scan-chunks 800

Env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (see .env)
"""

from __future__ import annotations

import argparse
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

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_retrieval as er  # noqa: E402


VECTOR_DIMS = er.VECTOR_DIMS


def query_chunk_vector(session, *, vector: list[float], limit: int) -> list[dict]:
    """Chunk vector search; same strategy as graph_rag_query.py + procedure fallback."""
    variants = [
        (
            """
            MATCH (c:Chunk)
              SEARCH c IN (
                VECTOR INDEX chunk_embeddings
                FOR $vector
                LIMIT $limit
              ) SCORE AS score
            RETURN c.doc_id AS doc_id, c.chunk_index AS chunk_index, score
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
            RETURN c.doc_id AS doc_id, c.chunk_index AS chunk_index, score
            ORDER BY score DESC
            """,
            {"vector": vector, "dims": VECTOR_DIMS, "limit": limit},
        ),
    ]
    last_exc: Exception | None = None
    for q, params in variants:
        try:
            return session.run(q, **params).data()
        except Exception as exc:
            last_exc = exc
    try:
        return session.run(
            """
            CALL db.index.vector.queryNodes('chunk_embeddings', $k, $vector)
            YIELD node AS c, score
            RETURN c.doc_id AS doc_id, c.chunk_index AS chunk_index, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            vector=vector,
            k=limit,
            limit=limit,
        ).data()
    except Exception as exc:
        if last_exc is not None:
            raise last_exc from exc
        raise


def rank_in_list(doc_id: str, ordered: list[str]) -> int | None:
    for i, x in enumerate(ordered, start=1):
        if x == doc_id:
            return i
    return None


def best_doc_scores_from_chunks(rows: list[dict]) -> tuple[dict[str, float], list[str]]:
    """Max chunk score per doc_id; return mapping and doc ids sorted by score DESC."""
    best: dict[str, float] = {}
    for r in rows:
        did = r.get("doc_id")
        if did is None:
            continue
        did = str(did)
        sc = float(r.get("score") or 0.0)
        if did not in best or sc > best[did]:
            best[did] = sc
    ordered = sorted(best.keys(), key=lambda d: -best[d])
    return best, ordered


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe document vs chunk retrieval ranks for gold doc ids")
    ap.add_argument("--data", type=Path, default=Path("data_clean/eval/retrieval_questions.jsonl"))
    ap.add_argument(
        "--topk",
        type=int,
        default=200,
        help="How many Document hits to pull when ranking (default 200)",
    )
    ap.add_argument(
        "--scan-chunks",
        type=int,
        default=800,
        help="How many Chunk hits to scan before aggregating by doc_id (default 800)",
    )
    args = ap.parse_args()

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not uri or not password:
        print("Set NEO4J_URI and NEO4J_PASSWORD in the environment or .env")
        return 1

    if not args.data.exists():
        print(f"ERROR: file not found: {args.data}")
        return 1

    rows = list(er.iter_jsonl(args.data))
    if not rows:
        print(f"ERROR: no rows in {args.data}")
        return 1

    print(f"Model: {er.EMBED_MODEL}")
    print(f"Document top-K: {args.topk}  |  Chunk scan: {args.scan_chunks}\n")

    model = SentenceTransformer(er.EMBED_MODEL)

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        try:
            driver.verify_connectivity()
        except ServiceUnavailable as exc:
            root = exc.__cause__
            if isinstance(root, socket.gaierror) or "DNS resolve" in str(exc):
                print("Neo4j DNS/connectivity failed. Check NEO4J_URI and network.")
                return 1
            raise

        with driver.session() as session:
            for idx, row in enumerate(rows, start=1):
                q = (row.get("query") or "").strip()
                gold = [str(x) for x in (row.get("expected_doc_ids") or [])]
                if not q or not gold:
                    continue

                print("=" * 72)
                print(f"[{idx}] Query: {q}")
                print(f"    Gold doc ids ({len(gold)}): {gold}")

                # --- Document node stats ---
                stats = session.run(
                    """
                    UNWIND $ids AS id
                    OPTIONAL MATCH (d:Document {id: id})
                    RETURN id, d IS NOT NULL AS ok, coalesce(size(d.text), -1) AS nchars
                    """,
                    ids=gold,
                ).data()
                for s in stats:
                    st = "present" if s["ok"] else "MISSING"
                    nc = s["nchars"] if s["nchars"] >= 0 else "n/a"
                    print(f"    Document {s['id'][:12]}... : {st}  |  size(text)={nc}")

                vec = model.encode(q).tolist()

                # --- Document index ranking ---
                doc_hits = er.query_document_vector(session, vector=vec, limit=int(args.topk))
                doc_order = [str(h["id"]) for h in doc_hits if h.get("id") is not None]
                print(f"\n    --- document_embeddings (top min(5,{len(doc_order)})) ---")
                for h in doc_hits[:5]:
                    star = " *" if str(h.get("id")) in gold else ""
                    print(f"      rank {doc_order.index(str(h['id'])) + 1}: {h['id'][:12]}... score={float(h['score']):.4f}{star}")

                print(f"\n    Gold rank by Document vector (scan top {args.topk}):")
                for gid in gold:
                    rnk = rank_in_list(gid, doc_order)
                    if rnk is None:
                        print(f"      {gid}  ->  NOT IN top {args.topk}")
                    else:
                        sc = next(float(x["score"]) for x in doc_hits if str(x.get("id")) == gid)
                        print(f"      {gid}  ->  rank {rnk}  score={sc:.4f}")

                # --- Chunk index: best score per doc ---
                ch_rows = query_chunk_vector(session, vector=vec, limit=int(args.scan_chunks))
                _best, chunk_doc_order = best_doc_scores_from_chunks(ch_rows)
                print(f"\n    --- chunk_embeddings (best score per doc; scanned {len(ch_rows)} chunks, {len(chunk_doc_order)} unique docs) ---")
                for i, did in enumerate(chunk_doc_order[:5], start=1):
                    star = " *" if did in gold else ""
                    print(f"      doc rank {i}: {did[:12]}... best_chunk_score={_best[did]:.4f}{star}")

                print(f"\n    Gold rank by Chunk->doc aggregation (scan top {args.scan_chunks} chunks):")
                for gid in gold:
                    rnk = rank_in_list(gid, chunk_doc_order)
                    if rnk is None:
                        print(f"      {gid}  ->  NOT IN aggregated top (doc never appeared in chunk hits)")
                    else:
                        print(f"      {gid}  ->  rank {rnk}  best_chunk_score={_best.get(gid, 0):.4f}")
                print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
