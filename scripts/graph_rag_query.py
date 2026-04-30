"""
GraphRAG terminal query tool with optional multi-hop traversal.

CLI flags (all optional):
- --compact-top-k: Number of documents to enrich in compact pass.
- --compact-chunk-k: Number of top chunk hits in compact pass.
- --compact-rel-cap: Max relations per document in compact pass.
- --compact-max-chars: Max context characters for compact pass.
- --compact-text-chars: Max text chars per doc/chunk line in compact pass.

- --expand-top-k: Number of documents to enrich in expanded pass.
- --expand-chunk-k: Number of top chunk hits in expanded pass.
- --expand-rel-cap: Max relations per document in expanded pass.
- --expand-max-chars: Max context characters for expanded pass.
- --expand-text-chars: Max text chars per doc/chunk line in expanded pass.

- --llm-char-budget: Hard cap on context chars before final answer call.
- --max-hops: Maximum traversal hops after retrieval passes.
- --hop-per-term-entities: Entity seeds per focus term during a hop.
- --hop-rel-limit: Outgoing relation lines per seeded entity.
- --hop-max-chars: Max chars added per traversal hop block.
"""

import os
import sys
import json
import re
import argparse
import subprocess
from pathlib import Path

# --- Dependency Bootstrap ---
def _ensure(pip_name: str, import_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing missing dependency: {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

# Ensure all external dependencies are present
_ensure("python-dotenv", "dotenv")
_ensure("neo4j", "neo4j")
_ensure("sentence-transformers", "sentence_transformers")
_ensure("google-generativeai", "google.generativeai")

from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j+s://92fb1806.databases.neo4j.io")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j") 
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

# LLM Configuration (Gemini)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Load the same model used for uploading
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

VECTOR_DIMS = 384

def _doc_vector_search(session, query_vector, top_k: int):
    # Prefer SEARCH clause; fall back to deprecated procedure.
    variants = [
        (
            """
            MATCH (d:Document)
              SEARCH d IN (
                VECTOR INDEX document_embeddings
                FOR $vector
                LIMIT $limit
              ) SCORE AS score
            RETURN d.id AS id, d.text AS text, d.source_path AS path, score
            ORDER BY score DESC
            """,
            {"vector": query_vector, "limit": top_k},
        ),
        (
            """
            MATCH (d:Document)
              SEARCH d IN (
                VECTOR INDEX document_embeddings
                FOR vector($vector, $dims, FLOAT)
                LIMIT $limit
              ) SCORE AS score
            RETURN d.id AS id, d.text AS text, d.source_path AS path, score
            ORDER BY score DESC
            """,
            {"vector": query_vector, "dims": VECTOR_DIMS, "limit": top_k},
        ),
    ]
    for q, params in variants:
        try:
            return session.run(q, **params).data()
        except Exception:
            pass
    return session.run(
        """
        CALL db.index.vector.queryNodes('document_embeddings', $k, $vector)
        YIELD node AS d, score
        RETURN d.id AS id, d.text AS text, d.source_path AS path, score
        ORDER BY score DESC
        LIMIT $limit
        """,
        vector=query_vector,
        k=top_k,
        limit=top_k,
    ).data()


def _chunk_vector_search(session, query_vector, top_k: int):
    variants = [
        (
            """
            MATCH (c:Chunk)
              SEARCH c IN (
                VECTOR INDEX chunk_embeddings
                FOR $vector
                LIMIT $limit
              ) SCORE AS score
            RETURN c.id AS id, c.doc_id AS doc_id, c.text AS text, c.chunk_index AS chunk_index, score
            ORDER BY score DESC
            """,
            {"vector": query_vector, "limit": top_k},
        ),
        (
            """
            MATCH (c:Chunk)
              SEARCH c IN (
                VECTOR INDEX chunk_embeddings
                FOR vector($vector, $dims, FLOAT)
                LIMIT $limit
              ) SCORE AS score
            RETURN c.id AS id, c.doc_id AS doc_id, c.text AS text, c.chunk_index AS chunk_index, score
            ORDER BY score DESC
            """,
            {"vector": query_vector, "dims": VECTOR_DIMS, "limit": top_k},
        ),
    ]
    for q, params in variants:
        try:
            return session.run(q, **params).data()
        except Exception:
            pass
    return []


def _entity_vector_search(session, query_vector, top_k: int):
    variants = [
        (
            """
            MATCH (node:Entity)
              SEARCH node IN (
                VECTOR INDEX entity_embeddings
                FOR $vector
                LIMIT $k
              ) SCORE AS score
            RETURN node, score
            """,
            {"vector": query_vector, "k": top_k},
        ),
        (
            """
            MATCH (node:Entity)
              SEARCH node IN (
                VECTOR INDEX entity_embeddings
                FOR vector($vector, $dims, FLOAT)
                LIMIT $k
              ) SCORE AS score
            RETURN node, score
            """,
            {"vector": query_vector, "dims": VECTOR_DIMS, "k": top_k},
        ),
    ]
    for q, params in variants:
        try:
            return session.run(q, **params)
        except Exception:
            pass
    return session.run(
        """
        CALL db.index.vector.queryNodes('entity_embeddings', $k, $vector)
        YIELD node, score
        RETURN node, score
        """,
        vector=query_vector,
        k=top_k,
    )


def _entity_neighbors(session, entity_id: str, rel_limit: int = 15):
    return session.run(
        """
        MATCH (e:Entity {id: $eid})-[r:REL]->(n:Entity)
        RETURN e.id AS src, r.label AS rel, n.id AS dst, r.score AS score, r.doc_id AS doc_id
        ORDER BY score DESC
        LIMIT $lim
        """,
        eid=entity_id,
        lim=rel_limit,
    ).data()


def traverse_graph_by_focus_terms(focus_terms: list[str], per_term_entities: int = 3, rel_limit: int = 12, max_chars: int = 5000):
    """
    Explore graph neighborhoods around focus terms using entity vector retrieval.
    Returns compact traversal context text.
    """
    lines: list[str] = []
    stats = {
        "focus_terms_used": 0,
        "entity_seeds": 0,
        "relations_traversed": 0,
    }
    used = 0

    def _append(line: str) -> bool:
        nonlocal used
        if used + len(line) + 1 > max_chars:
            return False
        lines.append(line)
        used += len(line) + 1
        return True

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        with driver.session() as session:
            seen_entities: set[str] = set()
            for term in focus_terms:
                if not term:
                    continue
                stats["focus_terms_used"] += 1
                qv = model.encode(term).tolist()
                hits = list(_entity_vector_search(session, qv, per_term_entities))
                if not _append(f"--- Traversal seed: {term} ---"):
                    break
                for rec in hits:
                    node = rec.get("node")
                    sc = float(rec.get("score") or 0.0)
                    if node is None:
                        continue
                    eid = node.get("id")
                    if not eid or eid in seen_entities:
                        continue
                    seen_entities.add(eid)
                    stats["entity_seeds"] += 1
                    if not _append(f"Entity: {eid} (sim={sc:.4f})"):
                        break
                    for edge in _entity_neighbors(session, eid, rel_limit=rel_limit):
                        stats["relations_traversed"] += 1
                        line = f"  ({edge['src']})-[{edge['rel']} score={float(edge.get('score') or 0.0):.3f}]->({edge['dst']}) [doc={edge.get('doc_id','')}]"
                        if not _append(line):
                            break
                _append("")
    return "\n".join(lines).strip(), stats


def get_graph_context(query_text, top_k=5, text_chars=6000, rel_cap=80, chunk_k=14, max_total_chars=12000):
    """Prefer chunk search, then compact doc+relation context under a strict char budget."""
    query_vector = model.encode(query_text).tolist()
    lines: list[str] = []
    stats = {
        "chunks_retrieved": 0,
        "chunks_used": 0,
        "docs_used": 0,
        "relations_used": 0,
    }
    used_chars = 0

    def _append(line: str) -> bool:
        nonlocal used_chars
        if used_chars + len(line) + 1 > max_total_chars:
            return False
        lines.append(line)
        used_chars += len(line) + 1
        return True

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        chunk_hits = []
        try:
            with driver.session() as session:
                chunk_hits = _chunk_vector_search(session, query_vector, chunk_k)
        except Exception:
            chunk_hits = []
        stats["chunks_retrieved"] = len(chunk_hits)

        if chunk_hits:
            seen_docs = set()
            for row in chunk_hits:
                did = row.get("doc_id")
                if not did:
                    continue
                score = float(row.get("score", 0.0))
                cidx = row.get("chunk_index")
                ctext = (row.get("text") or "")[:1200]
                if not _append(f"--- Chunk doc_id={did} chunk={cidx} sim={score:.4f} ---"):
                    break
                if not _append(ctext):
                    break
                _append("")
                seen_docs.add(did)
                stats["chunks_used"] += 1

            # Add graph signals for top related docs from chunk hits.
            for did in list(seen_docs)[:top_k]:
                stats["docs_used"] += 1
                with driver.session() as session:
                    path_rec = session.run(
                        "MATCH (d:Document {id: $did}) RETURN d.source_path AS path",
                        did=did,
                    ).single()
                    path = (path_rec and path_rec.get("path")) or ""
                    if not _append(f"--- Graph context for {path} (id={did}) ---"):
                        break
                    kws = session.run(
                        """
                        MATCH (:Document {id: $did})-[:HAS_ENTITY]->(e:Entity)
                        RETURN collect(DISTINCT coalesce(e.display, e.id)) AS kws
                        """,
                        did=did,
                    ).single()
                    keywords = (kws and kws.get("kws")) or []
                    if keywords and not _append("Keywords: " + ", ".join(keywords[:35])):
                        break
                    rels = session.run(
                        """
                        MATCH (a:Entity)-[r:REL]->(b:Entity)
                        WHERE r.doc_id = $did
                        RETURN a.id AS h, r.label AS rel, b.id AS t, r.score AS sc
                        ORDER BY sc DESC
                        LIMIT $rcap
                        """,
                        did=did,
                        rcap=min(rel_cap, 25),
                    )
                    for r in rels:
                        stats["relations_used"] += 1
                        if not _append(f"  ({r['h']})-[{r['rel']}]->({r['t']})"):
                            break
                    _append("")
            return "\n".join(lines).strip(), stats

        doc_hits = []
        try:
            with driver.session() as session:
                doc_hits = _doc_vector_search(session, query_vector, top_k)
        except Exception:
            doc_hits = []

        if doc_hits:
            for row in doc_hits:
                did = row["id"]
                stats["docs_used"] += 1
                path = row.get("path") or ""
                score = row.get("score", 0.0)
                text = (row.get("text") or "")[:text_chars]
                if not _append(f"--- Document {path} (id={did}, sim={float(score):.4f}) ---"):
                    break
                if not _append(text):
                    break
                with driver.session() as session:
                    kws = session.run(
                        """
                        MATCH (:Document {id: $did})-[:HAS_ENTITY]->(e:Entity)
                        RETURN collect(DISTINCT coalesce(e.display, e.id)) AS kws
                        """,
                        did=did,
                    ).single()
                    keywords = (kws and kws.get("kws")) or []
                    if keywords:
                        if not _append("Keywords: " + ", ".join(keywords[:50])):
                            break
                    rels = session.run(
                        """
                        MATCH (a:Entity)-[r:REL]->(b:Entity)
                        WHERE r.doc_id = $did
                        RETURN a.id AS h, r.label AS rel, b.id AS t, r.score AS sc
                        ORDER BY sc DESC
                        LIMIT $rcap
                        """,
                        did=did,
                        rcap=rel_cap,
                    )
                    for r in rels:
                        stats["relations_used"] += 1
                        if not _append(f"  ({r['h']})-[{r['rel']}]->({r['t']})"):
                            break
                _append("")
            return "\n".join(lines).strip(), stats

        # Legacy graphs: entity index only
        with driver.session() as session:
            result = _entity_vector_search(session, query_vector, top_k)
            # If we got only (node, score), expand neighbors here.
            rows = []
            for rec in result:
                node = rec.get("node")
                score = rec.get("score")
                if node is None:
                    continue
                rows.append((node.get("id"), node.get("label"), float(score or 0.0)))

            # Expand neighbors with a second query (avoid massive expansions inside vector search).
            for nid, nlab, sc in rows[:top_k]:
                neigh = session.run(
                    """
                    MATCH (node:Entity {id: $id})-[r]->(neighbor)
                    RETURN
                        node.id AS source,
                        node.label AS source_label,
                        type(r) AS rel_type,
                        r.label AS rel_label,
                        neighbor.id AS target,
                        neighbor.label AS target_label,
                        $score AS score
                    LIMIT 50
                    """,
                    id=nid,
                    score=sc,
                )
                for record in neigh:
                    stats["relations_used"] += 1
                    triple = (
                        f"({record['source']}:{record['source_label']}) "
                        f"--[{record['rel_label'] or record['rel_type']}]--> "
                        f"({record['target']}:{record['target_label']})"
                    )
                    lines.append(triple)
            return "\n".join(lines), stats
    return "\n".join(lines), stats

def _gemini_model():
    return genai.GenerativeModel('gemini-3-flash-preview')


def query_llm(prompt, context):
    full_prompt = f"""
You are a helpful assistant. Answer the user's question based ONLY on the following knowledge graph context.
If the context doesn't contain the answer, say you don't know.

### KNOWLEDGE GRAPH CONTEXT:
{context}

### USER QUESTION:
{prompt}

### ANSWER:
"""
    if not GOOGLE_API_KEY:
        return "ERROR: No GOOGLE_API_KEY found in .env"

    try:
        gemini_model = _gemini_model()
        response = gemini_model.generate_content(full_prompt)
        usage = getattr(response, "usage_metadata", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "total_tokens": getattr(usage, "total_token_count", None) if usage else None,
        }
        return response.text, usage_dict
    except Exception as e:
        return f"Error calling Gemini: {e}", {"prompt_tokens": None, "output_tokens": None, "total_tokens": None}


def _extract_json_obj(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def should_expand_context(question: str, context: str) -> dict:
    """
    Ask the model to decide if a second retrieval/traversal pass is needed.
    Returns dict: {need_more_context: bool, reason: str, focus_terms: list[str]}
    """
    if not GOOGLE_API_KEY:
        return {"need_more_context": False, "reason": "No API key", "focus_terms": []}
    prompt = f"""
You are a retrieval controller for GraphRAG.
Given the user question and current evidence, decide if we should run one more graph retrieval pass.

Return STRICT JSON only:
{{
  "need_more_context": true|false,
  "reason": "short reason",
  "focus_terms": ["term1", "term2", "term3"]
}}

Rules:
- need_more_context=true if evidence is likely insufficient, ambiguous, or missing direct support.
- need_more_context=false if evidence appears sufficient to answer confidently.
- focus_terms should be 0-5 short terms that would improve second retrieval.

QUESTION:
{question}

CURRENT_CONTEXT:
{context}
"""
    try:
        txt = _gemini_model().generate_content(prompt).text or ""
        obj = _extract_json_obj(txt)
        return {
            "need_more_context": bool(obj.get("need_more_context", False)),
            "reason": str(obj.get("reason", ""))[:200],
            "focus_terms": [str(x).strip() for x in (obj.get("focus_terms") or []) if str(x).strip()][:5],
        }
    except Exception:
        return {"need_more_context": False, "reason": "Controller fallback", "focus_terms": []}


def next_hop_plan(question: str, context: str) -> dict:
    """
    Decide if we should continue graph traversal and what to focus on next.
    Returns: {continue: bool, reason: str, focus_terms: list[str]}
    """
    if not GOOGLE_API_KEY:
        return {"continue": False, "reason": "No API key", "focus_terms": []}
    prompt = f"""
You are controlling multi-hop GraphRAG retrieval.
Given a user question and current evidence, decide whether another graph hop is useful.

Return STRICT JSON only:
{{
  "continue": true|false,
  "reason": "short reason",
  "focus_terms": ["term1", "term2", "term3"]
}}

Rules:
- continue=true only if another hop likely adds missing evidence.
- Stop when evidence is likely sufficient.
- focus_terms: 0-5 specific entities/topics for the next hop.

QUESTION:
{question}

CURRENT_EVIDENCE:
{context}
"""
    try:
        txt = _gemini_model().generate_content(prompt).text or ""
        obj = _extract_json_obj(txt)
        return {
            "continue": bool(obj.get("continue", False)),
            "reason": str(obj.get("reason", ""))[:200],
            "focus_terms": [str(x).strip() for x in (obj.get("focus_terms") or []) if str(x).strip()][:5],
        }
    except Exception:
        return {"continue": False, "reason": "Hop planner fallback", "focus_terms": []}

def parse_args():
    ap = argparse.ArgumentParser(description="GraphRAG terminal interface with optional multi-hop traversal")
    ap.add_argument("--compact-top-k", type=int, default=3)
    ap.add_argument("--compact-chunk-k", type=int, default=8)
    ap.add_argument("--compact-rel-cap", type=int, default=12)
    ap.add_argument("--compact-max-chars", type=int, default=7000)
    ap.add_argument("--compact-text-chars", type=int, default=1200)

    ap.add_argument("--expand-top-k", type=int, default=6)
    ap.add_argument("--expand-chunk-k", type=int, default=18)
    ap.add_argument("--expand-rel-cap", type=int, default=30)
    ap.add_argument("--expand-max-chars", type=int, default=14000)
    ap.add_argument("--expand-text-chars", type=int, default=1600)

    ap.add_argument("--llm-char-budget", type=int, default=22000)
    ap.add_argument("--max-hops", type=int, default=2)
    ap.add_argument("--hop-per-term-entities", type=int, default=3)
    ap.add_argument("--hop-rel-limit", type=int, default=10)
    ap.add_argument("--hop-max-chars", type=int, default=4500)
    return ap.parse_args()


def main():
    args = parse_args()
    if not NEO4J_PASSWORD:
        print("Error: NEO4J_PASSWORD environment variable not set.")
        return

    print("\n--- GraphRAG Terminal Interface (Gemini Powered) ---")
    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ").strip()
        if user_query.lower() in ['exit', 'quit']:
            break
        if not user_query:
            continue

        print("\n1. Searching graph for compact context...")
        context, metrics = get_graph_context(
            user_query,
            top_k=args.compact_top_k,
            rel_cap=args.compact_rel_cap,
            chunk_k=args.compact_chunk_k,
            max_total_chars=args.compact_max_chars,
            text_chars=args.compact_text_chars,
        )
        metrics["hops_executed"] = 0
        metrics["focus_terms_used"] = 0
        metrics["entity_seeds"] = 0
        metrics["relations_traversed"] = 0
        
        if not context or context == "No relevant nodes found in the graph.":
            print("No relevant graph context found.")
        else:
            all_lines = context.splitlines()
            preview_n = 24
            print(
                f"Context: {len(all_lines)} lines, {len(context)} chars "
                f"(compact pass). "
                f"Preview (first {min(preview_n, len(all_lines))} lines; full text still sent to Gemini):"
            )
            for line in all_lines[:preview_n]:
                print(f"  | {line}")
            if len(all_lines) > preview_n:
                print(f"  | ... ({len(all_lines) - preview_n} more lines omitted here)")

        print("2. Deciding whether to traverse more...")
        decision = should_expand_context(user_query, context)
        need_more = decision.get("need_more_context", False)
        if need_more:
            focus_terms = decision.get("focus_terms") or []
            focus_query = user_query if not focus_terms else f"{user_query} | " + " | ".join(focus_terms)
            print(f"   Expanding retrieval. Reason: {decision.get('reason', '')}")
            context_more, more_metrics = get_graph_context(
                focus_query,
                top_k=args.expand_top_k,
                rel_cap=args.expand_rel_cap,
                chunk_k=args.expand_chunk_k,
                max_total_chars=args.expand_max_chars,
                text_chars=args.expand_text_chars,
            )
            metrics["chunks_retrieved"] += more_metrics.get("chunks_retrieved", 0)
            metrics["chunks_used"] += more_metrics.get("chunks_used", 0)
            metrics["docs_used"] += more_metrics.get("docs_used", 0)
            metrics["relations_used"] += more_metrics.get("relations_used", 0)
            if context_more:
                context = f"{context}\n\n--- ADDITIONAL CONTEXT PASS ---\n{context_more}"
                # Hard ceiling before final LLM call (character budget proxy for token budget).
                context = context[:args.llm_char_budget]
        else:
            print("   Compact context looks sufficient; skipping second pass.")

        # Agentic multi-hop traversal loop.
        max_hops = args.max_hops
        hop = 0
        while hop < max_hops:
            plan = next_hop_plan(user_query, context)
            if not plan.get("continue", False):
                print(f"   Stopping traversal. Reason: {plan.get('reason','sufficient evidence')}")
                break
            focus_terms = plan.get("focus_terms") or []
            if not focus_terms:
                print("   Planner requested another hop but gave no focus terms; stopping.")
                break
            hop += 1
            print(f"   Hop {hop}/{max_hops}: traversing graph for {focus_terms}")
            hop_ctx, hop_metrics = traverse_graph_by_focus_terms(
                focus_terms,
                per_term_entities=args.hop_per_term_entities,
                rel_limit=args.hop_rel_limit,
                max_chars=args.hop_max_chars,
            )
            if not hop_ctx:
                print("   No additional traversal context found; stopping.")
                break
            metrics["hops_executed"] += 1
            metrics["focus_terms_used"] += hop_metrics.get("focus_terms_used", 0)
            metrics["entity_seeds"] += hop_metrics.get("entity_seeds", 0)
            metrics["relations_traversed"] += hop_metrics.get("relations_traversed", 0)
            context = f"{context}\n\n--- TRAVERSAL HOP {hop} ---\n{hop_ctx}"
            context = context[:args.llm_char_budget]

        print("3. Querying Gemini...")
        answer, usage = query_llm(user_query, context)
        
        print("\n=== GEMINI RESPONSE ===")
        print(answer)
        print("====================\n")
        print("--- QUERY STATS ---")
        print(
            "Tokens: "
            f"input={usage.get('prompt_tokens', 'n/a')} "
            f"output={usage.get('output_tokens', 'n/a')} "
            f"total={usage.get('total_tokens', 'n/a')}"
        )
        print(
            "Retrieval: "
            f"chunks_retrieved={metrics.get('chunks_retrieved', 0)} "
            f"chunks_used={metrics.get('chunks_used', 0)} "
            f"docs_used={metrics.get('docs_used', 0)} "
            f"relations_used={metrics.get('relations_used', 0)}"
        )
        print(
            "Traversal: "
            f"hops_executed={metrics.get('hops_executed', 0)} "
            f"focus_terms_used={metrics.get('focus_terms_used', 0)} "
            f"entity_seeds={metrics.get('entity_seeds', 0)} "
            f"relations_traversed={metrics.get('relations_traversed', 0)}"
        )
        print("-------------------\n")

if __name__ == "__main__":
    main()
