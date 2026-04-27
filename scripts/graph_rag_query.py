import os
import sys
import json
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


def get_graph_context(query_text, top_k=5, text_chars=6000, rel_cap=80):
    """Prefer document vector search + keywords + per-document relations."""
    query_vector = model.encode(query_text).tolist()
    lines: list[str] = []

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
        doc_hits = []
        try:
            with driver.session() as session:
                doc_hits = _doc_vector_search(session, query_vector, top_k)
        except Exception:
            doc_hits = []

        if doc_hits:
            for row in doc_hits:
                did = row["id"]
                path = row.get("path") or ""
                score = row.get("score", 0.0)
                text = (row.get("text") or "")[:text_chars]
                lines.append(f"--- Document {path} (id={did}, sim={float(score):.4f}) ---")
                lines.append(text)
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
                        lines.append("Keywords: " + ", ".join(keywords[:80]))
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
                        lines.append(f"  ({r['h']})-[{r['rel']}]->({r['t']})")
                lines.append("")
            return "\n".join(lines).strip()

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
                    triple = (
                        f"({record['source']}:{record['source_label']}) "
                        f"--[{record['rel_label'] or record['rel_type']}]--> "
                        f"({record['target']}:{record['target_label']})"
                    )
                    lines.append(triple)
            return "\n".join(lines)
    return "\n".join(lines)

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
        # gemini-3-flash-preview is the standard naming convention
        gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
        response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini: {e}"

def main():
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

        print("\n1. Searching graph for context...")
        context = get_graph_context(user_query)
        
        if not context or context == "No relevant nodes found in the graph.":
            print("No relevant graph context found.")
        else:
            # Not graph "triples": each line of pasted document body is one line (catalog pages add hundreds).
            all_lines = context.splitlines()
            preview_n = 40
            print(
                f"Context: {len(all_lines)} lines, {len(context)} chars "
                f"(vector top docs + text up to text_chars + keywords + rels). "
                f"Preview (first {min(preview_n, len(all_lines))} lines; full text still sent to Gemini):"
            )
            for line in all_lines[:preview_n]:
                print(f"  | {line}")
            if len(all_lines) > preview_n:
                print(f"  | ... ({len(all_lines) - preview_n} more lines omitted here)")

        print("2. Querying Gemini...")
        answer = query_llm(user_query, context)
        
        print("\n=== GEMINI RESPONSE ===")
        print(answer)
        print("====================\n")

if __name__ == "__main__":
    main()
