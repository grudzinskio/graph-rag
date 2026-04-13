"""
Load MSOE documents (embedded), entities, and per-document relations into Neo4j.

Documents get a vector index for retrieval; entities keep a vector index.
Relations are keyed by (head, tail, label, doc_id) so each page keeps its own edges.

Env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

VECTOR_DIMS = 384
EMBED_MODEL = "all-MiniLM-L6-v2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def setup_schema(tx, *, doc_index: bool, entity_index: bool) -> None:
    tx.run("CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
    if entity_index:
        tx.run(
            f"""
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {{indexConfig: {{
             `vector.dimensions`: {VECTOR_DIMS},
             `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
    if doc_index:
        tx.run(
            f"""
            CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
            FOR (d:Document) ON (d.embedding)
            OPTIONS {{indexConfig: {{
             `vector.dimensions`: {VECTOR_DIMS},
             `vector.similarity_function`: 'cosine'
            }}}}
            """
        )


def clear_graph(tx) -> None:
    tx.run("MATCH (n) DETACH DELETE n")


def load_documents_batch(tx, batch: list[dict]) -> None:
    tx.run(
        """
        UNWIND $batch AS item
        WITH item WHERE item.id IS NOT NULL AND item.id <> ''
        MERGE (d:Document {id: item.id})
        SET d.text = item.text,
            d.source_path = item.source_path,
            d.embedding = item.embedding
        """,
        batch=batch,
    )


def load_entities_batch(tx, batch: list[dict]) -> None:
    tx.run(
        """
        UNWIND $batch AS item
        WITH item WHERE item.id IS NOT NULL AND item.id <> ''
        MERGE (e:Entity {id: item.id})
        SET e.text = item.id,
            e.label = item.label,
            e.embedding = item.embedding
        """,
        batch=batch,
    )


def load_has_entity_batch(tx, batch: list[dict]) -> None:
    tx.run(
        """
        UNWIND $batch AS row
        MATCH (d:Document {id: row.doc_id})
        MATCH (e:Entity {id: row.entity_text})
        MERGE (d)-[h:HAS_ENTITY {start_char: row.start_char, end_char: row.end_char}]->(e)
        """,
        batch=batch,
    )


def load_relations_batch(tx, batch: list[dict]) -> None:
    tx.run(
        """
        UNWIND $batch AS rel
        MATCH (src:Entity {id: rel.head_text})
        MATCH (tgt:Entity {id: rel.tail_text})
        MATCH (:Document {id: rel.doc_id})
        MERGE (src)-[r:REL {label: rel.label, doc_id: rel.doc_id}]->(tgt)
        SET r.score = rel.score
        """,
        batch=batch,
    )


def build_allowed_entities(entities_path: Path, max_unique: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for o in iter_jsonl(entities_path):
        if len(out) >= max_unique:
            break
        t = (o.get("text") or "").strip()
        if t and t not in out:
            out[t] = {"id": t, "label": o.get("label")}
    return out


def load_documents_jsonl(docs_path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for o in iter_jsonl(docs_path):
        did = o.get("id")
        if not did:
            continue
        by_id[str(did)] = {
            "text": o.get("text") or "",
            "source_path": o.get("source_path") or "",
        }
    return by_id


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Upload documents + graph to Neo4j")
    ap.add_argument("--docs", type=Path, default=root / "data_clean/msoe/documents.jsonl")
    ap.add_argument("--entities", type=Path, default=root / "data_clean/extracted/entities.jsonl")
    ap.add_argument("--relations", type=Path, default=root / "data_clean/extracted/relations.jsonl")
    ap.add_argument("--max-entities", type=int, default=50_000)
    ap.add_argument("--max-relations", type=int, default=175_000)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--clear", action="store_true", help="Wipe the database before load")
    args = ap.parse_args()

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not uri or not password:
        print("Set NEO4J_URI and NEO4J_PASSWORD in the environment or .env")
        return 1

    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model ({EMBED_MODEL})...")
    st_model = SentenceTransformer(EMBED_MODEL)

    allowed = build_allowed_entities(args.entities, args.max_entities)
    allowed_set = set(allowed.keys())
    print(f"Allowed entity strings: {len(allowed_set):,}")

    print("Indexing documents.jsonl...")
    doc_by_id = load_documents_jsonl(args.docs)

    doc_ids: set[str] = set()
    for o in iter_jsonl(args.entities):
        t = (o.get("text") or "").strip()
        if t not in allowed_set:
            continue
        did = o.get("doc_id")
        if did is not None:
            doc_ids.add(str(did))
    print(f"Documents referenced by allowed entities: {len(doc_ids):,}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    with driver.session() as session:
        if args.clear:
            print("Clearing graph...")
            session.execute_write(clear_graph)
        session.execute_write(lambda tx: setup_schema(tx, doc_index=True, entity_index=True))

    # --- Documents with embeddings ---
    doc_rows = []
    for did in doc_ids:
        rec = doc_by_id.get(did)
        if not rec:
            continue
        doc_rows.append({"id": did, "text": rec["text"], "source_path": rec["source_path"]})

    print(f"Uploading {len(doc_rows):,} Document nodes with embeddings...")
    with driver.session() as session:
        for i in range(0, len(doc_rows), args.batch_size):
            batch = doc_rows[i : i + args.batch_size]
            texts = [b["text"] for b in batch]
            embs = st_model.encode(texts, show_progress_bar=False).tolist()
            for b, emb in zip(batch, embs):
                b["embedding"] = emb
            session.execute_write(load_documents_batch, batch)

    # --- Entities with embeddings ---
    items = list(allowed.values())
    print(f"Uploading {len(items):,} Entity nodes with embeddings...")
    with driver.session() as session:
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            texts = [x["id"] for x in batch]
            embs = st_model.encode(texts, show_progress_bar=False).tolist()
            for x, emb in zip(batch, embs):
                x["embedding"] = emb
            session.execute_write(load_entities_batch, batch)

    # --- HAS_ENTITY from entities.jsonl ---
    has_rows: list[dict] = []
    seen_m = set()
    for o in iter_jsonl(args.entities):
        t = (o.get("text") or "").strip()
        if t not in allowed_set:
            continue
        did = str(o.get("doc_id", ""))
        if did not in doc_ids or did not in doc_by_id:
            continue
        key = (did, t, o.get("start_char"), o.get("end_char"))
        if key in seen_m:
            continue
        seen_m.add(key)
        has_rows.append(
            {
                "doc_id": did,
                "entity_text": t,
                "start_char": o.get("start_char"),
                "end_char": o.get("end_char"),
            }
        )
    print(f"Uploading {len(has_rows):,} HAS_ENTITY edges...")
    with driver.session() as session:
        for i in range(0, len(has_rows), args.batch_size):
            session.execute_write(load_has_entity_batch, has_rows[i : i + args.batch_size])

    # --- Relations (per doc_id) ---
    accepted = 0
    rel_batch: list[dict] = []
    print(f"Uploading relations (cap {args.max_relations:,})...")
    with driver.session() as session:
        with args.relations.open(encoding="utf-8", errors="replace") as rf:
            for line in rf:
                if accepted >= args.max_relations:
                    break
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ht = (r.get("head") or {}).get("text", "").strip()
                tt = (r.get("tail") or {}).get("text", "").strip()
                lab = r.get("label")
                doc_id = str(r.get("doc_id", ""))
                if ht not in allowed_set or tt not in allowed_set or not lab:
                    continue
                if doc_id not in doc_by_id:
                    continue
                rel_batch.append(
                    {
                        "head_text": ht,
                        "tail_text": tt,
                        "label": lab,
                        "score": r.get("score"),
                        "doc_id": doc_id,
                    }
                )
                accepted += 1
                if len(rel_batch) >= args.batch_size:
                    session.execute_write(load_relations_batch, rel_batch)
                    rel_batch = []
        if rel_batch:
            session.execute_write(load_relations_batch, rel_batch)

    driver.close()
    print(f"Done. Documents={len(doc_rows):,}, entities={len(items):,}, HAS_ENTITY={len(has_rows):,}, REL={accepted:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
