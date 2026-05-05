"""
Load MSOE documents (embedded), entities, and per-document relations into Neo4j.

Documents get a vector index for retrieval; entities keep a vector index.
Relations are keyed by (head, tail, label, doc_id) so each page keeps its own edges.

Env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import hashlib
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
        tx.run(
            f"""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
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


def load_chunks_batch(tx, batch: list[dict]) -> None:
    tx.run(
        """
        UNWIND $batch AS item
        MATCH (d:Document {id: item.doc_id})
        MERGE (c:Chunk {id: item.id})
        SET c.doc_id = item.doc_id,
            c.chunk_index = item.chunk_index,
            c.text = item.text,
            c.embedding = item.embedding,
            c.token_count = item.token_count,
            c.signature = item.signature
        MERGE (d)-[:HAS_CHUNK {idx: item.chunk_index}]->(c)
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
            e.display = coalesce(item.display, item.id),
            e.label = item.label,
            e.embedding = item.embedding,
            e.mentions = coalesce(item.mentions, []),
            e.doc_freq = coalesce(item.doc_freq, 0),
            e.mention_freq = coalesce(item.mention_freq, 0)
        """,
        batch=batch,
    )


def load_has_entity_batch(tx, batch: list[dict]) -> None:
    tx.run(
        """
        UNWIND $batch AS row
        MATCH (d:Document {id: row.doc_id})
        MATCH (e:Entity {id: row.entity_id})
        MERGE (d)-[h:HAS_ENTITY {start_char: row.start_char, end_char: row.end_char}]->(e)
        SET h.text = coalesce(row.entity_text, h.text)
        """,
        batch=batch,
    )


def load_relations_batch(tx, batch: list[dict]) -> None:
    tx.run(
        """
        UNWIND $batch AS rel
        MATCH (src:Entity {id: rel.head_id})
        MATCH (tgt:Entity {id: rel.tail_id})
        MATCH (:Document {id: rel.doc_id})
        MERGE (src)-[r:REL {label: rel.label, doc_id: rel.doc_id}]->(tgt)
        SET r.score = rel.score
        """,
        batch=batch,
    )


_WS_RE = re.compile(r"\s+")
_PUNCT_EDGE_RE = re.compile(r"^[\W_]+|[\W_]+$", flags=re.UNICODE)
_NUMERIC_RE = re.compile(r"^\d+([./-]\d+)*$")


def canonicalize_entity(text: str) -> str:
    t = (text or "").strip().lower()
    t = _WS_RE.sub(" ", t)
    t = _PUNCT_EDGE_RE.sub("", t)
    return t


def is_low_signal_entity(canonical: str) -> bool:
    if not canonical:
        return True
    if len(canonical) <= 1:
        return True
    if _NUMERIC_RE.match(canonical):
        return True
    return False


def build_allowed_entities(entities_path: Path, max_unique: int) -> dict[str, dict]:
    mention_freq: dict[str, int] = {}
    doc_sets: dict[str, set[str]] = {}
    surface_counts: dict[str, dict[str, int]] = {}
    label_counts: dict[str, dict[str, int]] = {}

    for o in iter_jsonl(entities_path):
        raw = (o.get("text") or "").strip()
        if not raw:
            continue
        cid = canonicalize_entity(raw)
        if is_low_signal_entity(cid):
            continue
        did = o.get("doc_id")
        if did is None:
            continue
        did = str(did)

        mention_freq[cid] = mention_freq.get(cid, 0) + 1
        doc_sets.setdefault(cid, set()).add(did)

        surface_counts.setdefault(cid, {})
        surface_counts[cid][raw] = surface_counts[cid].get(raw, 0) + 1

        lab = (o.get("label") or "").strip()
        if lab:
            label_counts.setdefault(cid, {})
            label_counts[cid][lab] = label_counts[cid].get(lab, 0) + 1

    ranked = sorted(
        doc_sets.keys(),
        key=lambda k: (len(doc_sets.get(k, ())), mention_freq.get(k, 0), k),
        reverse=True,
    )[:max_unique]

    out: dict[str, dict] = {}
    for cid in ranked:
        surfaces = surface_counts.get(cid, {})
        display = max(surfaces.items(), key=lambda kv: kv[1])[0] if surfaces else cid
        top_surfaces = [s for s, _ in sorted(surfaces.items(), key=lambda kv: kv[1], reverse=True)[:8]]
        labs = label_counts.get(cid, {})
        best_label = max(labs.items(), key=lambda kv: kv[1])[0] if labs else None
        out[cid] = {
            "id": cid,
            "display": display,
            "mentions": top_surfaces,
            "label": best_label,
            "doc_freq": len(doc_sets.get(cid, ())),
            "mention_freq": mention_freq.get(cid, 0),
        }
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


_TOK_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOK_RE.findall((text or "").lower())


def _simhash64(tokens: list[str]) -> int:
    if not tokens:
        return 0
    acc = [0] * 64
    for tok in tokens:
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        x = int.from_bytes(h, "big")
        for i in range(64):
            bit = (x >> i) & 1
            acc[i] += 1 if bit else -1
    out = 0
    for i, v in enumerate(acc):
        if v >= 0:
            out |= (1 << i)
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _iter_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    i = 0
    while i < len(text):
        c = text[i : i + chunk_size].strip()
        if c:
            chunks.append(c)
        if i + chunk_size >= len(text):
            break
        i += step
    return chunks


def build_selected_chunks(
    doc_rows: list[dict],
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_chunks_per_doc: int,
    max_chunks_total: int,
    near_dup_hamming: int,
) -> list[dict]:
    out: list[dict] = []
    sig_buckets: dict[int, list[int]] = {}
    exact_seen: set[str] = set()

    for doc in doc_rows:
        if len(out) >= max_chunks_total:
            break
        did = doc["id"]
        raw_chunks = _iter_chunks(doc.get("text") or "", chunk_size, chunk_overlap)
        candidates: list[tuple[int, dict]] = []
        doc_exact: set[str] = set()
        for idx, ctext in enumerate(raw_chunks):
            canon = " ".join(_tokenize(ctext))
            if not canon:
                continue
            csha = hashlib.sha256(canon.encode("utf-8")).hexdigest()
            if csha in doc_exact:
                continue
            doc_exact.add(csha)
            toks = canon.split()
            sig = _simhash64(toks)
            score = len(set(toks))
            candidates.append(
                (
                    score,
                    {
                        "id": stable_chunk_id(did, idx, csha),
                        "doc_id": did,
                        "chunk_index": idx,
                        "text": ctext,
                        "token_count": len(toks),
                        "signature": str(sig),
                        "_sig": sig,
                        "_canon_sha": csha,
                    },
                )
            )

        chosen_for_doc = 0
        for _, payload in sorted(candidates, key=lambda x: x[0], reverse=True):
            if chosen_for_doc >= max_chunks_per_doc or len(out) >= max_chunks_total:
                break
            csha = payload["_canon_sha"]
            if csha in exact_seen:
                continue
            sig = payload["_sig"]
            key = sig >> 48
            near = False
            for prev in sig_buckets.get(key, []):
                if _hamming(sig, prev) <= near_dup_hamming:
                    near = True
                    break
            if near:
                continue
            exact_seen.add(csha)
            sig_buckets.setdefault(key, []).append(sig)
            payload.pop("_sig", None)
            payload.pop("_canon_sha", None)
            out.append(payload)
            chosen_for_doc += 1
    return out


def stable_chunk_id(doc_id: str, chunk_index: int, chunk_sha: str) -> str:
    return f"{doc_id}:c{chunk_index}:{chunk_sha[:10]}"


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Upload documents + graph to Neo4j")
    ap.add_argument("--docs", type=Path, default=root / "data_clean/msoe/documents.jsonl")
    ap.add_argument("--entities", type=Path, default=root / "data_clean/extracted/entities.jsonl")
    ap.add_argument("--relations", type=Path, default=root / "data_clean/extracted/relations.jsonl")
    ap.add_argument("--max-entities", type=int, default=50_000)
    ap.add_argument("--max-relations", type=int, default=175_000)
    ap.add_argument("--min-rel-score", type=float, default=0.05, help="Drop relations with score below this threshold")
    ap.add_argument("--top-rel-per-doc", type=int, default=200, help="Keep only top-K relations per document by score")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--chunk-size", type=int, default=900, help="Chunk size (characters)")
    ap.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap (characters)")
    ap.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=6,
        help="Max chunks retained per document (more coverage for long catalog pages; was 3)",
    )
    ap.add_argument("--max-chunks", type=int, default=12000, help="Global chunk cap for Neo4j upload")
    ap.add_argument("--chunk-near-dup-hamming", type=int, default=3, help="Near-duplicate chunk simhash threshold")
    ap.add_argument("--document-text-from-chunks", action="store_true", help="Store only selected chunks in Document.text")
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
    print(f"Allowed canonical entities: {len(allowed_set):,}")

    print("Indexing documents.jsonl...")
    doc_by_id = load_documents_jsonl(args.docs)

    doc_ids: set[str] = set()
    for o in iter_jsonl(args.entities):
        t = (o.get("text") or "").strip()
        cid = canonicalize_entity(t)
        if cid not in allowed_set:
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

    selected_chunks = build_selected_chunks(
        doc_rows,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_chunks_per_doc=args.max_chunks_per_doc,
        max_chunks_total=args.max_chunks,
        near_dup_hamming=args.chunk_near_dup_hamming,
    )
    print(f"Selected {len(selected_chunks):,} diverse chunks for upload")
    chunks_by_doc: dict[str, list[dict]] = {}
    for c in selected_chunks:
        chunks_by_doc.setdefault(c["doc_id"], []).append(c)
    if args.document_text_from_chunks:
        for d in doc_rows:
            kept = sorted(chunks_by_doc.get(d["id"], []), key=lambda x: x["chunk_index"])
            if kept:
                d["text"] = "\n\n".join(c["text"] for c in kept)

    print(f"Uploading {len(doc_rows):,} Document nodes with embeddings...")
    with driver.session() as session:
        for i in range(0, len(doc_rows), args.batch_size):
            batch = doc_rows[i : i + args.batch_size]
            texts = [b["text"] for b in batch]
            embs = st_model.encode(texts, show_progress_bar=False).tolist()
            for b, emb in zip(batch, embs):
                b["embedding"] = emb
            session.execute_write(load_documents_batch, batch)

    print(f"Uploading {len(selected_chunks):,} Chunk nodes with embeddings...")
    with driver.session() as session:
        for i in range(0, len(selected_chunks), args.batch_size):
            batch = selected_chunks[i : i + args.batch_size]
            texts = [b["text"] for b in batch]
            embs = st_model.encode(texts, show_progress_bar=False).tolist()
            for b, emb in zip(batch, embs):
                b["embedding"] = emb
            session.execute_write(load_chunks_batch, batch)

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
        cid = canonicalize_entity(t)
        if cid not in allowed_set:
            continue
        did = str(o.get("doc_id", ""))
        if did not in doc_ids or did not in doc_by_id:
            continue
        key = (did, cid, o.get("start_char"), o.get("end_char"))
        if key in seen_m:
            continue
        seen_m.add(key)
        has_rows.append(
            {
                "doc_id": did,
                "entity_id": cid,
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
        topk: dict[str, list[tuple[float, int, dict]]] = {}
        rel_seq = 0

        with args.relations.open(encoding="utf-8", errors="replace") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                lab = r.get("label")
                if not lab or lab == "Other":
                    continue
                sc = float(r.get("score") or 0.0)
                if sc < float(args.min_rel_score):
                    continue
                doc_id = str(r.get("doc_id", ""))
                if doc_id not in doc_by_id:
                    continue

                ht_raw = (r.get("head") or {}).get("text", "").strip()
                tt_raw = (r.get("tail") or {}).get("text", "").strip()
                hid = canonicalize_entity(ht_raw)
                tid = canonicalize_entity(tt_raw)
                if hid not in allowed_set or tid not in allowed_set:
                    continue

                payload = {"head_id": hid, "tail_id": tid, "label": lab, "score": sc, "doc_id": doc_id}
                heap = topk.setdefault(doc_id, [])
                rel_seq += 1
                item = (sc, rel_seq, payload)
                if len(heap) < int(args.top_rel_per_doc):
                    heapq.heappush(heap, item)
                else:
                    if sc > heap[0][0]:
                        heapq.heapreplace(heap, item)

        for heap in topk.values():
            if accepted >= args.max_relations:
                break
            for sc, _, payload in sorted(heap, key=lambda x: x[0], reverse=True):
                if accepted >= args.max_relations:
                    break
                rel_batch.append(payload)
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
