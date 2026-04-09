import argparse
import json
from pathlib import Path

import spacy
from spacy.tokens import Doc, DocBin
from spacy.util import filter_spans


def ensure_rel_extension() -> None:
    if not Doc.has_extension("rel"):
        Doc.set_extension("rel", default={})


def main() -> int:
    ap = argparse.ArgumentParser(description="Build spaCy DocBin for relation extraction training")
    ap.add_argument("--examples", type=Path, required=True, help="JSONL with text + e1/e2 + relation")
    ap.add_argument("--out", type=Path, required=True, help="Output .spacy DocBin path")
    ap.add_argument("--entity-label", type=str, default="ENT", help="Entity label to assign")
    ap.add_argument("--negative-label", type=str, default=None, help="Optional label to treat as negative/no-relation")
    args = ap.parse_args()

    ensure_rel_extension()

    nlp = spacy.blank("en")
    db = DocBin(store_user_data=True)

    # Collect labels so doc._.rel rows can include all labels (multi-label setup).
    labels = set()
    raw_examples = []
    for line in args.examples.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        raw_examples.append(ex)
        rel = ex.get("relation")
        if rel and rel != args.negative_label:
            labels.add(rel)
    labels = sorted(labels)

    for ex in raw_examples:
        text = ex["text"]
        doc = nlp.make_doc(text)
        e1 = ex.get("e1")
        e2 = ex.get("e2")
        if not (e1 and e2):
            continue
        s1 = doc.char_span(e1["char_start"], e1["char_end"], label=args.entity_label, alignment_mode="contract")
        s2 = doc.char_span(e2["char_start"], e2["char_end"], label=args.entity_label, alignment_mode="contract")
        if s1 is None or s2 is None:
            continue
        ents = filter_spans([s1, s2])
        # Relation extraction requires 2 distinct entities; skip degenerate overlaps.
        if len(ents) != 2:
            continue
        s1, s2 = ents[0], ents[1]
        doc.ents = [s1, s2]

        # Gold relation annotations live in doc._.rel
        doc._.rel = {}
        key = (s1.start, s2.start)
        gold_rel = ex.get("relation")
        if gold_rel and gold_rel != args.negative_label:
            doc._.rel[key] = {lab: 1.0 if lab == gold_rel else 0.0 for lab in labels}
        else:
            doc._.rel[key] = {lab: 0.0 for lab in labels}

        db.add(doc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(db.to_bytes())
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

