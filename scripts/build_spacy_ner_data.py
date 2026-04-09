import argparse
import json
from pathlib import Path

import spacy
from spacy.util import filter_spans
from spacy.tokens import DocBin


def main() -> int:
    ap = argparse.ArgumentParser(description="Build spaCy DocBin for NER training")
    ap.add_argument("--examples", type=Path, required=True, help="JSONL with text + e1/e2 spans")
    ap.add_argument("--out", type=Path, required=True, help="Output .spacy DocBin path")
    ap.add_argument("--label", type=str, default="ENT", help="Entity label to assign")
    args = ap.parse_args()

    nlp = spacy.blank("en")
    db = DocBin(store_user_data=False)

    for line in args.examples.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        text = ex["text"]
        doc = nlp.make_doc(text)
        ents = []
        for ent in (ex.get("e1"), ex.get("e2")):
            if not ent:
                continue
            span = doc.char_span(ent["char_start"], ent["char_end"], label=args.label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        # SemEval can produce overlapping spans after alignment; filter_spans keeps the
        # longest non-overlapping spans and dedupes exact duplicates.
        ents = filter_spans(ents)
        doc.ents = ents
        db.add(doc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(db.to_bytes())
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

