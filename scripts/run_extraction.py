import argparse
import json
from pathlib import Path
from typing import Any

import spacy


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run NER + RE over cleaned documents.jsonl")
    ap.add_argument("--docs", type=Path, required=True, help="Cleaned documents.jsonl")
    ap.add_argument("--ner-model", type=Path, required=True, help="Path to trained spaCy NER model directory")
    ap.add_argument("--re-model", type=Path, required=True, help="Path to trained spaCy RE model directory")
    ap.add_argument("--out", type=Path, default=Path("data_clean/extracted"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pair-max-tokens", type=int, default=100, help="Max token distance for candidate pairs (fallback)")
    args = ap.parse_args()

    nlp_ner = spacy.load(args.ner_model)
    nlp_re = spacy.load(args.re_model)

    entities_out = []
    relations_out = []

    count = 0
    for docrec in iter_jsonl(args.docs):
        count += 1
        if args.limit and count > args.limit:
            break
        text = docrec["text"]
        base = nlp_ner(text)

        # Use NER entities and run RE on a doc containing those entities.
        # The RE pipeline expects doc.ents to exist.
        doc_for_re = nlp_re.make_doc(text)
        doc_for_re.ents = list(base.ents)
        doc_for_re = nlp_re(doc_for_re)

        doc_id = docrec["id"]
        for ent in doc_for_re.ents:
            entities_out.append(
                {
                    "doc_id": doc_id,
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )

        rel: dict[Any, Any] = getattr(doc_for_re._, "rel", {}) or {}
        for (s1, s2), scores in rel.items():
            ent1 = next((e for e in doc_for_re.ents if e.start == s1), None)
            ent2 = next((e for e in doc_for_re.ents if e.start == s2), None)
            if ent1 is None or ent2 is None:
                continue
            best_label = None
            best_score = 0.0
            for lab, sc in scores.items():
                if sc > best_score:
                    best_label, best_score = lab, float(sc)
            if best_label is None:
                continue
            relations_out.append(
                {
                    "doc_id": doc_id,
                    "head": {"text": ent1.text, "start_char": ent1.start_char, "end_char": ent1.end_char},
                    "tail": {"text": ent2.text, "start_char": ent2.start_char, "end_char": ent2.end_char},
                    "label": best_label,
                    "score": best_score,
                }
            )

    write_jsonl(args.out / "entities.jsonl", entities_out)
    write_jsonl(args.out / "relations.jsonl", relations_out)
    print(f"Wrote {args.out / 'entities.jsonl'} and {args.out / 'relations.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

