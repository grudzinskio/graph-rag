import argparse
import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def main() -> int:
    ap = argparse.ArgumentParser(description="Split a spaCy DocBin (.spacy) into train/dev deterministically.")
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="Input .spacy file")
    ap.add_argument("--train-out", type=Path, required=True, help="Output train .spacy file")
    ap.add_argument("--dev-out", type=Path, required=True, help="Output dev .spacy file")
    ap.add_argument("--dev-ratio", type=float, default=0.2, help="Fraction to allocate to dev (default 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of docs before splitting")
    args = ap.parse_args()

    nlp = spacy.blank("en")
    db = DocBin().from_bytes(args.inp.read_bytes())
    docs = list(db.get_docs(nlp.vocab))

    if args.limit is not None:
        docs = docs[: args.limit]

    rng = random.Random(args.seed)
    rng.shuffle(docs)

    dev_n = int(round(len(docs) * args.dev_ratio))
    dev_docs = docs[:dev_n]
    train_docs = docs[dev_n:]

    train_db = DocBin(store_user_data=True)
    dev_db = DocBin(store_user_data=True)
    for d in train_docs:
        train_db.add(d)
    for d in dev_docs:
        dev_db.add(d)

    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.dev_out.parent.mkdir(parents=True, exist_ok=True)
    args.train_out.write_bytes(train_db.to_bytes())
    args.dev_out.write_bytes(dev_db.to_bytes())

    print(f"Docs total: {len(docs)}")
    print(f"Train: {len(train_docs)} -> {args.train_out}")
    print(f"Dev:   {len(dev_docs)} -> {args.dev_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

