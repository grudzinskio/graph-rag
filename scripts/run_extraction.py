import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import spacy

# Registers @Language.factory("relation_extractor") before loading a saved RE pipeline.
import extraction_spacy.relation_extractor  # noqa: F401


def iter_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def setup_logging(log_file: Path | None) -> logging.Logger:
    """Log to stderr; optionally mirror to a file."""
    log = logging.getLogger("run_extraction")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    log.addHandler(logging.StreamHandler(sys.stderr))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log.addHandler(logging.FileHandler(log_file, encoding="utf-8"))
    for h in log.handlers:
        h.setFormatter(fmt)
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description="Run NER + RE over cleaned documents.jsonl")
    ap.add_argument("--docs", type=Path, required=True, help="Cleaned documents.jsonl")
    ap.add_argument("--ner-model", type=Path, required=True, help="Path to trained spaCy NER model directory")
    ap.add_argument("--re-model", type=Path, required=True, help="Path to trained spaCy RE model directory")
    ap.add_argument("--out", type=Path, default=Path("data_clean/extracted"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pair-max-tokens", type=int, default=100, help="Max token distance for candidate pairs (fallback)")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Log progress every N documents (0 to disable)",
    )
    ap.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Also write the same log lines to this file",
    )
    args = ap.parse_args()

    log = setup_logging(args.log_file)
    t0 = time.perf_counter()

    log.info("Loading NER: %s", args.ner_model)
    nlp_ner = spacy.load(args.ner_model)
    log.info("Loading RE:  %s", args.re_model)
    nlp_re = spacy.load(args.re_model)

    entities_out = []
    relations_out = []

    count = 0
    n_ent = 0
    n_rel = 0
    for docrec in iter_jsonl(args.docs):
        if args.limit and count >= args.limit:
            break
        count += 1
        text = docrec["text"]
        base = nlp_ner(text)

        # Use NER entities and run RE on a doc containing those entities.
        # The RE pipeline expects doc.ents to exist.
        doc_for_re = nlp_re.make_doc(text)
        doc_for_re.ents = list(base.ents)
        doc_for_re = nlp_re(doc_for_re)

        doc_id = docrec["id"]
        doc_ent_count = 0
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
            doc_ent_count += 1

        rel: dict[Any, Any] = getattr(doc_for_re._, "rel", {}) or {}
        doc_rel_count = 0
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
            doc_rel_count += 1

        n_ent += doc_ent_count
        n_rel += doc_rel_count
        if args.progress_every and count % args.progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = count / elapsed if elapsed > 0 else 0.0
            log.info(
                "docs=%s | entities_total=%s | relations_total=%s | %.2f docs/s",
                count,
                n_ent,
                n_rel,
                rate,
            )

    write_jsonl(args.out / "entities.jsonl", entities_out)
    write_jsonl(args.out / "relations.jsonl", relations_out)
    elapsed = time.perf_counter() - t0
    log.info(
        "Done. docs=%s entities=%s relations=%s in %.1fs → %s and %s",
        count,
        len(entities_out),
        len(relations_out),
        elapsed,
        args.out / "entities.jsonl",
        args.out / "relations.jsonl",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

