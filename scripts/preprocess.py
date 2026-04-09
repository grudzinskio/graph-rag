import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable


MSOE_DROP_LINE_PATTERNS = [
    re.compile(r"^\s*Global Search\s*$", re.IGNORECASE),
    re.compile(r"^\s*Catalog Search\s*$", re.IGNORECASE),
    re.compile(r"^\s*Catalog Navigation\s*$", re.IGNORECASE),
    re.compile(r"^\s*Back to Top\s*$", re.IGNORECASE),
    re.compile(r"^\s*Skip to Main Content\s*$", re.IGNORECASE),
    re.compile(r"^\s*Menu\s*$", re.IGNORECASE),
    re.compile(r"^\s*Search\s*$", re.IGNORECASE),
    re.compile(r"^\s*submit\s*$", re.IGNORECASE),
    re.compile(r"^\s*Resources For\.\.\.\s*$", re.IGNORECASE),
    re.compile(r"^\s*Connect With MSOE\s*$", re.IGNORECASE),
    re.compile(r"^\s*MSOE University\s*$", re.IGNORECASE),
    re.compile(r"^\s*Print-Friendly Page.*$", re.IGNORECASE),
    re.compile(r"^\s*Powered by\s+Modern Campus Catalog.*$", re.IGNORECASE),
    re.compile(r"^\s*©\s*\d{4}\s+Milwaukee School of Engineering.*$", re.IGNORECASE),
]

MSOE_SECTION_START_PATTERNS = [
    # Common blocks that create very long menus. We drop until an end marker.
    re.compile(r"^\s*Select a Catalog\s*$", re.IGNORECASE),
]
MSOE_SECTION_END_PATTERNS = [
    re.compile(r"^\s*HELP\s*$", re.IGNORECASE),
    re.compile(r"^\s*Course Descriptions\s*$", re.IGNORECASE),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def read_text_maybe_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    """
    Reads a file and returns text + meta describing how it was obtained.

    - If the file begins with %PDF-, tries to extract text from the PDF bytes using PyMuPDF.
    - Otherwise reads as UTF-8 text (with replacement).
    """
    raw = path.read_bytes()
    meta: dict[str, Any] = {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
    if raw.startswith(b"%PDF-"):
        meta["detected_format"] = "pdf_bytes"
        try:
            import fitz  # PyMuPDF
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "PDF bytes detected but PyMuPDF is not installed. "
                "Install with `pip install pymupdf`."
            ) from e
        doc = fitz.open(stream=raw, filetype="pdf")
        pages = []
        for i in range(doc.page_count):
            pages.append(doc.load_page(i).get_text("text"))
        text = "\n".join(pages)
        meta["pdf_pages"] = doc.page_count
        meta["read_mode"] = "pymupdf"
        return text, meta

    meta["detected_format"] = "text"
    meta["read_mode"] = "utf8_replace"
    return raw.decode("utf-8", errors="replace"), meta


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove other control chars except newline + tab
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or (ord(ch) >= 32))
    # Collapse whitespace on each line and drop empty lines
    lines = []
    for line in text.split("\n"):
        s = re.sub(r"\s+", " ", line).strip()
        if s:
            lines.append(s)
    return "\n".join(lines).strip()


def clean_msoe_text(text: str) -> tuple[str, dict[str, Any]]:
    """
    Heuristic cleanup for MSOE scraped pages (catalog + sitemap text).
    Returns cleaned text and a small stats dict.
    """
    original_lines = [ln.strip() for ln in text.split("\n")]
    cleaned: list[str] = []

    dropping_section = False
    dropped_by_pattern = 0
    dropped_in_section = 0

    for ln in original_lines:
        if not ln:
            continue
        if any(pat.match(ln) for pat in MSOE_SECTION_START_PATTERNS):
            dropping_section = True
            dropped_in_section += 1
            continue
        if dropping_section:
            dropped_in_section += 1
            if any(pat.match(ln) for pat in MSOE_SECTION_END_PATTERNS):
                dropping_section = False
            continue
        if any(pat.match(ln) for pat in MSOE_DROP_LINE_PATTERNS):
            dropped_by_pattern += 1
            continue
        cleaned.append(ln)

    out = "\n".join(cleaned).strip()
    stats = {
        "lines_in": len([x for x in original_lines if x]),
        "lines_out": len(cleaned),
        "dropped_by_pattern": dropped_by_pattern,
        "dropped_in_section": dropped_in_section,
    }
    return out, stats


@dataclasses.dataclass(frozen=True)
class DocRecord:
    id: str
    dataset: str
    split: str | None
    text: str
    source_path: str
    source_sha256: str
    meta: dict[str, Any]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def iter_msoe_sources(scraped_data_dir: Path) -> Iterable[tuple[str, Path]]:
    """
    Yields (dataset_name, path) for MSOE sources.
    """
    for dataset_name in ("sitemap", "catalog"):
        base = scraped_data_dir / dataset_name / "text"
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.txt")):
            yield f"msoe_{dataset_name}", p


def build_msoe_docs(
    scraped_data_dir: Path,
    limit: int | None,
    quarantine_dir: Path,
    allow_pdf_failures: bool,
) -> tuple[list[DocRecord], dict[str, Any]]:
    docs: list[DocRecord] = []
    seen_content_hash: set[str] = set()
    stats: dict[str, Any] = {
        "docs_seen": 0,
        "docs_kept": 0,
        "docs_deduped": 0,
        "docs_empty_after_clean": 0,
        "pdf_detected": 0,
        "pdf_extracted": 0,
        "pdf_failed": 0,
        "read_failed": 0,
        "quarantined": 0,
    }

    for dataset, p in iter_msoe_sources(scraped_data_dir):
        stats["docs_seen"] += 1
        if limit is not None and stats["docs_seen"] > limit:
            break

        try:
            raw_text, read_meta = read_text_maybe_pdf(p)
        except Exception as e:
            if allow_pdf_failures:
                # Attempt to identify PDF bytes even when extraction fails.
                try:
                    head = p.read_bytes()[:5]
                    if head == b"%PDF-":
                        stats["pdf_detected"] += 1
                        stats["pdf_failed"] += 1
                    else:
                        stats["read_failed"] += 1
                except Exception:
                    stats["read_failed"] += 1
                q_path = quarantine_dir / "msoe" / (p.name + ".error.json")
                q_path.parent.mkdir(parents=True, exist_ok=True)
                q_path.write_text(
                    json.dumps({"path": str(p), "error": str(e)}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                stats["quarantined"] += 1
                continue
            raise

        if read_meta.get("detected_format") == "pdf_bytes":
            stats["pdf_detected"] += 1
            stats["pdf_extracted"] += 1

        norm = normalize_text(raw_text)
        cleaned, clean_stats = clean_msoe_text(norm)
        cleaned = normalize_text(cleaned)

        if not cleaned:
            stats["docs_empty_after_clean"] += 1
            continue

        content_h = sha256_bytes(cleaned.encode("utf-8"))
        if content_h in seen_content_hash:
            stats["docs_deduped"] += 1
            continue
        seen_content_hash.add(content_h)

        doc_id = stable_id(dataset, str(p.relative_to(scraped_data_dir)), read_meta["sha256"])
        docs.append(
            DocRecord(
                id=doc_id,
                dataset=dataset,
                split=None,
                text=cleaned,
                source_path=str(p),
                source_sha256=read_meta["sha256"],
                meta={"read": read_meta, "clean": clean_stats, "content_sha256": content_h},
            )
        )
        stats["docs_kept"] += 1

    return docs, stats


def parse_semeval_task8_file(path: Path, split: str) -> Iterable[dict[str, Any]]:
    """
    Parses SemEval 2010 Task-8 formatted files.

    Expected 4-line blocks:
      1) <id>\\t\"<sentence with <e1> ... </e1> and <e2> ... </e2> tags>\"
      2) relation label
      3) comment (ignored)
      4) blank line
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        if not header:
            i += 1
            continue
        if i + 1 >= len(lines):
            break
        rel = lines[i + 1].strip()

        m = re.match(r'^(\d+)\s+\t\s+\"(.*)\"$', header, flags=re.VERBOSE)
        if not m:
            # Some distributions use: 1\t"Sentence"
            m2 = re.match(r'^(\d+)\t\"(.*)\"$', header)
            if not m2:
                raise ValueError(f"Unrecognized SemEval line format: {header[:120]}")
            m = m2
        ex_id = m.group(1)
        sent = m.group(2)

        # Extract entities and remove tags
        e1_start = sent.find("<e1>")
        e1_end = sent.find("</e1>")
        e2_start = sent.find("<e2>")
        e2_end = sent.find("</e2>")
        if min(e1_start, e1_end, e2_start, e2_end) < 0:
            raise ValueError(f"Missing entity tags for example {ex_id}")

        # Build clean sentence and char spans after tag removal.
        # Approach: replace tags with "" and then locate entity strings.
        sent_no_tags = (
            sent.replace("<e1>", "")
            .replace("</e1>", "")
            .replace("<e2>", "")
            .replace("</e2>", "")
        )
        # entity texts:
        e1_text = sent[e1_start + 4 : e1_end]
        e2_text = sent[e2_start + 4 : e2_end]
        # locate in tagless text (first occurrence)
        e1_char = sent_no_tags.find(e1_text)
        e2_char = sent_no_tags.find(e2_text)
        if e1_char < 0 or e2_char < 0:
            raise ValueError(f"Could not locate entities after tag removal for {ex_id}")

        yield {
            "id": stable_id("semeval2010", split, ex_id),
            "dataset": "semeval2010_task8",
            "split": split,
            "text": sent_no_tags,
            "e1": {"text": e1_text, "char_start": e1_char, "char_end": e1_char + len(e1_text)},
            "e2": {"text": e2_text, "char_start": e2_char, "char_end": e2_char + len(e2_text)},
            "relation": rel,
            "source_path": str(path),
        }

        i += 4


def load_fewrel_json(path: Path) -> Iterable[dict[str, Any]]:
    """
    FewRel is distributed as a JSON mapping relation -> list[instances].
    Instance schema varies; we support the common `tokens`, `h`, `t` format.
    """
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("FewRel JSON expected to be an object mapping relation -> instances")

    for rel, instances in data.items():
        for idx, ex in enumerate(instances):
            tokens = ex.get("tokens")
            if not tokens:
                continue
            text = " ".join(tokens)
            h = ex.get("h") or {}
            t = ex.get("t") or {}
            h_pos = (h.get("pos") or [[None, None]])[0]
            t_pos = (t.get("pos") or [[None, None]])[0]

            def tok_span_to_char(pos_pair: list[int] | tuple[int, int]):
                s_tok, e_tok = int(pos_pair[0]), int(pos_pair[1])
                # char start = len of tokens before + spaces
                char_start = sum(len(tok) + 1 for tok in tokens[:s_tok])
                span_text = " ".join(tokens[s_tok:e_tok])
                return char_start, char_start + len(span_text), span_text

            e1_char_start, e1_char_end, e1_text = tok_span_to_char(h_pos)
            e2_char_start, e2_char_end, e2_text = tok_span_to_char(t_pos)

            yield {
                "id": stable_id("fewrel", rel, str(idx)),
                "dataset": "fewrel",
                "split": None,
                "text": text,
                "e1": {"text": e1_text, "char_start": e1_char_start, "char_end": e1_char_end},
                "e2": {"text": e2_text, "char_start": e2_char_start, "char_end": e2_char_end},
                "relation": rel,
                "source_path": str(path),
            }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible preprocessing for graph-rag")
    ap.add_argument("--scraped-data", type=Path, default=Path("scraped_data"))
    ap.add_argument("--out", type=Path, default=Path("data_clean"))
    ap.add_argument("--limit-msoe", type=int, default=None, help="Limit MSOE docs for quick runs")
    ap.add_argument(
        "--allow-pdf-failures",
        action="store_true",
        help="If PDF extraction fails, quarantine and continue instead of erroring",
    )

    ap.add_argument("--semeval-train", type=Path, default=None)
    ap.add_argument("--semeval-test", type=Path, default=None)
    ap.add_argument("--fewrel-json", type=Path, default=None)

    args = ap.parse_args()

    out_root: Path = args.out
    quarantine_dir = out_root / "quarantine"
    manifests_dir = out_root / "manifests"
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    run_id = stable_id("preprocess", now, str(os.getpid()))

    all_manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": now,
        "argv": sys.argv,
        "outputs": {},
        "stats": {},
        "versions": {"python": sys.version},
    }

    # MSOE
    msoe_docs, msoe_stats = build_msoe_docs(
        scraped_data_dir=args.scraped_data,
        limit=args.limit_msoe,
        quarantine_dir=quarantine_dir,
        allow_pdf_failures=args.allow_pdf_failures,
    )
    msoe_out = out_root / "msoe" / "documents.jsonl"
    write_jsonl(
        msoe_out,
        (
            {
                "id": d.id,
                "dataset": d.dataset,
                "split": d.split,
                "text": d.text,
                "source_path": d.source_path,
                "source_sha256": d.source_sha256,
                "meta": d.meta,
            }
            for d in msoe_docs
        ),
    )
    all_manifest["outputs"]["msoe_documents"] = str(msoe_out)
    all_manifest["stats"]["msoe"] = msoe_stats

    # SemEval
    if args.semeval_train and args.semeval_test:
        semeval_rows = []
        for row in parse_semeval_task8_file(args.semeval_train, "train"):
            semeval_rows.append(row)
        for row in parse_semeval_task8_file(args.semeval_test, "test"):
            semeval_rows.append(row)
        semeval_out = out_root / "benchmarks" / "semeval2010_task8" / "examples.jsonl"
        write_jsonl(semeval_out, semeval_rows)
        all_manifest["outputs"]["semeval_examples"] = str(semeval_out)
        all_manifest["stats"]["semeval"] = {"examples": len(semeval_rows)}

    # FewRel
    if args.fewrel_json:
        fewrel_rows = list(load_fewrel_json(args.fewrel_json))
        fewrel_out = out_root / "benchmarks" / "fewrel" / "examples.jsonl"
        write_jsonl(fewrel_out, fewrel_rows)
        all_manifest["outputs"]["fewrel_examples"] = str(fewrel_out)
        all_manifest["stats"]["fewrel"] = {"examples": len(fewrel_rows)}

    manifest_path = manifests_dir / f"preprocess_{run_id}.json"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(all_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

