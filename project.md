# Project notes (our implementation)

We built **reproducible data preprocessing** and a **spaCy-based extraction stack** focused on:

- **Preprocessing**: turn noisy `scraped_data/**/text/*.txt` into a deterministic cleaned corpus under `data_clean/`, plus converters for SemEval 2010 Task-8 and FewRel (when you provide the raw files).
- **Extraction**: spaCy **NER** + a supervised **Relation Extraction (RE)** model (custom spaCy trainable component) trained from SemEval/FewRel-style `(e1,e2,relation)` examples.

This does **not** build the graph store or RAG agent yet (those are downstream steps).

---

## Repo structure we added

- `scripts/`: reproducible CLIs
  - `scripts/preprocess.py`: main cleaning + dataset conversion entrypoint
  - `scripts/build_spacy_ner_data.py`: JSONL → `train.spacy` / `dev.spacy` for NER
  - `scripts/build_spacy_re_data.py`: JSONL → `train.spacy` / `dev.spacy` for RE
  - `scripts/train_ner.py`: wrapper around `python -m spacy train` for NER
  - `scripts/train_re.py`: wrapper around `python -m spacy train` for RE (loads custom code)
  - `scripts/run_extraction.py`: run trained NER+RE over cleaned docs → JSONL outputs
- `configs/`
  - `configs/ner.cfg`: spaCy training config for NER
  - `configs/re.cfg`: spaCy training config for RE (custom component)
- `extraction_spacy/`
  - `extraction_spacy/relation_extractor.py`: custom spaCy relation extractor component + model registry entries
- `data_clean/` (generated, ignored by git)
  - cleaned corpora + manifests + optional quarantines

---

## Preprocessing (what we do)

### Inputs supported

#### 1) MSOE scraped pages

From our existing scraper output:

- `scraped_data/sitemap/text/*.txt`
- `scraped_data/catalog/text/*.txt`

#### 2) SemEval 2010 Task-8 (when you provide raw files)

Provide the canonical train/test text files (with `<e1>...</e1>` and `<e2>...</e2>` tags) via CLI flags.

#### 3) FewRel (when you provide raw file)

Provide a FewRel JSON file (common `relation -> instances` structure).

### Outputs produced

#### Cleaned MSOE corpus (deterministic JSONL)

`data_clean/msoe/documents.jsonl` containing one JSON object per document:

- `id`: stable hash-based ID
- `dataset`: `msoe_sitemap` or `msoe_catalog`
- `text`: cleaned text
- `source_path`, `source_sha256`
- `meta`: read/clean stats + content hash

#### Benchmark examples (deterministic JSONL)

If you pass SemEval/FewRel inputs, it also writes:

- `data_clean/benchmarks/semeval2010_task8/examples.jsonl`
- `data_clean/benchmarks/fewrel/examples.jsonl`

Each benchmark row is normalized to:

- `text`
- `e1` (`char_start`, `char_end`, `text`)
- `e2` (`char_start`, `char_end`, `text`)
- `relation`
- `split` (SemEval only)

#### Run manifest (reproducibility)

Every run writes `data_clean/manifests/preprocess_<run_id>.json` capturing:

- exact CLI argv
- outputs written
- counts (seen/kept/deduped/etc.)
- Python version

### Cleaning rules (MSOE)

Implemented in `scripts/preprocess.py`:

- **Normalization**: NFKC, newline normalization, strip control chars, collapse whitespace.
- **Boilerplate line drops**: removes common UI/chrome lines like “Catalog Navigation”, “Menu”, “Search”, and footer boilerplate.
- **Menu block removal (catalog)**: drops the “Select a Catalog” mega-section until an end marker.
- **Exact de-duplication**: documents are deduped by cleaned content sha256.

### PDF handling (we parse PDFs properly)

Some `scraped_data/catalog/text/*pdf.txt` files contain **raw PDF bytes** (start with `%PDF-`).

`scripts/preprocess.py` detects this and extracts text using **PyMuPDF** (`pymupdf` / `fitz`).

- If parsing fails:
  - with `--allow-pdf-failures`, it quarantines an error report under `data_clean/quarantine/`
  - otherwise it raises an error

---

## spaCy extraction

### NER

NER training is configured in `configs/ner.cfg`.

#### Data format

spaCy `DocBin` (`.spacy`) with entity spans.

#### Build NER training data

We convert benchmark JSONL (SemEval/FewRel) into DocBin via:

```bash
python scripts/build_spacy_ner_data.py --examples data_clean/benchmarks/semeval2010_task8/examples.jsonl --out data_raw/tmp/ner_train.spacy
```

Then split into train/dev as you prefer (or generate separate JSONLs per split and build them independently).

#### Train NER model

```bash
python scripts/train_ner.py --train <train.spacy> --dev <dev.spacy> --output models/ner
```

### Relation Extraction (supervised)

RE training is configured in `configs/re.cfg` and implemented as a custom spaCy component in:

- `extraction_spacy/relation_extractor.py`

This is a **trainable spaCy pipeline component** that stores predictions in:

- `doc._.rel[(ent1.start, ent2.start)] = {label: score, ...}`

#### Build RE training data

```bash
python scripts/build_spacy_re_data.py --examples data_clean/benchmarks/semeval2010_task8/examples.jsonl --out data_raw/tmp/re_train.spacy
```

#### Train RE model

```bash
python scripts/train_re.py --train <train.spacy> --dev <dev.spacy> --output models/re
```

### Running extraction over cleaned docs

Once you have trained models:

```bash
python scripts/run_extraction.py ^
  --docs data_clean/msoe/documents.jsonl ^
  --ner-model models/ner/model-best ^
  --re-model models/re/model-best ^
  --out data_clean/extracted ^
  --limit 100
```

Outputs:

- `data_clean/extracted/entities.jsonl`
- `data_clean/extracted/relations.jsonl`

---

## Repro steps (end-to-end)

1) Install deps:

```bash
python -m pip install -r requirements.txt
```

1) Build cleaned data:

```bash
python scripts/preprocess.py --allow-pdf-failures
```

1) Provide SemEval/FewRel raw files, rerun preprocess with flags:

```bash
python scripts/preprocess.py --allow-pdf-failures --semeval-train <path> --semeval-test <path> --fewrel-json <path>
```

1) Convert benchmark JSONL → spaCy DocBin(s), train NER + RE, then run `scripts/run_extraction.py`.
