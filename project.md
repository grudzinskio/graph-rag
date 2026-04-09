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
- `data_clean/` (generated outputs; **may be committed** so others can use corpora + extraction without retraining)
  - cleaned corpora + manifests + optional quarantines
- `models/` (trained spaCy pipelines; **may be committed**)

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
python scripts/build_spacy_ner_data.py --examples data_clean/benchmarks/semeval2010_task8/examples.jsonl --out data_raw/tmp/ner_all.spacy
```

Then split with `scripts/split_spacy_docbin.py` (see README).

#### Train NER model

```bash
python scripts/train_ner.py --train data_raw/tmp/ner_train.spacy --dev data_raw/tmp/ner_dev.spacy --output models/ner
```

### Relation Extraction (supervised)

RE training is configured in `configs/re.cfg` and implemented as a custom spaCy component in:

- `extraction_spacy/relation_extractor.py`

This is a **trainable spaCy pipeline component** that stores predictions in:

- `doc._.rel[(ent1.start, ent2.start)] = {label: score, ...}`

#### Build RE training data

```bash
python scripts/build_spacy_re_data.py --examples data_clean/benchmarks/semeval2010_task8/examples.jsonl --out data_raw/tmp/re_all.spacy
```

#### Train RE model

```bash
python scripts/train_re.py --train data_raw/tmp/re_train.spacy --dev data_raw/tmp/re_dev.spacy --output models/re
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

- `data_clean/extracted/entities.jsonl` — one line per predicted span: `doc_id`, `text`, `label` (always **ENT** for the SemEval-trained head/tail style), `start_char`, `end_char`.
- `data_clean/extracted/relations.jsonl` — one line per **ordered entity pair** scored by the RE head: `doc_id`, `head` / `tail` (each with `text`, `start_char`, `end_char`), `label` (SemEval relation type), `score`. Row count is **much larger** than “facts in the corpus”: the model assigns a relation distribution to every pair; use **score thresholds** or **top‑k per head** before building a graph.

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

1) Provide SemEval/FewRel raw files, rerun preprocess with flags (see README for concrete paths).

```bash
python scripts/preprocess.py --allow-pdf-failures --semeval-train data_raw/benchmarks/semeval2010/TRAIN_FILE.TXT --semeval-test data_raw/benchmarks/semeval2010/TEST_FILE.TXT
```

1) Convert benchmark JSONL → spaCy DocBin(s), train NER + RE, then run `scripts/run_extraction.py`.

---

## Run log — commands we ran and what we got

This section records **actual outputs** from the workflow we executed (Spring 2026). Paths are relative to the repo root.

### Environment

- Python **3.13.7**
- Dependencies from `requirements.txt` (spaCy 3.8.x, PyMuPDF, scraper libs)

### Preprocess: MSOE only (`--allow-pdf-failures`)

Command:

```text
python scripts/preprocess.py --allow-pdf-failures
```

Representative manifest: `data_clean/manifests/preprocess_1834a871c6c71f9f2783f47e.json` (and similar runs).

**MSOE stats (one full run):**

- `docs_seen`: 5790
- `docs_kept`: 5714
- `docs_deduped`: 33
- `pdf_detected`: 43, `pdf_extracted`: 0, `pdf_failed`: 43, `quarantined`: 43

**Outputs:**

- `data_clean/msoe/documents.jsonl` — one JSON object per cleaned document
- `data_clean/manifests/preprocess_<run_id>.json` — argv, counts, Python version
- `data_clean/quarantine/msoe/*.error.json` — one file per PDF/read failure when `--allow-pdf-failures` is set

### Preprocess: MSOE + SemEval 2010 Task-8

Command:

```text
python scripts/preprocess.py --allow-pdf-failures ^
  --semeval-train data_raw/benchmarks/semeval2010/TRAIN_FILE.TXT ^
  --semeval-test  data_raw/benchmarks/semeval2010/TEST_FILE.TXT
```

Manifest: `data_clean/manifests/preprocess_a0c5c0c3405b4798882de427.json`

**SemEval:**

- `examples`: **10717** (train + test combined in `examples.jsonl`)

**Outputs:**

- `data_clean/benchmarks/semeval2010_task8/examples.jsonl`
- (MSOE outputs same as above)

### Training data (spaCy DocBin)

Commands (conceptually):

- `scripts/build_spacy_ner_data.py` → `data_raw/tmp/ner_all.spacy`
- `scripts/build_spacy_re_data.py` → `data_raw/tmp/re_all.spacy`
- `scripts/split_spacy_docbin.py` → `ner_train.spacy`, `ner_dev.spacy`, `re_train.spacy`, `re_dev.spacy`

(`data_raw/` may be committed with SemEval raw files and DocBins; regenerate when retraining on a fresh machine if omitted.)

### Trained models (`models/` — committed)

**NER** — `python scripts/train_ner.py ... --output models/ner`

- Packaged dirs: `models/ner/model-best`, `models/ner/model-last`
- `models/ner/model-best/meta.json` (dev performance snapshot):
  - `ents_f` ≈ **0.636**
  - `ents_p` ≈ **0.642**
  - `ents_r` ≈ **0.631**
- Entity label: **ENT** (SemEval head/tail spans)

**Relation extraction** — `python scripts/train_re.py ... --output models/re`

- Packaged dirs: `models/re/model-best`, `models/re/model-last`
- `models/re/model-best/meta.json`:
  - **19** relation labels (SemEval Task-8, including `Other`)
  - Reported `relation_extractor` score **0.0** in meta (custom pipe scoring not wired; training still completed and `model-best` / `model-last` were written)

### Extraction on MSOE (optional final step)

Command:

```text
python scripts/run_extraction.py --docs data_clean/msoe/documents.jsonl ^
  --ner-model models/ner/model-best --re-model models/re/model-best ^
  --out data_clean/extracted [--limit N]
```

**Outputs (under `data_clean/`, commit if you want to share them):**

- `data_clean/extracted/entities.jsonl`
- `data_clean/extracted/relations.jsonl`

**Full MSOE corpus run (no `--limit`, Spring 2026):**

Summary line from `run_extraction.py`:

- **5,714** documents, **108,569** entity rows, **656,506** relation rows, **851.7 s** wall time (~14.2 min).

Sanity check on a sample line: entities are generic noun phrases tagged **ENT**; relation lines include low scores (e.g. **0.01–0.17**) for many pairs—expected until you filter or rank for downstream graph/RAG.
