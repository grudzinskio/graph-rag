# Graph-RAG project — runbook and implementation notes

This document is written so someone can **run and test the project** without having to retrain everything.

It also documents what we built: **end-to-end preprocessing**, **spaCy NER + supervised relation extraction (RE)**, **MSOE-domain extraction**, **Neo4j graph loading with vector search**, and a **GraphRAG query demo**.

---

## Quickstart (run the demo first)

### 1) Create `.env`

Copy the template and fill in values:

- File: `.env.example` → `.env`
- Required:
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`
  - `GOOGLE_API_KEY` (for Gemini)

### 2) Install dependencies

From project root:

```bash
python -m pip install -r requirements.txt
python -m pip install neo4j python-dotenv sentence-transformers google-generativeai
```

### 3) Run the GraphRAG query demo

This is the main “try the system” entrypoint:

```bash
python scripts/graph_rag_query.py
```

Notes:

- This script retrieves top documents/entities from Neo4j, formats a context, and asks Gemini to answer using only that context.
- If Neo4j is empty, run the “Build the graph” section below.

---

## One summary (send this to a teammate)

This repo builds and tests a GraphRAG system for MSOE web/curriculum pages, and separately benchmarks the relation-extraction (RE) component on SemEval.

### What the benchmark is actually testing (important)

- **`cv_results.txt` / “k-fold Macro-F1 + Accuracy”** tests a **supervised relation extraction classifier** on the **SemEval-2010 Task-8** benchmark.
- It does **not** test Neo4j GraphRAG retrieval quality on MSOE pages.
- For GraphRAG “system accuracy”, we use retrieval metrics (**Recall@K / MRR**) from `scripts/eval_retrieval.py`.

### What changed in the MSOE → Neo4j graph (relations/entities)

Goal: reduce noise, fix duplication, and make GraphRAG context higher-signal.

- **Canonical entities**: entity node IDs are now canonicalized (lowercase/whitespace/punctuation normalization) so casing/format variants merge cleanly.
- **Better entity selection under cap**: instead of “first N unique strings”, we select top entities by **doc-frequency** (more retrieval-relevant).
- **Cleaner relations**:
  - Drop label `Other`
  - Drop low-confidence relations via `--min-rel-score`
  - Keep only **top-K relations per document** (`--top-rel-per-doc`) to prevent per-doc edge explosion.
- **Better context ordering**: GraphRAG context now explicitly orders by similarity and relation score (`ORDER BY ... DESC`) so returned edges are deterministic and higher quality.

Files:

- Extraction filtering knobs: `scripts/run_extraction.py`
- Neo4j loader + caps/canonicalization: `scripts/upload_to_neo4j.py`
- GraphRAG context ordering + SEARCH/fallback vector query: `scripts/graph_rag_query.py`

### What changed in the RE model (why the SemEval metrics improved)

Goal: stop training collapse, reduce negative imbalance, and improve classification capacity.

- **Corrected the RE training objective** for single-label SemEval by switching to a proper multi-class setup (softmax-style training).
- **Fixed gold targets for unlabeled candidate pairs** so they train as `Other` instead of “all zeros”.
- **Added negative sampling for candidate pairs** (`negatives_per_positive`) to reduce the massive negative imbalance that happens when many entity pairs exist.
- **Upgraded the classifier head** to a small MLP (`rel_classification_layer.v2`) for better Macro-F1.
- **Added mini-test flags** so we can iterate quickly without running full 5-fold + final training every time.

Files:

- RE model code: `extraction_spacy/relation_extractor.py`
- RE config: `configs/re.cfg`
- Faster CV runner: `scripts/train_cv_and_final.py` and `scripts/train_re.py`

### Transformer attempt (status)

- Added optional transformer config: `configs/re_transformer.cfg`
- Installed `spacy-transformers`
- It initializes and starts training, but **CPU-only runs are extremely slow**, so transformer training is only practical with GPU.

### How to run the system without retraining

1) Set up `.env` from `.env.example`
2) Run the demo:

```bash
python scripts/graph_rag_query.py
```

If Neo4j is empty, build the graph:

```bash
python scripts/preprocess.py --allow-pdf-failures
python scripts/run_extraction.py ^
  --docs data_clean/msoe/documents.jsonl ^
  --ner-model models/ner/model-best ^
  --re-model models/re_final/model-best ^
  --out data_clean/extracted ^
  --drop-other ^
  --min-rel-score 0.05 ^
  --top-rel-per-doc 200
python scripts/upload_to_neo4j.py --clear --max-entities 50000 --max-relations 175000
```

### How to test it

- **System test (GraphRAG retrieval quality)**:
  - Fill or generate eval JSONL (`data_clean/eval/retrieval_questions.jsonl`, or `generate_retrieval_eval_questions.py` → `retrieval_questions_many.jsonl`)
  - Run `python scripts/eval_retrieval.py --data data_clean/eval/retrieval_questions.jsonl --k 1 3 5 10` (or `--data data_clean/eval/retrieval_questions_many.jsonl`)
- **Component test (SemEval RE)**:
  - Mini test: `python scripts/train_cv_and_final.py --folds 2 --train-limit 1000 --skip-final --max-steps 500 --eval-frequency 100`
  - Benchmark snapshot used in writeups:
    - Command: `python scripts/train_cv_and_final.py --folds 3 --train-limit 3000 --skip-final --max-steps 2000`
    - Mean Macro-F1 (excl. Other): **0.3634**
    - Mean Accuracy: **0.4291**

---

## System-level testing (GraphRAG retrieval quality)

The SemEval CV metrics do **not** test GraphRAG. For GraphRAG quality, we evaluate **retrieval**.

### Document retrieval evaluation

1) Fill in evaluation questions and expected document ids:

- File: `data_clean/eval/retrieval_questions.jsonl` (small demo set), or `data_clean/eval/retrieval_questions_many.jsonl` (generated bulk set)
- Schema:
  - `query`: question text
  - `expected_doc_ids`: list of acceptable `(:Document {id})` values

If you do not have Neo4j access, use the offline helper to suggest likely doc ids from the local corpus:

```bash
python scripts/suggest_expected_docs.py --data data_clean/eval/retrieval_questions.jsonl --top 5
```

2) Run the retrieval eval against Neo4j (`document_embeddings` index):

```bash
python scripts/eval_retrieval.py --data data_clean/eval/retrieval_questions.jsonl --k 1 3 5 10
python scripts/eval_retrieval.py --data data_clean/eval/retrieval_questions_many.jsonl --k 1 3 5 10
```

Default is **`--mode document`** (single block, same as older runs). Use **`--mode both`** to also print **chunk → document** metrics (best chunk score per `doc_id`, closer to chunk-first GraphRAG). Tuning: **`--chunk-scan 2500`** (how many chunk hits to merge before ranking docs).

Expected outputs:

- `Recall@K`: whether an expected document appears in the top K results
- `MRR`: mean reciprocal rank of the first expected hit

### Retrieval tooling added (debugging and scaling eval)

- **`scripts/probe_retrieval_gold.py`** — For each eval row, prints whether gold `Document` ids exist in Neo4j, **`size(d.text)`** (upload often stores chunk-derived text, not the full JSONL page), **rank** under `document_embeddings`, and rank after aggregating **`chunk_embeddings`** hits by `doc_id`. Use this when scores look wrong because labels assume full-page semantics but the DB embeds shorter `Document.text`.

- **`scripts/generate_retrieval_eval_questions.py`** — Samples `N` documents from `data_clean/msoe/documents.jsonl`, uses the **first line** of each page (usually the HTML title line) as the query, and sets **`expected_doc_ids`** to that document’s id. This is a **self-retrieval** stress test (does the index return the same page when queried with its title line?).

- **`scripts/bootstrap_retrieval_labels.py`** — Rewrites **`expected_doc_ids`** from live Neo4j **document** vector search top hits (`--take 1` is typical). Metrics from `eval_retrieval.py` become **self-consistent with the index** (often **1.0**); use only when you explicitly want that behavior, not as an independent relevance benchmark.

See **Retrieval evaluation snapshots** at the end of this document for latest logged scores.

---

## Component testing (Relation Extraction benchmark metrics)

We also keep a benchmark “component health” test: SemEval 2010 Task-8 RE classification.

### Mini tests (fast sanity checks)

This avoids waiting for full 5-fold CV while iterating:

```bash
python scripts/train_cv_and_final.py --folds 2 --train-limit 1000 --skip-final --max-steps 500 --eval-frequency 100
```

### Full(er) benchmark runs

```bash
python scripts/train_cv_and_final.py --folds 5 --skip-final --max-steps 8000
python scripts/train_cv_and_final.py
python scripts/evaluate_prototype.py
```

Placeholders (fill in after final run):

- SemEval 5-fold Macro-F1 (excl. Other): **[TODO]**
- SemEval 5-fold Accuracy: **[TODO]**
- SemEval held-out test Macro-F1 (excl. Other): **[TODO]**
- SemEval held-out test Accuracy: **[TODO]**

Latest benchmark snapshot (from our development runs):

- **What it is testing**: supervised **relation extraction classification** on the **SemEval-2010 Task-8** benchmark. This reports **macro-F1 excluding `Other`** and accuracy on held-out folds. It is a **component test** for the RE model (not Neo4j/MSOE GraphRAG retrieval quality).
- **Command**: `python scripts/train_cv_and_final.py --folds 3 --train-limit 3000 --skip-final --max-steps 2000`
- **Result (3-fold CV)**:
  - Mean Macro-F1 (excl. Other): **0.3634**
  - Mean Accuracy: **0.4291**

---

## Build the graph (only required if Neo4j is empty)

### 1) Preprocess MSOE documents

```bash
python scripts/preprocess.py --allow-pdf-failures
```

Output:

- `data_clean/msoe/documents.jsonl`

### 2) Train models (optional if you already have them)

If you do not have a trained RE model, run:

```bash
python scripts/train_cv_and_final.py
```

This produces:

- `models/re_final/model-best`
- `test_data.jsonl`

### 3) Extract entities + relations over MSOE corpus

```bash
python scripts/run_extraction.py ^
  --docs data_clean/msoe/documents.jsonl ^
  --ner-model models/ner/model-best ^
  --re-model models/re_final/model-best ^
  --out data_clean/extracted ^
  --drop-other ^
  --min-rel-score 0.05 ^
  --top-rel-per-doc 200
```

Outputs:

- `data_clean/extracted/entities.jsonl`
- `data_clean/extracted/relations.jsonl`

### 4) Upload to Neo4j

```bash
python scripts/upload_to_neo4j.py --clear --max-entities 50000 --max-relations 175000
```

Key loader behavior (quality/performance):

- Entities are canonicalized and selected by **doc-frequency** (not “first 50k seen”)
- Relations are filtered (`Other` dropped, minimum score) and capped **top-K per doc**

---

## 1. Data we use (sources and scale)

### MSOE scraped web text (our domain corpus)

- **Sources:** `scraped_data/sitemap/text/*.txt` and `scraped_data/catalog/text/*.txt` (from our scraper: HTML → text).
- **After cleaning** (one full run with `--allow-pdf-failures`):
  - **5,790** documents seen → **5,714** kept (**33** exact duplicates removed by content hash).
  - **43** files detected as embedded **PDF bytes** (filename `*.txt` but content starts with `%PDF-`); extraction attempted with PyMuPDF; in that run **0** extracted successfully, **43** failed and were **quarantined** (see `data_clean/quarantine/msoe/*.error.json` when using `--allow-pdf-failures`).
- **Output file:** `data_clean/msoe/documents.jsonl` — one JSON object per document with stable `id`, `dataset` (`msoe_sitemap` / `msoe_catalog`), cleaned `text`, `source_path`, `source_sha256`, and `meta` (line counts, hashes, read mode).

### SemEval 2010 Task-8 (supervised training for NER + RE)

- **Raw inputs:** official train/test files with `<e1>...</e1>` and `<e2>...</e2>` (passed via `scripts/preprocess.py` flags).
- **Normalized output:** `data_clean/benchmarks/semeval2010_task8/examples.jsonl`.
- **Scale:** **10,717** examples total when both train and test are included; the JSONL stores **`split`: `train` | `test`** for the official **8,000 train / 2,717 test** split used in evaluation scripts.

### FewRel (optional)

- Supported by the same preprocessor when a FewRel JSON file is provided; writes `data_clean/benchmarks/fewrel/examples.jsonl`. Our main supervised pipeline uses **SemEval**.

---

## 1b. Example JSON (real shapes from this repo)

Below are **actual field names and structure** taken from generated files in the project (some `text` fields are shortened only where noted).

### Preprocess run manifest (`data_clean/manifests/preprocess_<run_id>.json`)

Records argv, outputs, aggregate stats, and Python version:

```json
{
  "run_id": "1834a871c6c71f9f2783f47e",
  "created_utc": "2026-04-09T19:28:57.721794+00:00",
  "argv": [
    "scripts/preprocess.py",
    "--allow-pdf-failures"
  ],
  "outputs": {
    "msoe_documents": "data_clean\\msoe\\documents.jsonl"
  },
  "stats": {
    "msoe": {
      "docs_seen": 5790,
      "docs_kept": 5714,
      "docs_deduped": 33,
      "docs_empty_after_clean": 0,
      "pdf_detected": 0,
      "pdf_extracted": 0,
      "pdf_failed": 43,
      "quarantined": 43
    }
  },
  "versions": {
    "python": "3.13.7 (tags/v3.13.7:bcee1c3, Aug 14 2025, 14:15:11) [MSC v.1944 64 bit (AMD64)]"
  }
}
```

### Quarantine entry (PDF read failure)

When a catalog `*.txt` is really PDF bytes and extraction fails (`--allow-pdf-failures`), one JSON error file per path under `data_clean/quarantine/msoe/`:

```json
{
  "path": "scraped_data\\catalog\\text\\mime_media_42_2368_Accounting.pdf.txt",
  "error": "Failed to open stream"
}
```

### Cleaned MSOE document (`data_clean/msoe/documents.jsonl`)

One JSON object **per line**. Example document (full `text` as in repo; line breaks shown as `\n` in the string):

```json
{
  "id": "0e6046d21e2c45f74321c734",
  "dataset": "msoe_sitemap",
  "split": null,
  "text": "Faculty Positions | MSOE\nHome\nAbout MSOE\nCareers at MSOE\nFaculty Positions\nCurrent Faculty Positions\nClick on the job title of the position you are interested in for more information about that position. Please note that a new window will open with the position details, so you may need to have pop-ups enabled on your internet browser.\nMSOE is an Equal Opportunity/Affirmative Action Employer\n1025 North Broadway\nMilwaukee,\nWI\n53202-3109\n(800) 332-6763\nexplore@msoe.edu\nFacebook\nInstagram\nTwitter\nLinkedin\nYouTube\nVimeo",
  "source_path": "scraped_data\\sitemap\\text\\about-msoe_careers-at-msoe_faculty-positions.txt",
  "source_sha256": "4d8e1713ee606cb13a0c954262bad194e2a18f481679fc4fe9da4802ea9461ed",
  "meta": {
    "read": {
      "sha256": "4d8e1713ee606cb13a0c954262bad194e2a18f481679fc4fe9da4802ea9461ed",
      "size_bytes": 637,
      "detected_format": "text",
      "read_mode": "utf8_replace"
    },
    "clean": {
      "lines_in": 27,
      "lines_out": 20,
      "dropped_by_pattern": 7,
      "dropped_in_section": 0
    },
    "content_sha256": "d270af472354f4c496a809c56825493a82f5d3b0f1339ae35d9d20be5abf4eb1"
  }
}
```

### SemEval benchmark row (`data_clean/benchmarks/semeval2010_task8/examples.jsonl`)

Used for NER/RE training; `e1` / `e2` are character spans into `text`; `relation` is the gold SemEval label (direction matters, e.g. `(e1,e2)` vs `(e2,e1)`):

```json
{
  "id": "e8afff741a6f2609235e4727",
  "dataset": "semeval2010_task8",
  "split": "train",
  "text": "The system as described above has its greatest application in an arrayed configuration of antenna elements.",
  "e1": { "text": "configuration", "char_start": 73, "char_end": 86 },
  "e2": { "text": "elements", "char_start": 98, "char_end": 106 },
  "relation": "Component-Whole(e2,e1)",
  "source_path": "data_raw\\benchmarks\\semeval2010\\TRAIN_FILE.TXT"
}
```

### Held-out test example (`test_data.jsonl`)

Same schema as SemEval examples, but only official **test** split (for `evaluate_prototype.py`):

```json
{
  "id": "c23944b3aeea2660b8dab11a",
  "dataset": "semeval2010_task8",
  "split": "test",
  "text": "The most common audits were about waste and recycling.",
  "e1": { "text": "audits", "char_start": 16, "char_end": 22 },
  "e2": { "text": "waste", "char_start": 34, "char_end": 39 },
  "relation": "Message-Topic(e1,e2)",
  "source_path": "data_raw\\benchmarks\\semeval2010\\TEST_FILE.TXT"
}
```

### Predicted entities (`data_clean/extracted/entities.jsonl`)

One span per line from NER over MSOE cleaned text (`label` is **`ENT`** from SemEval-style training):

```json
{
  "doc_id": "4b29634ca86207684b6a178d",
  "text": "university",
  "label": "ENT",
  "start_char": 102,
  "end_char": 112
}
```

### Predicted relations (`data_clean/extracted/relations.jsonl`)

One **ordered pair** per line: argmax relation label and its score from the RE head (many pairs per document are normal):

```json
{
  "doc_id": "4b29634ca86207684b6a178d",
  "head": {
    "text": "university",
    "start_char": 102,
    "end_char": 112
  },
  "tail": {
    "text": "community",
    "start_char": 243,
    "end_char": 252
  },
  "label": "Instrument-Agency(e2,e1)",
  "score": 0.015000326558947563
}
```

### Neo4j upload batches (`scripts/upload_to_neo4j.ipynb`)

The notebook does **not** store another JSON file on disk; it **unwinds** batches into Cypher. Each row matches these shapes:

**Entity batch item** (after embedding with `sentence-transformers` `all-MiniLM-L6-v2`, **384** dimensions). The `embedding` key is a JSON array of **384** floats; only the first few values are shown:

```json
{
  "id": "university",
  "label": "ENT",
  "embedding": [0.01234, -0.04567, 0.00891]
}
```

*(In real uploads the `embedding` array has length **384**, not 3.)*

`id` is the entity surface string used as the merge key; the Neo4j property `e.text` is set to that same string.

**Relation batch item** (edges only created if both endpoint strings exist as `Entity.id`):

```json
{
  "head_text": "university",
  "tail_text": "community",
  "label": "Instrument-Agency(e2,e1)",
  "score": 0.015000326558947563
}
```

Cypher merges `(src:Entity)-[:REL {label}]->(tgt:Entity)` and sets `r.score`.

---

## 2. Preprocessing — what we actually do (MSOE)

Implemented in `scripts/preprocess.py`. The goal is a **deterministic**, **deduplicated** plain-text corpus suitable for NER/RE and downstream graph building.

### Normalization (`normalize_text`)

- Unicode **NFKC** normalization.
- Strip null bytes; normalize newlines; remove control characters except tab/newline.
- Per line: collapse repeated whitespace, strip; drop empty lines; join with newlines.

### Catalog-specific cleanup (`clean_msoe_text`)

- **Section drop:** from a line matching **“Select a Catalog”** until an end marker (**“HELP”** or **“Course Descriptions”**) — removes long catalog navigation blocks.
- **Single-line drops** (regex matches on whole lines), including among others:
  - “Global Search”, “Catalog Search”, “Catalog Navigation”, “Back to Top”, “Skip to Main Content”, “Menu”, “Search”, “submit”, “Resources For…”, “Connect With MSOE”, “MSOE University”, “Print-Friendly Page…”, “Powered by Modern Campus Catalog…”, copyright line for MSOE.

### PDF-in-text files

- If a `*.txt` file begins with **`%PDF-`**, we treat it as **PDF bytes**, not UTF-8 text.
- Text is extracted with **PyMuPDF** (`fitz`), page by page, concatenated with newlines.
- With **`--allow-pdf-failures`**, failures are quarantined instead of aborting the run.

### Document identity and deduplication

- Each kept document gets a **stable hash-based `id`** and **`source_sha256`** of the raw file.
- **Exact deduplication:** after cleaning, documents with the same **SHA-256 of cleaned text** are skipped (counted as `docs_deduped`).

### Manifests (reproducibility)

- Every run writes `data_clean/manifests/preprocess_<run_id>.json`: CLI argv, output paths, counts, Python version.

---

## 3. Entity extraction (NER)

### Role

NER finds **entity spans** in text. For MSOE runs we use a model trained on **SemEval** marked entities, exported as a single spaCy label **`ENT`** (head/tail style), not fine-grained types like `PERSON`/`ORG`.

### Training data construction (SemEval NER)

- `scripts/build_spacy_ner_data.py` reads `examples.jsonl`, builds `doc.char_span(...)` for **e1** and **e2**, uses **`filter_spans`** to resolve overlaps, writes a **`DocBin`** (`.spacy`).
- Train/dev split: `scripts/split_spacy_docbin.py` (fixed seed).

### Training

- `scripts/train_ner.py` wraps `python -m spacy train` with `configs/ner.cfg`.
- **Reported dev scores** (committed `models/ner/model-best`): **ents_f ≈ 0.636**, **ents_p ≈ 0.642**, **ents_r ≈ 0.631** (`meta.json`).

### Inference on MSOE (NER + RE extraction script)

- Load **NER** and **RE** pipelines separately.
- Run **NER on full document text** → `doc.ents`.
- Copy those spans onto a fresh doc and run the **RE** pipeline so relations are scored for **predicted** entity pairs.

**Entity output:** `data_clean/extracted/entities.jsonl` — per span: `doc_id`, `text`, `label` (e.g. **`ENT`**), `start_char`, `end_char`.

**Scale (full MSOE run):** **108,569** entity rows across **5,714** documents (~**14.2 min** on our run).

---

## 4. Relation extraction (RE)

### Model

- Custom **trainable spaCy component** `relation_extractor` in `extraction_spacy/relation_extractor.py`.
- **Convention:** `doc._.rel[(ent1.start, ent2.start)] = {relation_label: score, ...}` for ordered entity pairs.
- Architecture: **Thinc** model (see file) producing scores over **19** SemEval labels (18 directional types + **`Other`**).

### Training data construction (SemEval RE)

- `scripts/build_spacy_re_data.py` builds `DocBin` with **gold** `doc._.rel` from each example’s `relation` field; supports a **global label list** via `--labels` (semicolon-separated) so every fold uses the same label inventory.

### Training + evaluation scripts (what to run)

- `scripts/train_re.py` + `configs/re.cfg` → `models/re/model-best` (as in early README flow). Custom pipe dev score in `meta.json` may show **0.0** if not wired for spaCy’s default scorer; training still produces usable weights.

### Evaluation pipeline

1. **`scripts/train_cv_and_final.py`**
   - Loads `data_clean/benchmarks/semeval2010_task8/examples.jsonl`.
   - **5-fold CV** on the **official train split** (shuffle seed **42**); each fold trains a full RE model and evaluates on the held-out fold with **macro-F1 excluding `Other`** (SemEval-style).
   - Writes **`cv_results.txt`** (per-fold reports + summary).
   - Trains a **final** model on **90%** of train (10% for spaCy early-stopping only — **not** the CV dev sets) → **`models/re_final/model-best`**.
   - Saves **`test_data.jsonl`** = official **test** split (**2,717** examples) for blind evaluation.

2. **`scripts/evaluate_prototype.py`**
   - Loads `models/re_final/model-best` and `test_data.jsonl`.
   - Injects gold spans as `doc.ents`, runs the pipeline, argmax over `doc._.rel` for the **(e1,e2)** key.
   - Prints classification report, **macro-F1 (excl. Other)**, and accuracy.

We report:

- **Macro-F1 excluding `Other`** (SemEval official metric)
- Accuracy

See the placeholders at the top of this document for the latest numbers.

### Inference on MSOE (RE output format)

- For each entity pair in the RE head’s instance set, we take the **argmax label** and **score** (see loop in script).
- **Relation output:** `data_clean/extracted/relations.jsonl` — `doc_id`, `head` / `tail` (text + offsets), `label`, **`score`**.

**Important:** row count is **one line per scored pair** from the model, not “validated facts only.” For a graph you typically **filter by score**, **cap degree**, or **top-k per entity**. Full run: **656,506** relation rows for **5,714** docs.

---

## 5. Neo4j graph store and vector index

### Upload path and schema

- Notebook **`scripts/upload_to_neo4j.ipynb`** (batch load from extracted JSONL).
- Dependencies used there: **`neo4j`** Python driver, **`sentence-transformers`** (embeddings).

### Schema (conceptual)

- **Nodes:** `:Entity` with unique **`id`**, **`text`** (stored same as id in notebook), **`label`** (entity type from extraction), **`embedding`** (dense vector).
- **Relationships:** `[:REL {label, score}]` from head entity to tail entity; **`score`** stores the RE confidence.
- **Constraint:** `entity_id_unique` on `Entity.id`.
- **Vector index:** `entity_embeddings` on **`Entity.embedding`** for similarity search (Neo4j 5+ vector index syntax in the notebook).

Entity keys in the loader use **entity text** as the graph `id` for merging (see notebook: `MERGE (e:Entity {id: item.id})` with edges matched by `head_text` / `tail_text`). Be aware this collapses identical strings across documents.

### Query / GraphRAG demo

- **`scripts/graph_rag_query.py`** — interactive loop:
  - Embeds the user question with `sentence-transformers` (`all-MiniLM-L6-v2`)
  - Uses Neo4j vector search (prefers `SEARCH`, falls back if needed)
  - Retrieves per-document entity keywords and top-scored relations
  - Sends context to Gemini with “answer only from context” instructions
- **Configuration:** `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GOOGLE_API_KEY` via **`.env`** (defaults in script are placeholders for Neo4j Aura-style URI — **override with your instance**).

---

## 6. Repository layout (main artifacts)

| Area | Paths |
| --- | --- |
| Preprocess | `scripts/preprocess.py` |
| NER/RE data builders | `scripts/build_spacy_ner_data.py`, `scripts/build_spacy_re_data.py`, `scripts/split_spacy_docbin.py` |
| Train | `scripts/train_ner.py`, `scripts/train_re.py`, `configs/ner.cfg`, `configs/re.cfg` |
| CV + final RE | `scripts/train_cv_and_final.py`, `cv_results.txt`, `models/re_final/model-best` |
| Test eval | `scripts/evaluate_prototype.py`, `test_data.jsonl` |
| Custom RE code | `extraction_spacy/relation_extractor.py` |
| Corpus extraction | `scripts/run_extraction.py` → `data_clean/extracted/*.jsonl` |
| Neo4j load | `scripts/upload_to_neo4j.ipynb`, `scripts/upload_to_neo4j.py` |
| GraphRAG CLI | `scripts/graph_rag_query.py` |
| Retrieval eval | `scripts/eval_retrieval.py`, `scripts/suggest_expected_docs.py`, `scripts/probe_retrieval_gold.py`, `scripts/generate_retrieval_eval_questions.py`, `scripts/bootstrap_retrieval_labels.py`, `data_clean/eval/retrieval_questions.jsonl`, `data_clean/eval/retrieval_questions_many.jsonl` |

---

## 7. Reproducible command flow (high level)

See “Quickstart” and “Build the graph” sections at the top of this document.

---

## 8. Talking points for the presentation

- **Preprocessing** is deliberately boring but **correct**: normalization, boilerplate stripping, catalog menu removal, PDF handling, **hash-based IDs**, and **content deduplication** — so downstream numbers are trustworthy.
- **Entities** come from a **supervised NER** model trained on **SemEval spans** (`ENT`); **relations** from a **custom spaCy RE head** with **19** labels; we report **5-fold CV** and **held-out test** metrics via `train_cv_and_final.py` + `evaluate_prototype.py`.
- **MSOE extraction** produces **large** `relations.jsonl` because every scored pair is logged; the **graph** should use **thresholds** or **top-k** to avoid noise.
- **Neo4j** stores **entities + typed edges + scores** and a **vector index** for **semantic retrieval**; **GraphRAG** combines **vector search over entities**, **local graph expansion**, and an **LLM** for grounded answers.

---

## Improvements since prototype draft

Everything below is what we changed/improved after the early prototype baseline.

### Graph + relations (MSOE → Neo4j) improvements

- **Entity canonicalization** (`scripts/upload_to_neo4j.py`): merge entity variants by normalizing the entity ID (case/whitespace/punctuation).
- **Better entity selection under cap** (`scripts/upload_to_neo4j.py`): replaced “first N unique entities encountered” with **doc-frequency + mention-frequency ranking** so the 50k cap keeps higher-signal entities.
- **Stronger page cleanup before graph load** (`scripts/preprocess.py`):
  - Added extra boilerplate/header filtering (for example repeated `Home`/breadcrumb noise)
  - Added heuristic footer-block trimming (address/social/contact clusters near page tail) to reduce repeated low-signal text
- **Relation filtering and caps**:
  - Extraction-time filtering (`scripts/run_extraction.py`): added `--drop-other`, `--min-rel-score`, and `--top-rel-per-doc` to reduce noisy edges.
  - Upload-time filtering (`scripts/upload_to_neo4j.py`): mirrors the same logic defensively and uploads only top-scored relations per document.
- **Chunk-first graph upload under storage constraints** (`scripts/upload_to_neo4j.py`):
  - Added `:Chunk` nodes and `(:Document)-[:HAS_CHUNK]->(:Chunk)` edges
  - Added a `chunk_embeddings` vector index for retrieval
  - Added chunk controls: `--chunk-size`, `--chunk-overlap`, `--max-chunks-per-doc`, `--max-chunks`
  - Added exact + near-duplicate chunk suppression (canonical hash + SimHash/Hamming threshold) so upload keeps diverse context instead of repeated text
  - Added `--document-text-from-chunks` so `Document.text` can be constrained to selected chunks only
- **More deterministic / higher-signal query context** (`scripts/graph_rag_query.py`):
  - Explicit ordering for doc retrieval and relation listing (`ORDER BY score DESC`)
  - Keywords prefer `Entity.display` when available
- **Neo4j vector search deprecation**: updated queries to prefer `SEARCH ... VECTOR INDEX ... SCORE AS score` with a fallback to the deprecated `db.index.vector.queryNodes`.

### GraphRAG query/runtime improvements

- **Chunk-first retrieval path** (`scripts/graph_rag_query.py`):
  - Query flow now prefers `chunk_embeddings` retrieval and only falls back to document/entity-only paths if needed.
- **Budgeted two-pass retrieval** (`scripts/graph_rag_query.py`):
  - Compact pass (small caps) tries to answer cheaply first.
  - Optional expanded pass runs only when the controller predicts additional evidence is needed.
- **Agentic multi-hop traversal loop** (`scripts/graph_rag_query.py`):
  - Added planner-guided hop decisions (`continue` + focus terms) and bounded traversal depth (`--max-hops`).
  - Each hop expands entity neighborhoods and appends additional graph evidence under per-hop char caps.
- **Runtime tuning flags** (`scripts/graph_rag_query.py`):
  - Added CLI flags for compact/expanded retrieval, hop settings, and overall LLM char budget.
- **Per-query telemetry** (`scripts/graph_rag_query.py`):
  - After each answer, prints token usage (when provided by Gemini) and retrieval/traversal stats:
    - chunks retrieved/used, docs used, relations used
    - hops executed, focus terms used, entity seeds, relations traversed

### System testing (GraphRAG) improvements

- **Added retrieval-focused evaluation** (`scripts/eval_retrieval.py`):
  - Measures **Recall@K** and **MRR** for MSOE queries over the Neo4j `document_embeddings` vector index.
- **No-Neo4j-access support** (`scripts/suggest_expected_docs.py`):
  - Offline helper that suggests `Document.id` values using local `data_clean/msoe/documents.jsonl` so you can fill `expected_doc_ids` without querying Neo4j.
- **Retrieval probes and bulk eval helpers** (same Neo4j env as upload):
  - `scripts/probe_retrieval_gold.py` — document vs chunk index ranks and `Document.text` lengths for gold ids.
  - `scripts/generate_retrieval_eval_questions.py` — builds `data_clean/eval/retrieval_questions_many.jsonl` for larger self-retrieval runs.
  - `scripts/bootstrap_retrieval_labels.py` — optional: set gold ids from retriever top hits so eval metrics align with the index.

### RE benchmark (SemEval) improvements

- **Fixed training objective / collapse issues** (`extraction_spacy/relation_extractor.py`):
  - Proper multi-class training behavior for SemEval (single-label)
  - Treat unlabeled candidate pairs as `Other` targets instead of invalid all-zero targets
- **Negative sampling for candidate pairs** (`extraction_spacy/relation_extractor.py`, `configs/re.cfg`):
  - Added `negatives_per_positive` to reduce the extreme negative imbalance from O(n²) pair generation.
- **Stronger classification head** (`extraction_spacy/relation_extractor.py`, `configs/re.cfg`):
  - Added `rel_classification_layer.v2` (MLP + dropout) and switched config to use it.
- **Faster iteration tools** (`scripts/train_cv_and_final.py`, `scripts/train_re.py`):
  - Added mini-test flags: `--folds`, `--train-limit`, `--skip-final`, `--max-steps`, `--eval-frequency`
  - Fixed label-set stability so mini runs don’t crash due to missing classes.
- **Latest dev benchmark snapshot** (SemEval RE component test):
  - Macro-F1 (excl. Other): **0.3634**
  - Accuracy: **0.4291**
  - Command: `python scripts/train_cv_and_final.py --folds 3 --train-limit 3000 --skip-final --max-steps 2000`

### Transformer attempt (optional, CPU-limited)

- Added `configs/re_transformer.cfg` and installed `spacy-transformers`.
- The transformer pipeline initializes and starts training, but CPU-only training is very slow; this path is mainly useful if GPU is available.

---

## Retrieval evaluation snapshots

Scores below use **`scripts/eval_retrieval.py`** with **`--mode document`** (**document** index only; comparable to older logs). Use **`--mode both`** for an added **chunk→doc** block. Same embedding model as upload: **sentence-transformers `all-MiniLM-L6-v2`**. Re-run after **`upload_to_neo4j.py`** if you change chunking; **`--max-chunks-per-doc`** default is now **6** (was 3) for better long-page coverage unless you override.

### 100-query self-retrieval run (stress test)

- **Dataset**: `data_clean/eval/retrieval_questions_many.jsonl` from `python scripts/generate_retrieval_eval_questions.py --n 100 --seed 7` (first line of each sampled page as query; gold id = source document).
- **Command**: `python scripts/eval_retrieval.py --data data_clean/eval/retrieval_questions_many.jsonl --k 1 3 5 10 --mode both` (prints **document** index metrics and **chunk→doc** aggregation; use `--mode document` for the legacy single block only).
- **Queries evaluated**: 100  
- **Chunk scan** (chunk→doc path): `--chunk-scan 2500` (default).

| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| **Document vector index** (`document_embeddings` on `(:Document)`) | 0.1900 | 0.3500 | 0.4800 | **0.6000** | 0.3103 |
| **Chunk→doc** (best chunk score per `doc_id`, then rank docs) | 0.2000 | 0.4200 | 0.4500 | 0.4900 | 0.3077 |

On this run, **Recall@10 is higher for the document index than for chunk→doc**. That is plausible: the eval query is the page **title line**, which may align well with a single **document** embedding, while chunk retrieval scores many competing spans first; chunk→doc aggregation is also only an approximation of the full GraphRAG query path (which may fuse chunks with entities, relations, and second-pass expansion).

#### Why these metrics may look modest

- **Self-retrieval is strict**: the query is derived from the **same** page’s first line; small mismatch between what was embedded in Neo4j (e.g. truncated `Document.text`, chunk-only text, or boilerplate-heavy spans) and the title line hurts similarity.
- **Many documents**: random guessing among thousands of pages would be near zero at top‑1; ~19–20% at **Recall@1** means the index is doing non-trivial ranking, not “random.”
- **Catalog/course titles** (“Program: …”, “AE 6222 …”) can match **many** similar catalog shells; embeddings confuse nearby programs.

#### What could improve measured or real retrieval

- **Re-upload after tuning upload**: more chunks per doc (`--max-chunks-per-doc`, now default **6**), full `Document.text` from `documents.jsonl` unless you intentionally use `--document-text-from-chunks`, then rebuild indexes.
- **Stronger or domain-tuned embedding models** (re-embed all nodes; heavier cost).
- **Hybrid search** (BM25/keyword + vector) for exact course codes and titles.
- **Reranking** top-K with a cross-encoder or small reranker on `(query, doc)` pairs.
- **Eval alignment**: use **`--mode both`** to track document vs chunk-style behavior; improving live **`graph_rag_query.py`** may require changes beyond this script’s chunk→doc aggregation.

### Small demo file after label bootstrap (not an independent relevance benchmark)

- **Dataset**: `data_clean/eval/retrieval_questions.jsonl` after `python scripts/bootstrap_retrieval_labels.py --take 1` (each **`expected_doc_ids`** entry set to the retriever’s **top-1** document id for that query).
- **Command**: `python scripts/eval_retrieval.py --data data_clean/eval/retrieval_questions.jsonl --k 1 3 5 10`
- **Queries evaluated**: 2  
- **Recall@1, @3, @5, @10** and **MRR**: **1.0000** (expected when gold labels are defined from the same index’s top hit).
