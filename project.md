# Graph-RAG project — implementation notes (presentation)

This document summarizes what we built: **end-to-end data preprocessing**, **spaCy NER + supervised relation extraction** (SemEval 2010 Task-8), **quantitative evaluation** (5-fold CV + held-out test), **extraction over the cleaned MSOE web corpus**, **Neo4j graph loading with vector search**, and a **GraphRAG query demo** (retrieve subgraph context → answer with an LLM).

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

### Training data construction

- `scripts/build_spacy_ner_data.py` reads `examples.jsonl`, builds `doc.char_span(...)` for **e1** and **e2**, uses **`filter_spans`** to resolve overlaps, writes a **`DocBin`** (`.spacy`).
- Train/dev split: `scripts/split_spacy_docbin.py` (fixed seed).

### Training

- `scripts/train_ner.py` wraps `python -m spacy train` with `configs/ner.cfg`.
- **Reported dev scores** (committed `models/ner/model-best`): **ents_f ≈ 0.636**, **ents_p ≈ 0.642**, **ents_r ≈ 0.631** (`meta.json`).

### Inference on MSOE (`scripts/run_extraction.py`)

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

### Training data construction

- `scripts/build_spacy_re_data.py` builds `DocBin` with **gold** `doc._.rel` from each example’s `relation` field; supports a **global label list** via `--labels` (semicolon-separated) so every fold uses the same label inventory.

### Baseline training (simple train/dev split)

- `scripts/train_re.py` + `configs/re.cfg` → `models/re/model-best` (as in early README flow). Custom pipe dev score in `meta.json` may show **0.0** if not wired for spaCy’s default scorer; training still produces usable weights.

### Rigorous evaluation pipeline (what we added)

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

**CV summary (from committed `cv_results.txt`):**

- Per-fold macro-F1 (excl. Other): **0.0757, 0.0764, 0.0732, 0.0666, 0.0833**
- **Mean macro-F1: 0.0750** (std **0.0054**)
- Mean accuracy: **0.1671**

These numbers reflect a **challenging multi-class** setting; SemEval leaderboards are typically **0.7–0.9** macro-F1. Our model is a **prototype baseline**; domain shift (SemEval news → MSOE catalog pages) is expected to hurt further at inference time.

### Inference on MSOE (`scripts/run_extraction.py`)

- For each entity pair in the RE head’s instance set, we take the **argmax label** and **score** (see loop in script).
- **Relation output:** `data_clean/extracted/relations.jsonl` — `doc_id`, `head` / `tail` (text + offsets), `label`, **`score`**.

**Important:** row count is **one line per scored pair** from the model, not “validated facts only.” For a graph you typically **filter by score**, **cap degree**, or **top-k per entity**. Full run: **656,506** relation rows for **5,714** docs.

---

## 5. Neo4j graph store and vector index

### Upload path

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
  - Embeds the user question with **`sentence-transformers`** model **`all-MiniLM-L6-v2`** (same family as upload).
  - **`CALL db.index.vector.queryNodes('entity_embeddings', ...)`** to retrieve top similar entities, then expands **outgoing** relationships to neighbors.
  - Builds a **text context** of triples and sends it to **Google Gemini** (`gemini-3-flash-preview`) with **“answer only from context”** instructions.
- **Configuration:** `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `GOOGLE_API_KEY` via **`.env`** (defaults in script are placeholders for Neo4j Aura-style URI — **override with your instance**).

---

## 6. Repository layout (main artifacts)

| Area | Paths |
|------|--------|
| Preprocess | `scripts/preprocess.py` |
| NER/RE data builders | `scripts/build_spacy_ner_data.py`, `scripts/build_spacy_re_data.py`, `scripts/split_spacy_docbin.py` |
| Train | `scripts/train_ner.py`, `scripts/train_re.py`, `configs/ner.cfg`, `configs/re.cfg` |
| CV + final RE | `scripts/train_cv_and_final.py`, `cv_results.txt`, `models/re_final/model-best` |
| Test eval | `scripts/evaluate_prototype.py`, `test_data.jsonl` |
| Custom RE code | `extraction_spacy/relation_extractor.py` |
| Corpus extraction | `scripts/run_extraction.py` → `data_clean/extracted/*.jsonl` |
| Neo4j load | `scripts/upload_to_neo4j.ipynb` |
| GraphRAG CLI | `scripts/graph_rag_query.py` |

---

## 7. Reproducible command flow (high level)

1. **Environment:** `python -m pip install -r requirements.txt` (core: BeautifulSoup stack, requests, **PyMuPDF**, **spaCy 3.8.x**). Neo4j upload / GraphRAG additionally need **`neo4j`**, **`sentence-transformers`**, **`python-dotenv`**, and **`google-generativeai`** as used in those scripts.

2. **Clean MSOE text:**  
   `python scripts/preprocess.py --allow-pdf-failures`

3. **SemEval JSONL (for training/eval):**  
   `python scripts/preprocess.py --allow-pdf-failures --semeval-train ... --semeval-test ...`

4. **Train NER + RE** (DocBin → `train_ner` / `train_re`) or run **`train_cv_and_final.py`** for CV + `re_final`.

5. **Extract from MSOE:**  
   `python scripts/run_extraction.py --docs data_clean/msoe/documents.jsonl --ner-model models/ner/model-best --re-model models/re/model-best --out data_clean/extracted`

6. **Evaluate RE on SemEval test:**  
   `python scripts/evaluate_prototype.py` (after `train_cv_and_final.py`).

7. **Load graph + query:** run **`upload_to_neo4j.ipynb`**, then **`python scripts/graph_rag_query.py`** with credentials set.

---

## 8. Talking points for the presentation

- **Preprocessing** is deliberately boring but **correct**: normalization, boilerplate stripping, catalog menu removal, PDF handling, **hash-based IDs**, and **content deduplication** — so downstream numbers are trustworthy.
- **Entities** come from a **supervised NER** model trained on **SemEval spans** (`ENT`); **relations** from a **custom spaCy RE head** with **19** labels; we report **5-fold CV** and **held-out test** metrics via `train_cv_and_final.py` + `evaluate_prototype.py`.
- **MSOE extraction** produces **large** `relations.jsonl` because every scored pair is logged; the **graph** should use **thresholds** or **top-k** to avoid noise.
- **Neo4j** stores **entities + typed edges + scores** and a **vector index** for **semantic retrieval**; **GraphRAG** combines **vector search over entities**, **local graph expansion**, and an **LLM** for grounded answers.
