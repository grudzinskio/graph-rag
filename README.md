# graph-rag

This repo currently contains:

- **MSOE web scraping** (`scrape.py`) that writes raw HTML + extracted text under `scraped_data/`
- **Reproducible preprocessing** (`scripts/preprocess.py`) that builds a deterministic cleaned corpus under `data_clean/`
- **spaCy extraction** scaffolding: NER + supervised Relation Extraction (RE) training/inference scripts

## Setup

### Windows: create a `.venv` (recommended)

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CMD:

```bat
py -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Install dependencies (without a venv)

```bash
python -m pip install -r requirements.txt
```

## What you can run right now (MSOE cleaning only)

If you already have `scraped_data/` in this repo, you can run:

```powershell
python scripts/preprocess.py --allow-pdf-failures
```

This produces:

- `data_clean/msoe/documents.jsonl`
- `data_clean/manifests/preprocess_<run_id>.json`

At this point you have **cleaned text**, but you do **not** have trained spaCy models yet (those require SemEval/FewRel).

## 1) (Optional) Scrape MSOE data (only if `scraped_data/` is missing)

```bash
python scrape.py
```

## 2) Preprocess into cleaned data (reproducible)

```bash
python scripts/preprocess.py --allow-pdf-failures
```

Outputs:

- `data_clean/msoe/documents.jsonl`
- `data_clean/manifests/preprocess_<run_id>.json`

## 3) Add benchmark datasets (SemEval / FewRel) and generate `examples.jsonl`

To train NER + Relation Extraction, we need benchmark-style **examples** with:

- `text` (sentence)
- `e1` and `e2` (entity spans)
- `relation` label

Our preprocessing script writes those into:

- `data_clean/benchmarks/semeval2010_task8/examples.jsonl`
- `data_clean/benchmarks/fewrel/examples.jsonl`

### 3A) SemEval 2010 Task-8 (recommended first)

Put the SemEval raw files here.

Easiest option: **rename** the official SemEval train/test files to match these names:

- `data_raw/benchmarks/semeval2010/TRAIN_FILE.TXT`
- `data_raw/benchmarks/semeval2010/TEST_FILE.TXT`

Then run:

```powershell
python scripts/preprocess.py --allow-pdf-failures `
  --semeval-train data_raw/benchmarks/semeval2010/TRAIN_FILE.TXT `
  --semeval-test  data_raw/benchmarks/semeval2010/TEST_FILE.TXT
```

If you don’t want to rename files, just use your real filenames instead. For example, if your folder contains:

- `data_raw/benchmarks/semeval2010/TRAIN_FILE.TXT` (example placeholder)
- `data_raw/benchmarks/semeval2010/TEST_FILE.TXT` (example placeholder)

…replace the filenames in the command with whatever you actually have.

### 3B) FewRel (optional)

Put a FewRel JSON file here:

- `data_raw/benchmarks/fewrel/fewrel.json`

Then run:

```powershell
python scripts/preprocess.py --allow-pdf-failures --fewrel-json data_raw/benchmarks/fewrel/fewrel.json
```

## 4) Train spaCy NER + Relation Extraction (RE)

This section assumes you ran **SemEval preprocessing** and have:

- `data_clean/benchmarks/semeval2010_task8/examples.jsonl`

### 4A) Build spaCy DocBins

Convert SemEval `examples.jsonl` into spaCy DocBin(s) (each produces a single `*_all.spacy` file):

```bash
python scripts/build_spacy_ner_data.py --examples data_clean/benchmarks/semeval2010_task8/examples.jsonl --out data_raw/tmp/ner_all.spacy
python scripts/build_spacy_re_data.py  --examples data_clean/benchmarks/semeval2010_task8/examples.jsonl --out data_raw/tmp/re_all.spacy
```

### 4B) Split train/dev (deterministic)

```bash
python scripts/split_spacy_docbin.py --in data_raw/tmp/ner_all.spacy --train-out data_raw/tmp/ner_train.spacy --dev-out data_raw/tmp/ner_dev.spacy --seed 42 --dev-ratio 0.2
python scripts/split_spacy_docbin.py --in data_raw/tmp/re_all.spacy  --train-out data_raw/tmp/re_train.spacy  --dev-out data_raw/tmp/re_dev.spacy  --seed 42 --dev-ratio 0.2
```

### 4C) Train models

```bash
python scripts/train_ner.py --train data_raw/tmp/ner_train.spacy --dev data_raw/tmp/ner_dev.spacy --output models/ner
python scripts/train_re.py  --train data_raw/tmp/re_train.spacy  --dev data_raw/tmp/re_dev.spacy  --output models/re
```

## 5) Run extraction (NER + RE) over cleaned MSOE docs

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

## More details

See `project.md`.
