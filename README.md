# graph-rag

Pipeline for **scraping MSOE pages**, **cleaning text**, **training spaCy NER + relation extraction** (SemEval Task-8), and **running extraction** over cleaned documents. Graph store and RAG are not in this repo yet.

**Committing artifacts:** This repo is set up so you can commit **`models/`**, **`data_clean/`**, **`data_raw/`**, **`*.spacy`**, and **`scraped_data/`** if you want clones to work **without retraining** (see `.gitignore` — only `.venv/`, `.env`, Python cache, and `*.log` are ignored). GitHub blocks single files **over 100 MB**; **`data_clean/extracted/relations.jsonl`** is stored with [**Git LFS**](https://git-lfs.github.com/) (see `.gitattributes`).

After cloning, install LFS once and fetch blobs: `git lfs install` then `git lfs pull` (or clone with `git lfs clone` if your Git version supports it).

---

## Initial setup (run this first)

### Windows — virtual environment (recommended)

PowerShell:

```powershell
cd "C:\path\to\graph-rag"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### What we run for the “app” (extraction on cleaned MSOE text)

You need **`data_clean/msoe/documents.jsonl`** and trained pipelines under **`models/`**.

- If you **cloned a repo that already commits** those paths, you can **skip preprocessing** and go straight to **Run extraction** below.
- If `documents.jsonl` is missing, run preprocessing once (step 1).

1. **Build cleaned MSOE corpus** (skip if `data_clean/msoe/documents.jsonl` is already in the repo):

```powershell
python scripts/preprocess.py --allow-pdf-failures
```

1. **Run NER + relation extraction** over the cleaned corpus (safe to re-run after canceling a job — it **overwrites** `entities.jsonl` / `relations.jsonl`):

```powershell
python scripts/run_extraction.py `
  --docs data_clean/msoe/documents.jsonl `
  --ner-model models/ner/model-best `
  --re-model models/re/model-best `
  --out data_clean/extracted
```

Outputs (under `data_clean/extracted/`):

- `entities.jsonl` — one row per NER span (`ENT`), with `doc_id` and character offsets.
- `relations.jsonl` — one row per **scored entity pair** (label + `score`), so line counts are large; threshold or top‑k before treating rows as graph edges.

A full run over **5,714** cleaned MSOE docs produced **108,569** entity rows and **656,506** relation rows in ~**852 s** (see `project.md` → **Run log**).

To cap cost while debugging, add e.g. `--limit 100`.

**Watching a long run:** `run_extraction.py` logs to stderr every `--progress-every` docs (default **50**). Mirror the same lines to a file:

```powershell
python scripts/run_extraction.py `
  --docs data_clean/msoe/documents.jsonl `
  --ner-model models/ner/model-best `
  --re-model models/re/model-best `
  --out data_clean/extracted `
  --progress-every 25 `
  --log-file data_clean/extracted/extraction.log
```

In another terminal you can **tail** the file: `Get-Content data_clean/extracted/extraction.log -Wait` (PowerShell). Or run in the foreground so lines print live.

---

## If you want to scrape data

Use this when you need **fresh or missing** raw pages under `scraped_data/`.

1. **Scrape** (writes `scraped_data/.../raw_html` and `.../text`):

```powershell
python scrape.py
```

1. **Clean into `data_clean/`** again:

```powershell
python scripts/preprocess.py --allow-pdf-failures
```

Notes:

- Some catalog `*pdf.txt` files are raw PDF bytes; we try PyMuPDF. Failures go to `data_clean/quarantine/msoe/` when you pass `--allow-pdf-failures`.
- SemEval **raw** train/test files (for training) are not produced by the scraper; place them under `data_raw/benchmarks/semeval2010/` when retraining (see next section).

---

## If you want to retrain relation extraction (and NER)

Relation extraction (and the NER model we use for entity spans on MSOE) is trained on **SemEval 2010 Task-8** examples. Retraining does **not** require rescraping MSOE.

### 1) Put SemEval files in the repo layout

Example (rename your downloads to match, or adjust paths):

- `data_raw/benchmarks/semeval2010/TRAIN_FILE.TXT`
- `data_raw/benchmarks/semeval2010/TEST_FILE.TXT`

### 2) Regenerate `examples.jsonl` and refresh MSOE cleaning in one run

```powershell
python scripts/preprocess.py --allow-pdf-failures `
  --semeval-train data_raw/benchmarks/semeval2010/TRAIN_FILE.TXT `
  --semeval-test  data_raw/benchmarks/semeval2010/TEST_FILE.TXT
```

This writes `data_clean/benchmarks/semeval2010_task8/examples.jsonl` (10,717 examples when both splits are included).

### 3) Build spaCy DocBins, split train/dev, train

```powershell
python scripts/build_spacy_ner_data.py --examples data_clean/benchmarks/semeval2010_task8/examples.jsonl --out data_raw/tmp/ner_all.spacy
python scripts/build_spacy_re_data.py  --examples data_clean/benchmarks/semeval2010_task8/examples.jsonl --out data_raw/tmp/re_all.spacy

python scripts/split_spacy_docbin.py --in data_raw/tmp/ner_all.spacy --train-out data_raw/tmp/ner_train.spacy --dev-out data_raw/tmp/ner_dev.spacy --seed 42 --dev-ratio 0.2
python scripts/split_spacy_docbin.py --in data_raw/tmp/re_all.spacy  --train-out data_raw/tmp/re_train.spacy  --dev-out data_raw/tmp/re_dev.spacy  --seed 42 --dev-ratio 0.2

python scripts/train_ner.py --train data_raw/tmp/ner_train.spacy --dev data_raw/tmp/ner_dev.spacy --output models/ner
python scripts/train_re.py  --train data_raw/tmp/re_train.spacy  --dev data_raw/tmp/re_dev.spacy  --output models/re
```

RE training uses `--code extraction_spacy/relation_extractor.py` (wrapped inside `scripts/train_re.py`).

Trained artifacts to commit (not gitignored):

- `models/ner/model-best`, `models/ner/model-last`
- `models/re/model-best`, `models/re/model-last`

---

## More detail

See [project.md](project.md) for design notes and **recorded outputs** from our runs.
