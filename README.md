# GraphRAG System: Relation Extraction & Querying

This repository implements a complete pipeline for building and querying a Graph-based RAG (Retrieval-Augmented Generation) system. It covers **web scraping MSOE pages**, **deterministic text cleaning**, **supervised Relation Extraction (RE)** training with **5-fold cross-validation**, and a **self-contained terminal interface** for querying results.

---

## Quick Start: Run the GraphRAG demo

1) Create `.env` from `.env.example` and set:
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `GOOGLE_API_KEY`

2) Run:

```powershell
python scripts/graph_rag_query.py
```

This is the main “try the system” entrypoint: it retrieves context from Neo4j and asks Gemini to answer using only that context.

---

## Quick Start: Evaluation Prototype (RE component test)
This repository also includes a benchmark evaluation script for the supervised relation-extraction component (SemEval-2010 Task 8).

**Run from the project root:**
```powershell
python scripts/evaluate_prototype.py
```

### What this does:
*   **Dependency Bootstrap:** Automatically installs `spacy` and `scikit-learn` if missing.
*   **Model Loading:** Loads our pre-trained supervised RE model binary from `models/re_final/model-best`.
*   **Test Data:** Evaluates against the SemEval-2010 Task 8 test set (`test_data.jsonl`).
*   **Output Report:** Displays a prediction preview (first 10 items) and a full classification report (precision, recall, F1) for all 19 relation classes.

---

## Repository Structure
*   **`scripts/`**: Reproducible CLI tools.
    *   `evaluate_prototype.py`: Self-contained evaluation interface.
    *   `graph_rag_query.py`: Terminal interface for GraphRAG (Neo4j + Gemini).
    *   `train_cv_and_final.py`: 5-fold cross-validation and master model trainer.
    *   `preprocess.py`: Cleans raw HTML/PDF into deterministic JSONL.
    *   `run_extraction.py`: Runs trained NER+RE over the cleaned MSOE corpus.
*   **`models/`**: Trained spaCy pipelines (committed for zero-build usage).
*   **`extraction_spacy/`**: Source code for the custom `relation_extractor` spaCy component.
*   **`data_clean/`**: Deterministic cleaned corpora and benchmark examples.
*   **`configs/`**: spaCy training configurations.

---

## Technical Validation (RE benchmark vs GraphRAG system test)

Important: the SemEval CV metrics below test the **relation extraction (RE) classifier**, not MSOE/Neo4j GraphRAG retrieval quality.

### Latest benchmark snapshot (3-fold dev run)

- **What it is testing**: supervised RE classification on **SemEval-2010 Task-8**
- **Command**: `python scripts/train_cv_and_final.py --folds 3 --train-limit 3000 --skip-final --max-steps 2000`
- **Macro-F1 (excl. Other)**: **0.3634**
- **Accuracy**: **0.4291**

### System test (GraphRAG retrieval)

Use `scripts/eval_retrieval.py` with `data_clean/eval/retrieval_questions.jsonl` to report retrieval metrics (Recall@K / MRR). Add `--mode both` to also score **chunk → document** retrieval (closer to chunk-first GraphRAG).

**To re-run benchmark validation:**
```powershell
python scripts/train_cv_and_final.py
```
*(Warning: This trains 6 models total and may take 10-30 minutes.)*

---

## Initial Setup (Development)
If you wish to modify or retrain the system, follow these steps:

### Windows — Virtual Environment
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Full Pipeline Workflow:
1.  **Scrape Data:** `python scrape.py`
2.  **Clean Data:** `python scripts/preprocess.py --allow-pdf-failures`
3.  **Run Extraction:**
    ```powershell
    python scripts/run_extraction.py `
      --docs data_clean/msoe/documents.jsonl `
      --ner-model models/ner/model-best `
      --re-model models/re/model-best `
      --out data_clean/extracted
    ```

---

## Submission Checklist
For the grading instructor:
1.  **`scripts/evaluate_prototype.py`**: The main executable script.
2.  **`models/re_final/`**: The standalone model binary.
3.  **`test_data.jsonl`**: The set of test data used for validation.
4.  **`cv_results.txt`**: Records of the 5-fold cross-validation test run.
5.  **`GraphRAG_System_Paper.md`** (or README): Detailed system documentation.

---

## Git LFS Note
The large extracted relations file `data_clean/extracted/relations.jsonl` is managed via **Git LFS**. Ensure LFS is installed and pulled before running the full extraction scripts:
```bash
git lfs install
git lfs pull
```
