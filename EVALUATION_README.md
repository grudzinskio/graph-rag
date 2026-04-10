# GraphRAG — Evaluation Prototype: Instructions

This document explains how to train, cross-validate, and evaluate the
relation extraction model. Two scripts handle the two phases.

---

## Step 1 — Build the Model (5-Fold CV + Final Training)

> **Skip this step if `models/re_final/model-best` is already committed
> to the repo.** The pre-trained binary is ready to use immediately.

Run from the **project root**:

```powershell
python scripts/train_cv_and_final.py
```

### What this script does

1. **Loads** the 10,717-item SemEval 2010 Task-8 dataset from
   `data_clean/benchmarks/semeval2010_task8/examples.jsonl`.
2. **Splits** data into the official `train` (8,000) and `test` (2,717) sets.
3. **Saves** the test split to `test_data.jsonl` (used by Step 2).
4. **Runs 5-fold cross-validation** on the training set:
   - Each fold trains a full spaCy RE model and evaluates it on the held-out
     fold dev split.
   - Prints **per-fold Macro-F1** (SemEval metric: macro-F1 excluding `Other`)
     and a full **classification report** to the terminal.
5. **Saves** all CV results to `cv_results.txt`.
6. **Trains the final master model** on 90% of the training data (10% is used
   only for spaCy's early-stopping; it does not affect the CV scores).
7. **Saves** the final model binary to `models/re_final/model-best`.

> **Runtime:** ~10–30 minutes depending on CPU/GPU (trains 6 spaCy models total).

---

## Step 2 — Run the Evaluation Prototype

Once `models/re_final/model-best` and `test_data.jsonl` exist, run:

```powershell
python scripts/evaluate_prototype.py
```

### What the grader sees

| Section | Description |
|---|---|
| **Dependency check** | Auto-installs `spacy` and `scikit-learn` if missing |
| **Prediction preview** | First 10 test examples with ✓/✗, predicted label, gold label, and confidence score |
| **Full classification report** | Per-class precision, recall, F1, support for all 19 relation types |
| **Official metric** | Macro-F1 excluding `Other` — the SemEval-2010 Task-8 standard |
| **Overall accuracy** | Raw accuracy across all 19 classes |

### How to interpret the output

- **Macro-F1 (excl. Other)** is the primary metric. Competitive systems score
  0.70–0.90; a baseline prototype is expected to score 0.20–0.40.
- **Accuracy** is inflated by the `Other` class (~21% of test data), so it is
  a secondary metric only.
- The **classification report** shows where the model struggles class-by-class.

---

## Test Data Format

`test_data.jsonl` contains one JSON object per line:

```json
{
  "text": "The most common audits were about waste and recycling.",
  "e1":   { "text": "audits", "char_start": 16, "char_end": 22 },
  "e2":   { "text": "waste",  "char_start": 34, "char_end": 39 },
  "relation": "Message-Topic(e1,e2)"
}
```

The script reads `test_data.jsonl` automatically — no path arguments needed.

---

## Submission Checklist

| Item | Path |
|---|---|
| Evaluation script | `scripts/evaluate_prototype.py` |
| Training + CV script | `scripts/train_cv_and_final.py` |
| Test data | `test_data.jsonl` |
| Model binary | `models/re_final/model-best/` |
| Custom RE component | `extraction_spacy/relation_extractor.py` |
| CV results log | `cv_results.txt` |
| Training data | `data_clean/benchmarks/semeval2010_task8/examples.jsonl` |
