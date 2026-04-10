# -*- coding: utf-8 -*-
"""
GraphRAG: 5-Fold Cross Validation + Final Model Training
=========================================================
Usage (from project root):
    python scripts/train_cv_and_final.py

What it does:
  1. Loads SemEval 2010 Task-8 JSONL from data_clean/benchmarks/...
  2. Runs 5-fold CV on the official TRAIN split, printing and saving
     per-fold macro-F1 (SemEval metric: macro-F1 excluding "Other").
  3. Saves a full CV report to cv_results.txt.
  4. Trains a final model on 90% of train data (10% held out only for
     spaCy early-stopping; does NOT affect the reported CV scores).
  5. Saves the final model to models/re_final/
  6. Saves the official TEST split to test_data.jsonl.
"""

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

# ── Project root on sys.path so extraction_spacy is importable ──────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force UTF-8 stdout so Unicode chars don't crash on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import spacy  # noqa: E402  (needs path insert first)
from spacy.tokens import Doc  # noqa: E402
from spacy.util import filter_spans  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

try:
    import extraction_spacy.relation_extractor  # registers @Language.factory
except ImportError as exc:
    print(f"ERROR: Cannot import extraction_spacy.relation_extractor -- {exc}")
    print("       Run this script from the project root directory.")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────
INPUT_DATA      = PROJECT_ROOT / "data_clean/benchmarks/semeval2010_task8/examples.jsonl"
SPACY_BUILD     = PROJECT_ROOT / "scripts/build_spacy_re_data.py"
SPACY_TRAIN     = PROJECT_ROOT / "scripts/train_re.py"
TMP_DIR         = PROJECT_ROOT / "tmp_cv"
FINAL_MODEL_DIR = PROJECT_ROOT / "models/re_final"
TEST_DATA_PATH  = PROJECT_ROOT / "test_data.jsonl"
CV_RESULTS_PATH = PROJECT_ROOT / "cv_results.txt"

N_FOLDS     = 5
RANDOM_SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────

def ensure_rel_extension() -> None:
    if not Doc.has_extension("rel"):
        Doc.set_extension("rel", default={})


def run_build(jsonl_path: Path, spacy_path: Path, labels_str: str) -> None:
    """Convert JSONL -> .spacy DocBin with a fixed global label set."""
    subprocess.check_call(
        [
            sys.executable, str(SPACY_BUILD),
            "--examples", str(jsonl_path),
            "--out",      str(spacy_path),
            "--labels",   labels_str,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_train(train_spacy: Path, dev_spacy: Path, output_dir: Path) -> None:
    """Invoke scripts/train_re.py (wraps `python -m spacy train`)."""
    subprocess.check_call(
        [
            sys.executable, str(SPACY_TRAIN),
            "--train",  str(train_spacy),
            "--dev",    str(dev_spacy),
            "--output", str(output_dir),
        ]
    )


def load_re_model(model_dir: Path) -> spacy.Language:
    ensure_rel_extension()
    return spacy.load(model_dir)


def semeval_macro_f1(gold: list, pred: list, non_other_labels: list) -> float:
    """Macro-F1 over all relation classes *excluding* 'Other'."""
    if not non_other_labels:
        return 0.0
    return f1_score(gold, pred, labels=non_other_labels, average="macro", zero_division=0)


def evaluate_examples(nlp: spacy.Language, examples: list) -> tuple:
    """
    Run inference over gold examples with manually injected entities.
    Returns (gold_labels, pred_labels).
    """
    ensure_rel_extension()
    gold_labels, pred_labels = [], []

    for ex in examples:
        text    = ex.get("text", "")
        e1_meta = ex.get("e1", {})
        e2_meta = ex.get("e2", {})
        gold    = ex.get("relation", "Other")

        if not text or not e1_meta or not e2_meta:
            continue

        doc = nlp.make_doc(text)
        s1 = doc.char_span(e1_meta["char_start"], e1_meta["char_end"],
                           label="ENT", alignment_mode="contract")
        s2 = doc.char_span(e2_meta["char_start"], e2_meta["char_end"],
                           label="ENT", alignment_mode="contract")

        if s1 is None or s2 is None:
            continue

        filtered = filter_spans([s1, s2])
        if len(filtered) < 2:
            continue

        doc.ents = filtered

        # Run every pipeline component in order
        for _, pipe in nlp.pipeline:
            doc = pipe(doc)

        # Re-derive token indices from original char offsets (stable after make_doc)
        s1_post = doc.char_span(e1_meta["char_start"], e1_meta["char_end"],
                                alignment_mode="contract")
        s2_post = doc.char_span(e2_meta["char_start"], e2_meta["char_end"],
                                alignment_mode="contract")

        predicted = "Other"
        if s1_post and s2_post and doc._.rel:
            key = (s1_post.start, s2_post.start)
            pair_scores = doc._.rel.get(key, {})
            if pair_scores:
                predicted = max(pair_scores, key=pair_scores.get)

        gold_labels.append(gold)
        pred_labels.append(predicted)

    return gold_labels, pred_labels


# -- Main -------------------------------------------------------------

def main() -> None:
    sep = "=" * 65
    print(sep)
    print("  GraphRAG - 5-Fold CV + Final Model Training")
    print(sep)

    if not INPUT_DATA.exists():
        print(f"ERROR: Dataset not found at:\n  {INPUT_DATA}")
        print("Run preprocessing first (see README).")
        sys.exit(1)

    # 1. Load data ────────────────────────────────────────────────────
    all_examples = []
    with open(INPUT_DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_examples.append(json.loads(line))

    train_data = [e for e in all_examples if e.get("split") == "train"]
    test_data  = [e for e in all_examples if e.get("split") == "test"]

    print(f"\nDataset: {len(train_data):,} train  |  {len(test_data):,} test")

    # Save the official test split for evaluate_prototype.py
    with open(TEST_DATA_PATH, "w", encoding="utf-8") as f:
        for ex in test_data:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved test split -> {TEST_DATA_PATH.name}")

    # 2. Global label set ─────────────────────────────────────────────
    labels_set = sorted({ex["relation"] for ex in train_data if "relation" in ex})
    non_other  = [l for l in labels_set if l != "Other"]
    global_labels_str = ";".join(labels_set)
    print(f"Relation labels: {len(labels_set)}  (non-Other: {len(non_other)})\n")

    # 3. Prepare tmp dir ──────────────────────────────────────────────
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # 4. 5-Fold Cross Validation ──────────────────────────────────────
    print("-" * 65)
    print(f"  {N_FOLDS}-Fold Cross Validation")
    print("-" * 65)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    log_lines = [
        f"GraphRAG RE -- {N_FOLDS}-Fold Cross Validation Results",
        "=" * 65,
        "",
    ]
    fold_f1s, fold_accs = [], []

    for fold, (train_idx, dev_idx) in enumerate(kf.split(train_data)):
        fold_num = fold + 1
        print(f"\n[ Fold {fold_num}/{N_FOLDS} ]")
        log_lines.append(f"[ Fold {fold_num}/{N_FOLDS} ]")

        fold_train = [train_data[i] for i in train_idx]
        fold_dev   = [train_data[i] for i in dev_idx]
        print(f"  Train: {len(fold_train):,}  |  Dev: {len(fold_dev):,}")

        # Write JSONL
        j_train = TMP_DIR / f"train_f{fold}.jsonl"
        j_dev   = TMP_DIR / f"dev_f{fold}.jsonl"
        with open(j_train, "w", encoding="utf-8") as f:
            for ex in fold_train: f.write(json.dumps(ex) + "\n")
        with open(j_dev, "w", encoding="utf-8") as f:
            for ex in fold_dev: f.write(json.dumps(ex) + "\n")

        # Convert to .spacy
        s_train = TMP_DIR / f"train_f{fold}.spacy"
        s_dev   = TMP_DIR / f"dev_f{fold}.spacy"
        print("  Building DocBins...")
        run_build(j_train, s_train, global_labels_str)
        run_build(j_dev,   s_dev,   global_labels_str)

        # Train
        fold_model_dir = TMP_DIR / f"model_f{fold}"
        print(f"  Training fold {fold_num} (may take several minutes)...")
        run_train(s_train, s_dev, fold_model_dir)

        # Evaluate
        best = fold_model_dir / "model-best"
        if not best.exists():
            best = fold_model_dir / "model-last"
        print(f"  Evaluating on dev split ({len(fold_dev):,} examples)...")
        nlp = load_re_model(best)
        gold, pred = evaluate_examples(nlp, fold_dev)
        del nlp

        # Metrics
        macro_f1 = semeval_macro_f1(gold, pred, non_other)
        accuracy  = sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1)
        fold_f1s.append(macro_f1)
        fold_accs.append(accuracy)

        report = classification_report(gold, pred, labels=labels_set, zero_division=0)
        summary = (f"  Macro-F1 (excl. Other): {macro_f1:.4f}  |  "
                   f"Accuracy: {accuracy:.4f}  |  Evaluated: {len(gold):,}")

        print(summary)
        print("  Classification Report:\n")
        print(report)

        log_lines += [
            f"  Train: {len(fold_train):,}  |  Dev: {len(fold_dev):,}",
            summary,
            "  Classification Report:",
            report,
            "",
        ]

    # 5. CV Summary ───────────────────────────────────────────────────
    avg_f1  = sum(fold_f1s)  / len(fold_f1s)
    avg_acc = sum(fold_accs) / len(fold_accs)
    std_f1  = (sum((x - avg_f1)**2 for x in fold_f1s) / len(fold_f1s)) ** 0.5

    cv_summary = [
        "",
        "=" * 65,
        "  CROSS VALIDATION SUMMARY",
        "=" * 65,
        "  Per-fold Macro-F1 : " + "  ".join(f"{v:.4f}" for v in fold_f1s),
        "  Per-fold Accuracy : " + "  ".join(f"{v:.4f}" for v in fold_accs),
        "",
        f"  Mean Macro-F1 (excl. Other) : {avg_f1:.4f}",
        f"  Std  Macro-F1               : {std_f1:.4f}",
        f"  Mean Accuracy               : {avg_acc:.4f}",
        "=" * 65,
    ]
    for line in cv_summary:
        print(line)
    log_lines += cv_summary

    with open(CV_RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nCV results saved -> {CV_RESULTS_PATH.name}")

    # 6. Final Model Training ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Training Final Master Model")
    print("=" * 65)

    # Use a 90/10 split of training data so spaCy has a real dev set
    # for early-stopping. This holdout does NOT affect CV scores above.
    rng = random.Random(RANDOM_SEED)
    shuffled = train_data[:]
    rng.shuffle(shuffled)
    dev_size    = max(1, len(shuffled) // 10)
    final_dev   = shuffled[:dev_size]
    final_train = shuffled[dev_size:]

    print(f"\n  Final train : {len(final_train):,}")
    print(f"  Monitor dev : {len(final_dev):,}  (10% holdout for early-stopping only)")

    j_ftrain = TMP_DIR / "final_train.jsonl"
    j_fdev   = TMP_DIR / "final_dev.jsonl"
    with open(j_ftrain, "w", encoding="utf-8") as f:
        for ex in final_train: f.write(json.dumps(ex) + "\n")
    with open(j_fdev, "w", encoding="utf-8") as f:
        for ex in final_dev: f.write(json.dumps(ex) + "\n")

    s_ftrain = TMP_DIR / "final_train.spacy"
    s_fdev   = TMP_DIR / "final_dev.spacy"
    print("  Building final DocBins...")
    run_build(j_ftrain, s_ftrain, global_labels_str)
    run_build(j_fdev,   s_fdev,   global_labels_str)

    # Remove any pre-existing final model dir — spaCy refuses non-empty output dirs.
    if FINAL_MODEL_DIR.exists():
        print(f"  Removing existing model dir: {FINAL_MODEL_DIR}")
        shutil.rmtree(FINAL_MODEL_DIR)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Training -> {FINAL_MODEL_DIR} ...")
    run_train(s_ftrain, s_fdev, FINAL_MODEL_DIR)

    # Cleanup intermediates (keep cv_results.txt)
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
    print(f"  Final model  -> {FINAL_MODEL_DIR}/model-best")
    print(f"  Test data    -> {TEST_DATA_PATH.name}")
    print(f"  CV results   -> {CV_RESULTS_PATH.name}")
    print(f"\n  Next: python scripts/evaluate_prototype.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
