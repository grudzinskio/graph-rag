"""
GraphRAG: Relation Extraction Evaluation Prototype
===================================================
Usage (from project root):
    python scripts/evaluate_prototype.py

Prerequisites:
    Run  python scripts/train_cv_and_final.py  first to generate:
        • models/re_final/model-best   (trained RE model binary)
        • test_data.jsonl              (held-out SemEval test split)

This script is self-contained: it auto-installs spacy and scikit-learn
if they are missing, then loads the saved model binary and evaluates it
against the gold test data without any additional configuration.

Output:
    • Preview of the first 10 predictions (with [OK]/[X] correctness marks)
    • Full classification report (all 19 relation classes)
    • SemEval-2010 Task-8 official metric: macro-F1 excluding "Other"
    • Overall accuracy
"""

import subprocess
import sys
from pathlib import Path

# Force UTF-8 stdout so Windows cp1252 terminals don't choke on Unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ── 0. Dependency bootstrap ──────────────────────────────────────────
def _ensure(pip_name: str, import_name: str) -> None:
    try:
        __import__(import_name)
    except ImportError:
        print(f"  Installing '{pip_name}'...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name],
            stdout=subprocess.DEVNULL,
        )

print("Checking dependencies...")
_ensure("spacy", "spacy")
_ensure("scikit-learn", "sklearn")
print("  OK.\n")

# ── 1. Remaining imports (after bootstrap) ───────────────────────────
import json  # noqa: E402

import spacy  # noqa: E402
from spacy.tokens import Doc  # noqa: E402
from spacy.util import filter_spans  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    f1_score,
)

# ── 2. Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MODEL_DIR      = PROJECT_ROOT / "models/re_final/model-best"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.jsonl"

# Project root must be on sys.path so extraction_spacy is importable
sys.path.insert(0, str(PROJECT_ROOT))

# ── 3. Register the custom spaCy component ───────────────────────────
try:
    import extraction_spacy.relation_extractor  # noqa: F401 - side-effect: registers factory
except ImportError:
    print("ERROR: Could not import 'extraction_spacy.relation_extractor'.")
    print("       Ensure you are running from the project root and that the")
    print("       'extraction_spacy/' directory is present.")
    sys.exit(1)


def ensure_rel_extension() -> None:
    if not Doc.has_extension("rel"):
        Doc.set_extension("rel", default={})

ensure_rel_extension()


# ── 4. Main ──────────────────────────────────────────────────────────
def main() -> None:
    sep = "=" * 65
    print(sep)
    print("  GraphRAG -- Relation Extraction Evaluation Prototype")
    print(sep)

    # Validate required files exist
    if not MODEL_DIR.exists():
        print(f"\nERROR: Model binary not found at:\n  {MODEL_DIR}")
        print("Run  python scripts/train_cv_and_final.py  to generate it.")
        sys.exit(1)

    if not TEST_DATA_PATH.exists():
        print(f"\nERROR: Test data not found at:\n  {TEST_DATA_PATH}")
        print("Run  python scripts/train_cv_and_final.py  to generate it.")
        sys.exit(1)

    # Load model
    print(f"\nLoading model from:\n  {MODEL_DIR}")
    nlp = spacy.load(MODEL_DIR)

    if "relation_extractor" not in nlp.pipe_names:
        print(
            f"ERROR: Loaded pipeline has no 'relation_extractor' component.\n"
            f"  Found: {nlp.pipe_names}\n"
            f"  Make sure MODEL_DIR points at the RE model, not the NER model."
        )
        sys.exit(1)

    re_pipe          = nlp.get_pipe("relation_extractor")
    labels_in_model  = list(re_pipe.labels)
    non_other        = sorted(l for l in labels_in_model if l != "Other")
    all_labels       = sorted(labels_in_model)

    print(f"  Components     : {nlp.pipe_names}")
    print(f"  Relation labels: {len(labels_in_model)}")

    # Load test data
    test_examples = []
    with open(TEST_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_examples.append(json.loads(line))

    print(f"\nLoaded {len(test_examples):,} test examples from:\n  {TEST_DATA_PATH.name}")

    # Inference
    print("\n" + "-" * 65)
    print("  Running inference...  (first 10 predictions shown below)")
    print("-" * 65 + "\n")

    gold_labels: list = []
    pred_labels: list = []
    skipped = 0

    for i, ex in enumerate(test_examples):
        text    = ex.get("text", "")
        e1_meta = ex.get("e1", {})
        e2_meta = ex.get("e2", {})
        gold    = ex.get("relation", "Other")

        if not text or not e1_meta or not e2_meta:
            skipped += 1
            continue

        doc = nlp.make_doc(text)
        s1 = doc.char_span(e1_meta["char_start"], e1_meta["char_end"],
                           label="ENT", alignment_mode="contract")
        s2 = doc.char_span(e2_meta["char_start"], e2_meta["char_end"],
                           label="ENT", alignment_mode="contract")

        if s1 is None or s2 is None:
            skipped += 1
            continue

        filtered = filter_spans([s1, s2])
        if len(filtered) < 2:
            skipped += 1
            continue

        doc.ents = filtered

        # Run every component in order (does NOT re-tokenize)
        for _, pipe in nlp.pipeline:
            doc = pipe(doc)

        # Re-derive token indices from original char offsets (stable after make_doc)
        s1_post = doc.char_span(e1_meta["char_start"], e1_meta["char_end"],
                                alignment_mode="contract")
        s2_post = doc.char_span(e2_meta["char_start"], e2_meta["char_end"],
                                alignment_mode="contract")

        predicted    = "Other"
        best_score   = -1.0

        if s1_post and s2_post and doc._.rel:
            # Look up the specific directional pair (e1 -> e2)
            key = (s1_post.start, s2_post.start)
            pair_scores = doc._.rel.get(key, {})
            if pair_scores:
                predicted  = max(pair_scores, key=pair_scores.get)
                best_score = pair_scores[predicted]

        gold_labels.append(gold)
        pred_labels.append(predicted)

        # Print preview of first 10
        if i < 10:
            mark   = "[OK]" if predicted == gold else "[X]"
            trunc  = text[:75] + "..." if len(text) > 75 else text
            e1_txt = e1_meta.get("text", "?")
            e2_txt = e2_meta.get("text", "?")
            print(f"[{i+1:2d}] {mark} \"{trunc}\"")
            print(f"      e1='{e1_txt}'  ->  e2='{e2_txt}'")
            print(f"      Predicted : {predicted:<42s} (conf: {best_score:.4f})")
            print(f"      Gold      : {gold}")
            print()

    if skipped:
        print(f"  (Skipped {skipped} malformed / un-alignable examples)\n")

    total = len(gold_labels)
    if total == 0:
        print("ERROR: No valid examples could be evaluated. Check test_data.jsonl format.")
        sys.exit(1)

    # ── Metrics ──────────────────────────────────────────────────────
    print("\n" + sep)
    print("  EVALUATION RESULTS")
    print(sep)

    print("\n--- Full Classification Report (all relation classes) ---\n")
    print(
        classification_report(
            gold_labels,
            pred_labels,
            labels=all_labels,
            zero_division=0,
        )
    )

    macro_f1 = f1_score(
        gold_labels, pred_labels,
        labels=non_other,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(gold_labels, pred_labels)

    print("--- SemEval-2010 Task-8 Official Metric ---")
    print("    Macro-F1 over the 18 directional relation types, excluding 'Other'\n")
    print(f"  Macro-F1  (excl. Other) : {macro_f1:.4f}   ({macro_f1 * 100:.2f}%)")
    print(f"  Accuracy  (all classes) : {accuracy:.4f}   ({accuracy * 100:.2f}%)")
    print(f"  Evaluated               : {total:,} examples")
    print(f"  Skipped                 : {skipped:,} examples")

    print("\n" + sep)
    print("  HOW TO INTERPRET")
    print(sep)
    print("  Macro-F1 (excl. Other) is the SemEval-2010 Task-8 official metric.")
    print("  Competitive systems score ~0.70-0.90; a prototype baseline is ~0.20-0.40.")
    print("  Accuracy is inflated by the 'Other' class (~21% of test data).")
    print("  See the classification report above for per-class precision/recall/F1.")
    print(sep + "\n")


if __name__ == "__main__":
    main()
