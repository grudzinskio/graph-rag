import argparse
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train spaCy relation extractor (custom component; wrapper around `spacy train`)."
    )
    ap.add_argument("--config", type=Path, default=Path("configs/re.cfg"))
    ap.add_argument("--train", type=Path, required=True, help="Path to train.spacy")
    ap.add_argument("--dev", type=Path, required=True, help="Path to dev.spacy")
    ap.add_argument("--output", type=Path, default=Path("models/re"))
    ap.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs")
    ap.add_argument("--max-steps", type=int, default=None, help="Override training.max_steps (0 = unlimited)")
    ap.add_argument("--eval-frequency", type=int, default=None, help="Override training.eval_frequency")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "spacy",
        "train",
        str(args.config),
        "--output",
        str(args.output),
        "--paths.train",
        str(args.train),
        "--paths.dev",
        str(args.dev),
        "--code",
        "extraction_spacy/relation_extractor.py",
    ]
    if args.max_epochs is not None:
        cmd += ["--training.max_epochs", str(int(args.max_epochs))]
    if args.max_steps is not None:
        cmd += ["--training.max_steps", str(int(args.max_steps))]
    if args.eval_frequency is not None:
        cmd += ["--training.eval_frequency", str(int(args.eval_frequency))]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

