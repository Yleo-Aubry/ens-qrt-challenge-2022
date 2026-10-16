"""Fit the model on the 50 training stocks and write the submission file.

Usage, from the repository root:
    python scripts/make_submission.py submissions/new_submission.csv
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrt_model import fit, load_data, save_submission, score  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="output CSV file (must not exist)")
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="folder with X_train.csv and Y_train.csv")
    args = parser.parse_args()

    X, y, _, _ = load_data(args.data)
    A, beta = fit(X, y)
    save_submission(A, beta, args.out)
    print(f"in-sample score: {score(y, X @ A @ beta):.4f}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
