"""Trains the SVM classifier on the 4-d pose-angle features -- the exact
arm the ablation study selected as its winner (RBF-kernel SVM, all
features, no PCA). Default: the 2,000-image synthetic dataset only.
`--with-personal <csv>` concatenates a personal-session CSV (same 4-column
schema, produced by `data_collection.py`) before fitting, per the "both
data sources" decision in the design spec.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .. import config
from .dataset import build_dataset, ensure_downloaded, load_personal_csv

MODEL_PICKLE_PATH = config.MODELS_DIR / "posture_svm.joblib"


def train(with_personal: Path | None = None, n_per_class: int = 1000) -> tuple[Pipeline, list[str]]:
    ensure_downloaded(n_per_class)
    print("Building features from the synthetic dataset...")
    X, y = build_dataset()

    if with_personal is not None:
        print(f"Adding personal session data from {with_personal}...")
        X_personal, y_personal = load_personal_csv(with_personal)
        X = np.concatenate([X, X_personal], axis=0)
        y = np.concatenate([y, y_personal], axis=0)

    labels = sorted(set(y.tolist()))
    print(f"Training SVM on {len(y)} samples, labels={labels}...")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )
    pipeline.fit(X, y)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PICKLE_PATH)
    print(f"Saved trained pipeline to {MODEL_PICKLE_PATH}")

    return pipeline, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-personal", type=Path, default=None, help="path to a personal-session CSV to add to training data"
    )
    parser.add_argument("--n", type=int, default=1000, help="images per class to download/use from the synthetic dataset")
    args = parser.parse_args()

    train(with_personal=args.with_personal, n_per_class=args.n)


if __name__ == "__main__":
    main()
