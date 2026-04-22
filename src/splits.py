"""
Time-based train/val/test splits + evaluation harness.

Split rule:
    train : year <= 2019
    val   : 2020 <= year <= 2021
    test  : 2022 <= year <= 2023

Every model in this project is scored via `evaluate()` for consistency.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

LABELED = Path("data/processed/papers_labeled.parquet")

TRAIN_END = 2019
VAL_YEARS = (2020, 2021)
TEST_YEARS = (2022, 2023)


def load_splits():
    """Return (train, val, test) DataFrames."""
    df = pd.read_parquet(LABELED)
    train = df[df.year <= TRAIN_END].reset_index(drop=True)
    val = df[df.year.isin(VAL_YEARS)].reset_index(drop=True)
    test = df[df.year.isin(TEST_YEARS)].reset_index(drop=True)
    return train, val, test


def evaluate(y_true, y_pred, y_score=None) -> dict:
    """Compute standard metrics. y_score = probability of positive class (optional)."""
    out = {
        "n": int(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
            out["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = float("nan")
            out["pr_auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def format_metrics(name: str, m: dict) -> str:
    keys = ["n", "pos_rate", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    parts = [f"{k}={m.get(k, float('nan')):.3f}" if isinstance(m.get(k), float) else f"{k}={m.get(k)}"
             for k in keys if k in m]
    return f"{name:20s} | " + "  ".join(parts)


def main():
    """Smoke test: load splits and print baseline scores."""
    train, val, test = load_splits()
    print(f"train: n={len(train)}  pos={train.label.sum()}  neg={(train.label==0).sum()}  years={sorted(train.year.unique().tolist())}")
    print(f"val  : n={len(val)}    pos={val.label.sum()}   neg={(val.label==0).sum()}   years={sorted(val.year.unique().tolist())}")
    print(f"test : n={len(test)}   pos={test.label.sum()}  neg={(test.label==0).sum()}  years={sorted(test.year.unique().tolist())}")
    print()

    rng = np.random.default_rng(0)
    print("=== BASELINES (test set) ===")

    # Majority class (predict all 0)
    y_true = test.label.values
    y_pred_maj = np.zeros_like(y_true)
    print(format_metrics("majority-class", evaluate(y_true, y_pred_maj, y_score=np.zeros_like(y_true, dtype=float))))

    # Random with train prior
    p = train.label.mean()
    y_score_rand = rng.random(len(y_true))
    y_pred_rand = (y_score_rand >= (1 - p)).astype(int)
    print(format_metrics("random-prior", evaluate(y_true, y_pred_rand, y_score_rand)))


if __name__ == "__main__":
    main()
