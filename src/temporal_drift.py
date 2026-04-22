"""Temporal-drift experiment: for each cutoff year Y, train TF-IDF+LR on year<=Y
and evaluate on year=Y+1, Y+2, Y+3. Saves a curve of test-PR-AUC vs gap."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

LABELED = Path("data/processed/papers_labeled.parquet")
OUT = Path("data/processed/temporal_drift.parquet")


def fit_predict(train, test):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform((train.title.fillna("") + " " + train.abstract.fillna("")).tolist())
    Xte = vec.transform((test.title.fillna("") + " " + test.abstract.fillna("")).tolist())
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    clf.fit(Xtr, train.label.values)
    return clf.predict_proba(Xte)[:, 1]


def main():
    df = pd.read_parquet(LABELED)
    rows = []
    for cutoff in range(2014, 2021):  # 2014..2020 inclusive
        train = df[df.year <= cutoff]
        if train.label.sum() < 10:
            continue
        for gap in (1, 2, 3):
            test = df[df.year == cutoff + gap]
            if len(test) < 20 or test.label.sum() < 3:
                continue
            s = fit_predict(train, test)
            pr = float(average_precision_score(test.label, s))
            # threshold = positive rate of train
            t = float(np.quantile(s, 1 - train.label.mean()))
            f1 = float(f1_score(test.label, (s >= t).astype(int), zero_division=0))
            rows.append({"cutoff": cutoff, "gap": gap, "test_year": cutoff + gap,
                          "n_train": int(len(train)), "n_test": int(len(test)),
                          "pr_auc": pr, "f1": f1})
            print(f"cutoff<= {cutoff}  test={cutoff+gap}  n_test={len(test):3d}  PR-AUC={pr:.3f}  F1={f1:.3f}")
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")
    print(out.groupby("gap")[["pr_auc", "f1"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
