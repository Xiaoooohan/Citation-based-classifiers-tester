"""Sentence-BERT embeddings + Logistic Regression impact classifier."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from splits import load_splits, evaluate, format_metrics

OUT = Path("data/processed/scores_embed.parquet")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def make_text(df: pd.DataFrame) -> list[str]:
    return (df.title.fillna("") + ". " + df.abstract.fillna("")).tolist()


def tune_threshold(y_true, y_score):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_score >= t).astype(int)
        m = evaluate(y_true, y_pred)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t, best_f1


def main():
    from sentence_transformers import SentenceTransformer
    train, val, test = load_splits()
    enc = SentenceTransformer(MODEL_NAME)
    print("encoding train..."); E_tr = enc.encode(make_text(train), batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    print("encoding val...");   E_va = enc.encode(make_text(val),   batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    print("encoding test...");  E_te = enc.encode(make_text(test),  batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    sc = StandardScaler()
    X_tr = sc.fit_transform(E_tr); X_va = sc.transform(E_va); X_te = sc.transform(E_te)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    clf.fit(X_tr, train.label.values)
    s_tr = clf.predict_proba(X_tr)[:, 1]
    s_va = clf.predict_proba(X_va)[:, 1]
    s_te = clf.predict_proba(X_te)[:, 1]

    t, vf1 = tune_threshold(val.label.values, s_va)
    print(f"tuned threshold on val = {t:.2f} (val F1={vf1:.3f})")

    print(format_metrics("embed-train", evaluate(train.label.values, (s_tr >= t).astype(int), s_tr)))
    print(format_metrics("embed-val",   evaluate(val.label.values,   (s_va >= t).astype(int), s_va)))
    print(format_metrics("embed-test",  evaluate(test.label.values,  (s_te >= t).astype(int), s_te)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "paper_id": test.paper_id.values,
        "year": test.year.values,
        "label": test.label.values,
        "score": s_te,
        "pred": (s_te >= t).astype(int),
    }).to_parquet(OUT, index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
