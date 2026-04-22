"""TF-IDF + Logistic Regression impact classifier."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from splits import load_splits, evaluate, format_metrics

OUT = Path("data/processed/scores_tfidf.parquet")
FEATURES_OUT = Path("reports/features.md")


def make_text(df: pd.DataFrame) -> list[str]:
    return (df.title.fillna("") + " " + df.abstract.fillna("")).tolist()


def tune_threshold(y_true, y_score):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_score >= t).astype(int)
        m = evaluate(y_true, y_pred)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t, best_f1


def main():
    train, val, test = load_splits()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True)
    X_train = vec.fit_transform(make_text(train))
    X_val = vec.transform(make_text(val))
    X_test = vec.transform(make_text(test))

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
    clf.fit(X_train, train.label.values)

    s_train = clf.predict_proba(X_train)[:, 1]
    s_val = clf.predict_proba(X_val)[:, 1]
    s_test = clf.predict_proba(X_test)[:, 1]

    t, vf1 = tune_threshold(val.label.values, s_val)
    print(f"tuned threshold on val = {t:.2f} (val F1={vf1:.3f})")

    print(format_metrics("tfidf-train", evaluate(train.label.values, (s_train >= t).astype(int), s_train)))
    print(format_metrics("tfidf-val",   evaluate(val.label.values,   (s_val   >= t).astype(int), s_val)))
    print(format_metrics("tfidf-test",  evaluate(test.label.values,  (s_test  >= t).astype(int), s_test)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "paper_id": test.paper_id.values,
        "year": test.year.values,
        "label": test.label.values,
        "score": s_test,
        "pred": (s_test >= t).astype(int),
    }).to_parquet(OUT, index=False)
    print(f"wrote {OUT}")

    # Top features by signed coefficient
    feat_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_[0]
    top_pos = np.argsort(coefs)[-20:][::-1]
    top_neg = np.argsort(coefs)[:20]
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Top TF-IDF features (Logistic Regression coefficients)\n",
             "## Top 20 positive (predict high-impact)\n", "| term | coef |\n|---|---|"]
    lines += [f"| {feat_names[i]} | {coefs[i]:+.3f} |" for i in top_pos]
    lines += ["\n## Top 20 negative (predict low-impact)\n", "| term | coef |\n|---|---|"]
    lines += [f"| {feat_names[i]} | {coefs[i]:+.3f} |" for i in top_neg]
    FEATURES_OUT.write_text("\n".join(lines))
    print(f"wrote {FEATURES_OUT}")


if __name__ == "__main__":
    main()
