"""Second novelty probe: semantic distance to prior-year centroid.

For each paper p in year y, compute 1 - cosine(embedding(p), mean_embedding(corpus where year < y)).
High value -> paper sits far from the prior-year cloud -> semantically novel.

Independent of citation counts and of the impact classifier.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ALL_HCI = Path("data/processed/papers_hci.parquet")
LABELED = Path("data/processed/papers_labeled.parquet")
OUT = Path("data/processed/novelty_embed.parquet")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    from sentence_transformers import SentenceTransformer
    corpus = pd.read_parquet(ALL_HCI).reset_index(drop=True)
    text = (corpus.title.fillna("") + ". " + corpus.abstract.fillna("")).tolist()
    enc = SentenceTransformer(MODEL_NAME)
    print(f"encoding {len(text)} HCI papers...")
    E = enc.encode(text, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    corpus["_idx"] = np.arange(len(corpus))

    years = sorted(corpus.year.unique().tolist())
    centroids = {}
    for y in years:
        prior = corpus[corpus.year < y]
        if len(prior) >= 20:
            centroids[y] = E[prior._idx.values].mean(axis=0)

    # score every corpus paper that has a centroid for its year
    sem_nov = np.full(len(corpus), np.nan)
    for y, c in centroids.items():
        c = c / (np.linalg.norm(c) + 1e-9)
        idx = corpus.index[corpus.year == y].values
        sims = E[idx] @ c
        sem_nov[idx] = 1.0 - sims  # cosine distance

    out = corpus[["paper_id", "year"]].copy()
    out["semantic_novelty"] = sem_nov

    # restrict to labelled set
    lab = pd.read_parquet(LABELED)[["paper_id", "label", "citation_count"]]
    out = out.merge(lab, on="paper_id", how="inner")
    out = out.dropna(subset=["semantic_novelty"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  rows={len(out)}")
    print(out.describe())

    from scipy.stats import spearmanr
    rho1, p1 = spearmanr(out.semantic_novelty, out.citation_count)
    rho2, p2 = spearmanr(out.semantic_novelty, out.label)
    print(f"Spearman(semantic_novelty, citations) = {rho1:+.3f} (p={p1:.2g})")
    print(f"Spearman(semantic_novelty, label)     = {rho2:+.3f} (p={p2:.2g})")


if __name__ == "__main__":
    main()
