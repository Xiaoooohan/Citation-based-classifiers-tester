"""Independent novelty probe — fraction of a paper's noun-phrase-like terms
that first appear in the corpus in this paper's own publication year.

Built only from text + year. Does NOT use any classifier or label.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

LABELED = Path("data/processed/papers_labeled.parquet")
ALL_HCI = Path("data/processed/papers_hci.parquet")  # broader corpus for first-year index
OUT = Path("data/processed/novelty_scores.parquet")

STOP = set("""a an the of in on for to with from by and or as is are was were be been being this that these those it its their them they we our you your he she his her i not no but if then than so such at into over under via using based use used study research paper we present propose proposed introduce introduces method methods approach approaches model models result results show shows shown found finding findings new novel can may might could should would also however moreover further furthermore additionally one two three first second third high low large small more most less least better best good""".split())

TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


def phrases(text: str) -> set[str]:
    """Unigrams + bigrams + trigrams (stopwords stripped)."""
    toks = tokens(text)
    out = set(toks)
    out.update(f"{a} {b}" for a, b in zip(toks, toks[1:]))
    out.update(f"{a} {b} {c}" for a, b, c in zip(toks, toks[1:], toks[2:]))
    return out


def main():
    # Build first-year index from the broader HCI corpus (more coverage)
    corpus = pd.read_parquet(ALL_HCI) if ALL_HCI.exists() else pd.read_parquet(LABELED)
    corpus = corpus.sort_values("year").reset_index(drop=True)
    print(f"corpus for first-year index: {len(corpus)} papers, years {corpus.year.min()}–{corpus.year.max()}")

    first_year: dict[str, int] = {}
    term_freq: dict[str, int] = defaultdict(int)
    for _, row in corpus.iterrows():
        text = (row.title or "") + " " + (row.abstract or "")
        ph = phrases(text)
        y = int(row.year)
        for p in ph:
            term_freq[p] += 1
            if p not in first_year or y < first_year[p]:
                first_year[p] = y
    print(f"unique phrases: {len(first_year)}")

    # Score the labeled set
    df = pd.read_parquet(LABELED)
    scores = []
    for _, row in df.iterrows():
        text = (row.title or "") + " " + (row.abstract or "")
        ph = phrases(text)
        # Restrict to phrases with corpus freq >= 2 (drop one-off OCR-like junk)
        ph = [p for p in ph if term_freq.get(p, 0) >= 2]
        y = int(row.year)
        if not ph:
            scores.append(0.0)
            continue
        n_new = sum(1 for p in ph if first_year.get(p) == y)
        scores.append(n_new / len(ph))

    out = pd.DataFrame({
        "paper_id": df.paper_id.values,
        "year": df.year.values,
        "label": df.label.values,
        "novelty_score": scores,
        "citation_count": df.citation_count.values,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}")
    print(out.describe())

    # Sanity: Spearman correlation novelty vs citations
    from scipy.stats import spearmanr
    rho, p = spearmanr(out.novelty_score, out.citation_count)
    print(f"\nSpearman(novelty, citations) = {rho:+.3f}  (p={p:.3g})")
    rho2, p2 = spearmanr(out.novelty_score, out.label)
    print(f"Spearman(novelty, label)     = {rho2:+.3f}  (p={p2:.3g})")


if __name__ == "__main__":
    main()
