"""
Filter the raw OpenAlex pull to papers that are *actually* HCI.

OpenAlex tags many papers with HCI as a minor concept. We keep only papers where
'Human-computer interaction' appears in the top-3 concepts.

Input:  data/raw/papers.parquet
Output: data/processed/papers_hci.parquet
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW = Path("data/raw/papers.parquet")
OUT = Path("data/processed/papers_hci.parquet")
HCI_NAME = "Human\u2013computer interaction"  # en-dash, matches OpenAlex
TOP_K = 5

# Surveys/reviews inflate citations without introducing concepts — drop them.
REVIEW_PATTERNS = [
    r"\bsurvey\b",
    r"\breview\b",
    r"\bmeta[- ]analysis\b",
    r"\bliterature review\b",
    r"\boverview\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bsystematic review\b",
    r"\bscoping review\b",
]


def is_hci(concepts) -> bool:
    if concepts is None:
        return False
    top = list(concepts)[:TOP_K]
    return HCI_NAME in top


def main():
    df = pd.read_parquet(RAW)
    print(f"raw rows: {len(df)}")

    # Basic quality filters
    df = df[df["abstract"].str.len() > 100]
    df = df.drop_duplicates(subset=["paper_id"])
    print(f"after abstract/dedup filter: {len(df)}")

    # HCI concept filter
    mask = df["concepts"].apply(is_hci)
    df = df[mask].reset_index(drop=True)
    print(f"after HCI top-{TOP_K} filter: {len(df)}")

    # Drop survey/review papers (title OR first 200 chars of abstract)
    import re
    rx = re.compile("|".join(REVIEW_PATTERNS), flags=re.IGNORECASE)
    text = (df["title"].fillna("") + " " + df["abstract"].fillna("").str.slice(0, 200))
    review_mask = text.str.contains(rx)
    print(f"review/survey papers flagged: {review_mask.sum()}")
    df = df[~review_mask].reset_index(drop=True)
    print(f"after review filter: {len(df)}")

    # Per-year counts
    print("\nper-year counts:")
    print(df.groupby("year").size().to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
