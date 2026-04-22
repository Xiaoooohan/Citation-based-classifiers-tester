"""
Build citation-percentile labels, bucketed by year to remove age bias.

Rule:
    Within each publication year, compute the citation percentile rank.
    y = 1  if percentile >= 0.90  (top 10%)
    y = 0  if percentile <= 0.50  (bottom 50%)
    drop otherwise (ambiguous middle 40%)

Input:  data/processed/papers_hci.parquet
Output: data/processed/papers_labeled.parquet   (adds pct, label columns, drops middle)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

IN = Path("data/processed/papers_hci.parquet")
OUT = Path("data/processed/papers_labeled.parquet")

HI = 0.90
LO = 0.50


def main():
    df = pd.read_parquet(IN)
    print(f"input rows: {len(df)}")

    # Percentile rank within year
    df["pct"] = df.groupby("year")["citation_count"].rank(pct=True, method="average")

    # Assign labels, drop the middle
    def to_label(p: float):
        if p >= HI:
            return 1
        if p <= LO:
            return 0
        return -1

    df["label"] = df["pct"].apply(to_label)
    before = len(df)
    df = df[df["label"] != -1].reset_index(drop=True)
    print(f"kept {len(df)} / {before} after dropping middle {LO}-{HI}")

    # Sanity check: positive rate by year should be ~ (1-HI)/(1-HI+LO) = 0.10/0.60 = 0.167
    rates = df.groupby("year").agg(
        n=("label", "size"),
        pos=("label", "sum"),
    )
    rates["pos_rate"] = rates["pos"] / rates["n"]
    print("\nper-year label distribution:")
    print(rates.to_string())
    print(f"\noverall: n={len(df)}  pos={df.label.sum()}  neg={(df.label==0).sum()}  pos_rate={df.label.mean():.3f}")

    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
