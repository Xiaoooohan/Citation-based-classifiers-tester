"""
Pull Human-Computer Interaction papers from OpenAlex, 2010-2023.

Output: data/raw/papers.parquet with columns:
    paper_id, title, abstract, year, venue, citation_count, concepts

Uses the 'Human-computer interaction' concept (OpenAlex concept id C107457646).
Abstracts come back as inverted-index dicts; we reconstruct plain text.

Run:
    python src/pull_openalex.py --per-year-cap 2000 --out data/raw/papers.parquet
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from pyalex import Works, config
from tqdm import tqdm

# Polite pool — faster and more reliable. Set your email.
config.email = "yexiaohan@umich.edu"

HCI_CONCEPT_ID = "C107457646"  # Human-computer interaction


def invert_abstract(inv_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}. Reconstruct."""
    if not inv_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def pull_year(year: int, cap: int) -> list[dict]:
    """Pull up to `cap` HCI papers for a single year."""
    q = (
        Works()
        .filter(concepts={"id": HCI_CONCEPT_ID})
        .filter(publication_year=year)
        .filter(has_abstract=True)
        .filter(language="en")
        .sort(cited_by_count="desc")  # prioritize papers with citation signal
    )
    rows: list[dict] = []
    try:
        for rec in q.paginate(per_page=200, n_max=cap):
            for w in rec:
                rows.append(
                    {
                        "paper_id": w.get("id"),
                        "title": w.get("title") or "",
                        "abstract": invert_abstract(w.get("abstract_inverted_index")),
                        "year": w.get("publication_year"),
                        "venue": (w.get("host_venue") or {}).get("display_name") or "",
                        "citation_count": w.get("cited_by_count", 0),
                        "concepts": [
                            c.get("display_name") for c in (w.get("concepts") or [])
                        ],
                    }
                )
    except Exception as e:
        print(f"[year={year}] ERROR: {e}", file=sys.stderr, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year", type=int, default=2023)
    ap.add_argument("--per-year-cap", type=int, default=2000)
    ap.add_argument("--out", type=str, default="data/raw/papers.parquet")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    t0 = time.time()
    for year in tqdm(range(args.start_year, args.end_year + 1), desc="years"):
        rows = pull_year(year, args.per_year_cap)
        print(f"  year={year}  pulled={len(rows)}", flush=True)
        all_rows.extend(rows)
        # checkpoint every year in case of crash
        pd.DataFrame(all_rows).to_parquet(out, index=False)

    df = pd.DataFrame(all_rows)
    # Drop empties / dedupe
    df = df[df["abstract"].str.len() > 50].drop_duplicates(subset=["paper_id"])
    df.to_parquet(out, index=False)

    elapsed = time.time() - t0
    print(f"\nDONE. rows={len(df)}  elapsed={elapsed/60:.1f} min  -> {out}")


if __name__ == "__main__":
    main()
