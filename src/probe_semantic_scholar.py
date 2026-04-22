"""External validation: pull Semantic Scholar's influentialCitationCount for
the 270 test papers and probe it against our two corpus-novelty scores.

If a real, deployed citation-derived signal also fails the novelty probe,
the report's claim about citation-trained tools is empirically defended.
"""
from __future__ import annotations
import re, time, json, urllib.parse
import requests
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

LABELED = Path("data/processed/papers_labeled.parquet")
NOV_NG = Path("data/processed/novelty_scores.parquet")
NOV_EM = Path("data/processed/novelty_embed.parquet")
OUT = Path("data/processed/ss_external.parquet")
SLEEP = 4.0  # public SS API rate-limits aggressively without an API key

API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,influentialCitationCount,citationCount,year"
HEADERS = {"User-Agent": "stats507-final-project/1.0"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())[:80]


def lookup(title: str, year: int):
    q = urllib.parse.quote(title[:200])
    url = f"{API}?query={q}&limit=3&fields={FIELDS}"
    backoff = 5
    for attempt in range(6):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
        except Exception as e:
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue
        if r.status_code == 200:
            try:
                data = r.json(); break
            except Exception:
                return None, "bad_json"
        if r.status_code in (429, 503):
            time.sleep(backoff); backoff = min(backoff * 2, 60); continue
        return None, f"http_{r.status_code}"
    else:
        return None, "rate_limited"
    cands = data.get("data") or []
    target = norm(title)
    for c in cands:
        if norm(c.get("title", "")) == target:
            return c, "exact"
    # fallback: first hit if titles share >= 60 chars prefix
    if cands:
        c = cands[0]
        if target and norm(c.get("title", ""))[:60] == target[:60]:
            return c, "fuzzy"
    return None, "no_match"


def main():
    df = pd.read_parquet(LABELED)
    test = df[df.year.isin([2022, 2023])].reset_index(drop=True)
    print(f"querying SS for {len(test)} test papers...")
    rows = []
    for i, r in test.iterrows():
        c, status = lookup(r.title, int(r.year))
        rows.append({
            "paper_id": r.paper_id,
            "title": r.title,
            "year": int(r.year),
            "label": int(r.label),
            "openalex_citations": int(r.citation_count),
            "ss_paper_id": (c or {}).get("paperId"),
            "ss_title": (c or {}).get("title"),
            "ss_inf_cites": (c or {}).get("influentialCitationCount"),
            "ss_cites": (c or {}).get("citationCount"),
            "match": status,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(test)}  last status={status}")
        time.sleep(SLEEP)
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  matches: {(out.match!='no_match').sum()}/{len(out)}")
    print(out.match.value_counts())

    matched = out[out.ss_inf_cites.notna()].copy()
    matched["ss_inf_cites"] = matched["ss_inf_cites"].astype(float)
    matched["ss_cites"] = matched["ss_cites"].astype(float)
    print(f"\nusable rows: {len(matched)}")
    print(matched[["openalex_citations", "ss_cites", "ss_inf_cites"]].describe())

    # merge novelty
    nng = pd.read_parquet(NOV_NG)[["paper_id", "novelty_score"]]
    nem = pd.read_parquet(NOV_EM)[["paper_id", "semantic_novelty"]]
    m = matched.merge(nng, on="paper_id", how="left").merge(nem, on="paper_id", how="left")
    m = m.dropna(subset=["novelty_score", "semantic_novelty", "ss_inf_cites"])
    print(f"rows with novelty + SS inf-cites: {len(m)}")

    rng = np.random.default_rng(0)
    def boot(x, y, n=2000):
        x = np.asarray(x); y = np.asarray(y); N = len(x)
        out = np.empty(n)
        for i in range(n):
            idx = rng.integers(0, N, N)
            out[i] = spearmanr(x[idx], y[idx]).correlation
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    def perm(x, y, n=2000):
        obs = spearmanr(x, y).correlation
        x = np.asarray(x); y = np.asarray(y)
        cnt = 0
        for _ in range(n):
            if abs(spearmanr(x, rng.permutation(y)).correlation) >= abs(obs):
                cnt += 1
        return obs, (cnt + 1) / (n + 1)

    print("\n=== external SS probe ===")
    for col in ["ss_inf_cites", "ss_cites"]:
        for novcol in ["novelty_score", "semantic_novelty"]:
            rho, p = perm(m[col], m[novcol])
            lo, hi = boot(m[col], m[novcol])
            print(f"  Spearman({col} , {novcol}) = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  perm-p={p:.3g}")

    # also: agreement between SS inf-cites and OpenAlex citations
    rho_a, p_a = spearmanr(m.ss_inf_cites, m.openalex_citations)
    print(f"\nsanity: Spearman(ss_inf_cites, openalex_citations) = {rho_a:+.3f} (p={p_a:.2g})")


if __name__ == "__main__":
    main()
