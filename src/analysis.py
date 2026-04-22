"""Impact-vs-Novelty analysis: merge classifier scores with novelty probe,
produce scatter, quadrant table, qualitative examples, confusion matrices."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
REPORTS = Path("reports")

NOVELTY = pd.read_parquet("data/processed/novelty_scores.parquet")
TFIDF = pd.read_parquet("data/processed/scores_tfidf.parquet")
EMBED = pd.read_parquet("data/processed/scores_embed.parquet")
LABELED = pd.read_parquet("data/processed/papers_labeled.parquet")[["paper_id", "title"]]


def merge(scores: pd.DataFrame) -> pd.DataFrame:
    m = scores.merge(NOVELTY[["paper_id", "novelty_score"]], on="paper_id", how="left")
    m = m.merge(LABELED, on="paper_id", how="left")
    return m


def scatter(df: pd.DataFrame, name: str):
    rho, p = spearmanr(df.score, df.novelty_score)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for lab, color, marker in [(0, "#888", "o"), (1, "#d62728", "x")]:
        sub = df[df.label == lab]
        ax.scatter(sub.score, sub.novelty_score, c=color, marker=marker, s=24, alpha=0.6,
                   label=f"label={lab} (n={len(sub)})")
    ax.set_xlabel(f"{name} classifier score (impact)")
    ax.set_ylabel("novelty score (corpus-first-year fraction)")
    ax.set_title(f"{name}: impact vs. novelty   Spearman ρ = {rho:+.3f} (p={p:.2g})")
    ax.legend()
    fig.tight_layout()
    out = FIG / f"scatter_{name}.png"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}  ρ={rho:+.3f}")
    return rho, p


def quadrants(df: pd.DataFrame, name: str) -> str:
    ms = df.score.median()
    mn = df.novelty_score.median()
    rows = []
    for s_hi in [True, False]:
        for n_hi in [True, False]:
            sub = df[((df.score >= ms) == s_hi) & ((df.novelty_score >= mn) == n_hi)]
            label = f"impact_{'hi' if s_hi else 'lo'} / novelty_{'hi' if n_hi else 'lo'}"
            rows.append((label, len(sub), float(sub.label.mean()) if len(sub) else float("nan")))
    md = [f"### {name} — quadrant breakdown (medians: impact={ms:.3f}, novelty={mn:.3f})\n",
          "| quadrant | n | pos_rate (true label=1) |", "|---|---|---|"]
    md += [f"| {r[0]} | {r[1]} | {r[2]:.3f} |" for r in rows]
    return "\n".join(md) + "\n"


def confusion_plot(df: pd.DataFrame, name: str):
    cm = confusion_matrix(df.label, df.pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title(f"{name} confusion (test)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = FIG / f"cm_{name}.png"
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}")


def examples(df: pd.DataFrame, name: str) -> str:
    ms = df.score.median(); mn = df.novelty_score.median()
    pop_not_novel = df[(df.score >= ms) & (df.novelty_score < mn)].sort_values("score", ascending=False).head(5)
    novel_missed = df[(df.score < ms) & (df.novelty_score >= mn)].sort_values("novelty_score", ascending=False).head(5)
    out = [f"## {name} — qualitative examples\n",
           "### Popular-but-not-novel (high impact-score, low novelty-score)\n",
           "| score | novelty | label | title |", "|---|---|---|---|"]
    for _, r in pop_not_novel.iterrows():
        out.append(f"| {r.score:.3f} | {r.novelty_score:.3f} | {int(r.label)} | {str(r.title)[:140]} |")
    out += ["\n### Novel-but-missed (low impact-score, high novelty-score)\n",
            "| score | novelty | label | title |", "|---|---|---|---|"]
    for _, r in novel_missed.iterrows():
        out.append(f"| {r.score:.3f} | {r.novelty_score:.3f} | {int(r.label)} | {str(r.title)[:140]} |")
    return "\n".join(out) + "\n"


def main():
    tfidf = merge(TFIDF); embed = merge(EMBED)
    rho_t, _ = scatter(tfidf, "tfidf")
    rho_e, _ = scatter(embed, "embed")
    confusion_plot(tfidf, "tfidf")
    confusion_plot(embed, "embed")

    quads = quadrants(tfidf, "tfidf") + "\n" + quadrants(embed, "embed")
    (REPORTS / "quadrants.md").write_text("# Impact vs. Novelty quadrant analysis (test set)\n\n" + quads)

    ex = examples(tfidf, "tfidf") + "\n" + examples(embed, "embed")
    (REPORTS / "examples.md").write_text("# Qualitative examples (test set)\n\n" + ex)

    summary = pd.DataFrame({
        "model": ["tfidf", "embed"],
        "spearman_score_vs_novelty": [rho_t, rho_e],
    })
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
