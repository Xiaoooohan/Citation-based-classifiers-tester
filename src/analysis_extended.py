"""Extended analysis: bootstrap CIs, permutation test, multi-probe triangulation,
PR/ROC curves, calibration, per-year corpus stats, top-feature bar chart, drift plot."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import (
    precision_recall_curve, roc_curve, average_precision_score,
    roc_auc_score, f1_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve

FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(0)
B = 2000  # bootstrap reps

LABELED = pd.read_parquet("data/processed/papers_labeled.parquet")
HCI = pd.read_parquet("data/processed/papers_hci.parquet")
TFIDF = pd.read_parquet("data/processed/scores_tfidf.parquet")
EMBED = pd.read_parquet("data/processed/scores_embed.parquet")
NOV_NG = pd.read_parquet("data/processed/novelty_scores.parquet")
NOV_EM = pd.read_parquet("data/processed/novelty_embed.parquet")
DRIFT = pd.read_parquet("data/processed/temporal_drift.parquet")


def boot_ci(stat_fn, *arrays, n=B, q=(2.5, 97.5)):
    arrs = [np.asarray(a) for a in arrays]
    N = len(arrs[0])
    out = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, N, N)
        out[i] = stat_fn(*[a[idx] for a in arrs])
    return float(np.percentile(out, q[0])), float(np.percentile(out, q[1]))


def perm_p_spearman(x, y, n=B):
    obs = spearmanr(x, y).correlation
    x = np.asarray(x); y = np.asarray(y)
    cnt = 0
    for _ in range(n):
        yp = RNG.permutation(y)
        if abs(spearmanr(x, yp).correlation) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n + 1)


def merge_test(scores):
    m = scores.merge(NOV_NG[["paper_id", "novelty_score"]], on="paper_id", how="left")
    m = m.merge(NOV_EM[["paper_id", "semantic_novelty"]], on="paper_id", how="left")
    m = m.merge(LABELED[["paper_id", "title"]], on="paper_id", how="left")
    return m.dropna(subset=["score", "novelty_score", "semantic_novelty"]).reset_index(drop=True)


# ---------- 1. Bootstrap CIs + permutation tests ----------
def stats_table():
    rows = []
    for name, df in [("tfidf", merge_test(TFIDF)), ("embed", merge_test(EMBED))]:
        y = df.label.values; s = df.score.values; pred = df.pred.values
        f1 = f1_score(y, pred, zero_division=0)
        pr = average_precision_score(y, s)
        roc = roc_auc_score(y, s)
        f1_ci = boot_ci(lambda yy, pp: f1_score(yy, pp, zero_division=0), y, pred)
        pr_ci = boot_ci(lambda yy, ss: average_precision_score(yy, ss) if yy.sum() else np.nan, y, s)
        roc_ci = boot_ci(lambda yy, ss: roc_auc_score(yy, ss) if 0 < yy.sum() < len(yy) else np.nan, y, s)
        for probe, col in [("ngram", "novelty_score"), ("semantic", "semantic_novelty")]:
            rho, p = perm_p_spearman(s, df[col].values, n=B)
            rho_ci = boot_ci(lambda a, b: spearmanr(a, b).correlation, s, df[col].values)
            rows.append({"model": name, "probe": probe, "rho": rho, "rho_lo": rho_ci[0], "rho_hi": rho_ci[1], "perm_p": p,
                          "f1": f1, "f1_lo": f1_ci[0], "f1_hi": f1_ci[1],
                          "pr_auc": pr, "pr_lo": pr_ci[0], "pr_hi": pr_ci[1],
                          "roc": roc, "roc_lo": roc_ci[0], "roc_hi": roc_ci[1]})
    out = pd.DataFrame(rows)
    out_r = out.round(3)
    cols = list(out_r.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in out_r.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    Path("reports/stats.md").write_text("# Bootstrap CIs and permutation tests\n\n" + "\n".join(md))
    print(out.round(3).to_string(index=False))
    return out


# ---------- 2. Multi-probe agreement scatter ----------
def probe_agreement():
    df = NOV_NG.merge(NOV_EM[["paper_id", "semantic_novelty"]], on="paper_id", how="inner")
    rho, p = spearmanr(df.novelty_score, df.semantic_novelty)
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    ax.scatter(df.novelty_score, df.semantic_novelty, c=df.label, cmap="coolwarm", alpha=0.5, s=14)
    ax.set_xlabel("n-gram first-year novelty")
    ax.set_ylabel("semantic novelty (cosine dist to prior years)")
    ax.set_title(f"Two novelty probes: Spearman rho = {rho:+.3f} (p={p:.2g})")
    fig.tight_layout(); fig.savefig(FIG / "probe_agreement.png"); plt.close(fig)
    print(f"probe agreement rho = {rho:+.3f}")


# ---------- 3. PR + ROC overlays ----------
def pr_roc_overlay():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    for name, df in [("tfidf", merge_test(TFIDF)), ("embed", merge_test(EMBED))]:
        y = df.label.values; s = df.score.values
        p, r, _ = precision_recall_curve(y, s)
        axes[0].plot(r, p, label=f"{name} (AP={average_precision_score(y,s):.3f})")
        fpr, tpr, _ = roc_curve(y, s)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y,s):.3f})")
    axes[0].axhline(y.mean(), ls=":", c="gray", label=f"prior={y.mean():.3f}")
    axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].set_title("PR curve (test)"); axes[0].legend()
    axes[1].plot([0,1],[0,1], ls=":", c="gray")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].set_title("ROC curve (test)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(FIG / "pr_roc.png"); plt.close(fig)
    print("wrote pr_roc.png")


# ---------- 4. Calibration curves ----------
def calibration():
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    for name, df in [("tfidf", merge_test(TFIDF)), ("embed", merge_test(EMBED))]:
        y = df.label.values; s = df.score.values
        frac_pos, mean_pred = calibration_curve(y, s, n_bins=10, strategy="quantile")
        b = brier_score_loss(y, s)
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier={b:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", c="gray", label="perfect")
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("empirical positive rate")
    ax.set_title("Calibration (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "calibration.png"); plt.close(fig)
    print("wrote calibration.png")


# ---------- 5. Per-year corpus stats ----------
def per_year_stats():
    g = HCI.groupby("year").size().rename("n_papers")
    lab = LABELED.groupby("year").label.mean().rename("pos_rate_in_labeled")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].bar(g.index, g.values, color="#4477aa"); axes[0].set_title("HCI corpus size per year"); axes[0].set_xlabel("year"); axes[0].set_ylabel("# papers")
    axes[1].plot(lab.index, lab.values, marker="o", color="#cc3311"); axes[1].set_ylim(0, 0.3); axes[1].set_title("Labelled positive-rate per year (sanity)"); axes[1].set_xlabel("year"); axes[1].set_ylabel("pos rate")
    fig.tight_layout(); fig.savefig(FIG / "per_year_stats.png"); plt.close(fig)

    # vocabulary growth: cumulative unique unigrams
    import re
    STOP = set("a an the of in on for to with from by and or as is are was were be been this that".split())
    seen = set(); rows = []
    for y in sorted(HCI.year.unique()):
        sub = HCI[HCI.year == y]
        for _, r in sub.iterrows():
            for tok in re.findall(r"[a-z]{4,}", ((r.title or "") + " " + (r.abstract or "")).lower()):
                if tok in STOP: continue
                seen.add(tok)
        rows.append((y, len(seen)))
    yrs, cum = zip(*rows)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(yrs, cum, marker="o", color="#117733")
    ax.set_xlabel("year"); ax.set_ylabel("cumulative unique unigrams"); ax.set_title("HCI corpus vocabulary growth")
    fig.tight_layout(); fig.savefig(FIG / "vocab_growth.png"); plt.close(fig)
    print("wrote per_year_stats.png, vocab_growth.png")


# ---------- 6. Top features bar chart ----------
def top_features_bar():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    train = LABELED[LABELED.year <= 2019]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True)
    X = vec.fit_transform((train.title.fillna("") + " " + train.abstract.fillna("")).tolist())
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0).fit(X, train.label.values)
    feats = np.array(vec.get_feature_names_out()); coefs = clf.coef_[0]
    pos_idx = np.argsort(coefs)[-15:]; neg_idx = np.argsort(coefs)[:15]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    axes[0].barh(np.arange(15), coefs[pos_idx], color="#cc3311"); axes[0].set_yticks(range(15)); axes[0].set_yticklabels(feats[pos_idx]); axes[0].set_title("Top + (predicts high-impact)")
    axes[1].barh(np.arange(15), coefs[neg_idx], color="#4477aa"); axes[1].set_yticks(range(15)); axes[1].set_yticklabels(feats[neg_idx]); axes[1].set_title("Top - (predicts low-impact)")
    fig.tight_layout(); fig.savefig(FIG / "top_features.png"); plt.close(fig)
    print("wrote top_features.png")


# ---------- 7. Temporal drift plot ----------
def drift_plot():
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
    g = DRIFT.groupby("gap")["pr_auc"].agg(["mean", "std", "count"]).reset_index()
    ax.errorbar(g["gap"], g["mean"], yerr=g["std"], marker="o", capsize=4, color="#cc3311", label="mean ± SD across cutoffs")
    for _, r in DRIFT.iterrows():
        ax.scatter(r["gap"] + RNG.uniform(-0.1, 0.1), r["pr_auc"], alpha=0.3, color="#4477aa", s=20)
    ax.set_xlabel("gap (years between train cutoff and test year)")
    ax.set_ylabel("PR-AUC on held-out year")
    ax.set_xticks([1, 2, 3])
    ax.set_title("Temporal drift: classifier decays as gap grows")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "temporal_drift.png"); plt.close(fig)
    print("wrote temporal_drift.png")


def main():
    print("\n=== bootstrap + permutation tests ===")
    stats_table()
    print("\n=== probe agreement ==="); probe_agreement()
    print("\n=== PR/ROC ===");          pr_roc_overlay()
    print("\n=== calibration ===");     calibration()
    print("\n=== per-year stats ===");  per_year_stats()
    print("\n=== top-features bar ==="); top_features_bar()
    print("\n=== temporal drift ===");  drift_plot()


if __name__ == "__main__":
    main()
