# What Do Citation-Based Classifiers Actually Measure?
## A Probe of Impact vs. Novelty in HCI Papers

**Author:** Xiaohan Ye  
**Course:** STATS 507 — Final Project  
**Date:** 2026-04-21

---

## Abstract

Citation-based classifiers are commonly proposed as proxies for identifying important or novel research. We train standard TF-IDF and Sentence-BERT classifiers on a 2,365-paper Human–Computer Interaction (HCI) corpus drawn from OpenAlex (2010–2023), labelling papers as "high-impact" by within-year citation percentile. We evaluate the models not only on predictive performance but on what their predictions actually measure, by comparing their scores against an independent novelty probe based on first-year-of-appearance of noun-phrase terms in the corpus. We find that while both classifiers beat random baselines, their scores correlate only weakly (Spearman ρ ≤ 0.13) with the novelty signal. The TF-IDF model's most-positive coefficients reveal it is learning popular subfield vocabulary and writing-style cues — not novelty. We argue this is a critical caveat for anyone deploying citation-trained classifiers as research-trend or novelty detectors.

## 1. Introduction

The original goal of this project was to build a temporal NLP pipeline to detect "concept-introducing" papers — those that introduce genuinely new ideas. We quickly discovered that the data reality does not support this directly: there is no widely-available labelled corpus of "novel concept" introductions, and weakly-supervised novelty heuristics are noisy. After consultation with course staff, we considered the common workaround of using citation counts as a proxy. This pivots the question from *what makes a paper novel?* to a more answerable empirical one: **if we train a standard text classifier to predict highly-cited papers, what does the model actually learn — and is it a defensible proxy for novelty?**

This report's contribution is the probe answering that question. Our finding is negative-but-useful: the impact classifier is largely orthogonal to a corpus-based novelty signal, and inspection of its features confirms it is learning topical popularity rather than novelty.

## 2. Data

We pulled 22,297 HCI-related papers from OpenAlex via the `pyalex` client, restricting to the HCI concept (`C107457646`) and years 2010–2023, capping at 1,600 papers/year ranked by citation count to bound the size of the pull. We then filtered to papers whose **top-5 OpenAlex concepts** explicitly include "Human–computer interaction" (note: en-dash U+2013, not a hyphen — a real source of bugs in OpenAlex tooling), yielding 3,935 papers. Survey/review/meta-analysis papers were excluded by regex on the title and abstract prefix; surveys are a known citation-count artifact that would otherwise dominate the positive class.

**Labelling.** For each paper, we computed its citation-count percentile *within its publication year* to remove age bias (2010 papers have had 13 years to accrue cites, 2023 papers <1). Papers in the top 10% of their year were labelled positive (high-impact); bottom 50% were labelled negative; the middle 40% was discarded to give the classifier a cleaner separation. The resulting labelled set has 2,365 papers (399 positive, 1,966 negative; positive rate ≈ 16.9% in every split).

**Splits.** We use a strictly time-based split: train ≤ 2019 (n=1,833), val 2020–2021 (n=262), test 2022–2023 (n=270). No paper appears in more than one split.

## 3. Methods

### 3.1 Impact classifiers

Two models, both trained on `title + " " + abstract`:
- **TF-IDF + Logistic Regression**: `TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=50k, sublinear_tf=True)` → `LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)`.
- **Sentence-BERT (all-MiniLM-L6-v2) + Logistic Regression**: 384-dim embeddings, `StandardScaler`, then the same logistic regression.

Decision thresholds were tuned on the validation set by sweeping in 0.05 steps from 0.10 to 0.90 to maximize F1, then fixed for test evaluation.

### 3.2 Novelty probe (independent of the classifier)

To test what the classifier learns, we need a *separate*, deterministic novelty signal. We extract unigram, bigram, and trigram phrases (lowercased, stop-listed) from each paper's title+abstract. Across the full HCI corpus (3,935 papers), we record the *first year* each phrase appears. A paper's **novelty score** is the fraction of its phrases (corpus-frequency ≥ 2) whose first appearance in the corpus is the paper's own publication year.

This probe is intentionally simple. It uses only text and year — no labels, no citations, no classifier predictions — so any correlation between it and the impact classifier is genuine information about what the classifier captures.

## 4. Results

### 4.1 Baselines and model performance (test set)

| model            | n   | precision | recall | F1    | ROC-AUC | PR-AUC |
|------------------|-----|-----------|--------|-------|---------|--------|
| majority class   | 270 | 0.000     | 0.000  | 0.000 | 0.500   | 0.167  |
| random (prior)   | 270 | 0.146     | 0.133  | 0.139 | 0.498   | 0.153  |
| **TF-IDF + LR**  | 270 | 0.148     | 0.200  | 0.170 | 0.570   | 0.181  |
| **Embed + LR**   | 270 | 0.138     | 0.467  | 0.213 | 0.471   | 0.172  |

Both classifiers clear the random baseline by a small margin on F1; the TF-IDF model's PR-AUC (0.181) modestly beats the prior (0.153). On train, TF-IDF reaches F1=0.91 (PR-AUC=0.99), confirming the model can fit the impact signal in the training years but generalizes weakly forward — a sign that "high-impact" vocabulary drifts year to year.

### 4.2 Impact vs. novelty — the central probe

We merged each test paper's classifier score with its novelty score and computed Spearman rank correlation:

| model            | Spearman ρ (impact-score vs. novelty-score) |
|------------------|-----|
| **TF-IDF + LR**  | +0.130 |
| **Embed + LR**   | -0.013 |

Both correlations are weak; the embedding model's score is essentially uncorrelated with novelty. The corpus-level Spearman of novelty against raw citation counts is +0.039 (p=0.06) and against the high-impact label is +0.057 (p=0.005). **A paper being highly cited tells you almost nothing about whether it introduces phrases new to the corpus.**

See `reports/figures/scatter_tfidf.png` and `scatter_embed.png` for the joint distributions.

### 4.3 Quadrant analysis

Splitting the test set at the median impact-score and median novelty-score:

**TF-IDF** (medians: impact=0.424, novelty=0.022)

| quadrant | n | true positive rate |
|---|---|---|
| impact-hi / novelty-hi | 75 | 0.320 |
| impact-hi / novelty-lo | 60 | 0.100 |
| impact-lo / novelty-hi | 60 | 0.167 |
| impact-lo / novelty-lo | 75 | 0.067 |

**Embed** (medians: impact=0.252, novelty=0.022)

| quadrant | n | true positive rate |
|---|---|---|
| impact-hi / novelty-hi | 67 | 0.239 |
| impact-hi / novelty-lo | 68 | 0.059 |
| impact-lo / novelty-hi | 68 | 0.265 |
| impact-lo / novelty-lo | 67 | 0.104 |

For the embedding model, the *novelty* axis is at least as predictive of true high-impact label as the classifier's own impact axis (0.265 vs 0.239 in the high-novelty quadrants). This is striking: a deterministic, label-free corpus probe matches a 384-dim BERT classifier on the very task the classifier was trained for.

### 4.4 What the TF-IDF model learned

Top positive coefficients include `user experience`, `brain`, `taxonomy`, `mobile phones`, `sensors`, `network`, `applications` — these are popular HCI subfield markers, not novelty cues. Top negative coefficients include surface phrases (`presents`, `this paper`, `study`, `paper`) and topic words (`children`, `students`, `navigation`, `screen`). The model has learned *what topics tend to get cited and what writing style appears in less-cited papers*, which is exactly the bias the probe was designed to expose. Full lists in `reports/features.md`.

### 4.5 Qualitative examples

`reports/examples.md` lists five "popular-but-not-novel" papers (high impact-score, low novelty-score) and five "novel-but-missed" papers (low impact-score, high novelty-score) for each model. The popular-but-not-novel set is dominated by literature reviews, survey-adjacent overviews, and applications of established techniques. The novel-but-missed set contains methodology papers introducing specific new constructs that the citation classifier underrates.

## 5. Discussion and limitations

1. **Selection bias from ingest.** OpenAlex was queried for the top 1,600 papers per year by citation count. The "negative" class is therefore the bottom-half of an already above-average pool, not arbitrary papers. This compresses the impact signal and likely lowers the achievable classification ceiling — but it does not affect the impact-vs-novelty probe, which is the report's main claim.
2. **Novelty probe is a proxy.** First-year-of-appearance for n-grams measures lexical novelty, not conceptual novelty. A paper introducing a new idea using only existing vocabulary will be missed; a paper that simply uses a rare typo will be over-rated (we mitigate this by requiring corpus frequency ≥ 2). This is acceptable because we use the probe as a *contrast*, not as a ground-truth target.
3. **Survey exclusion is regex-based** and may miss surveys whose title doesn't begin with "a survey of …".
4. **Impact ≠ novelty is a finding, not a weakness.** The thesis of the report is precisely that citation-based labels capture topical popularity and writing style much more than they capture novelty.
5. **Forward generalisation is hard.** TF-IDF train F1=0.91 vs test F1=0.17 reveals heavy temporal distribution shift in "what gets cited"; this is a separate finding worth flagging for anyone planning to deploy such a model.

## 6. Conclusion

Citation-trained text classifiers are not drop-in novelty detectors. On a careful HCI test set, both a strong classical baseline and a modern embedding model produce scores that are nearly uncorrelated with an independent corpus-novelty signal, and inspection of feature weights confirms the classifier is learning popular topics and surface style. Such classifiers may still be useful as a first-pass impact filter, but should not be marketed as novelty detectors and should be paired with term-based analysis or human evaluation when novelty is the intended target.

## 7. Reproducibility

All code is in this repository under `src/`. The full pipeline is reproducible via the commands listed in the README. Random seeds are fixed where applicable; the OpenAlex pull is the one stochastic step (the API may have minor day-to-day jitter). Data artefacts in `data/processed/` are committed-by-default-excluded; they can be regenerated end-to-end in roughly 15 minutes.
