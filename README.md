# Temporal NLP Pipeline — Impact vs. Novelty in HCI Papers

**STATS 507 Final Project — Xiaohan Ye, April 2026.**

Trains TF-IDF and Sentence-BERT (MiniLM) classifiers to predict highly-cited
HCI papers, then **probes what those classifiers actually measure** with two
independent corpus-novelty signals and a temporal-drift experiment.

**Headline finding.** Citation-based impact and corpus-novelty are nearly
orthogonal axes. The two novelty probes themselves disagree, indicating
"novelty" is not a single quantity. PR-AUC decays ~10% per two years of
forward gap. See [reports/report_ieee.tex](reports/report_ieee.tex) (IEEE 2-column, ≤5 pages, with references).

**Try the live screener:** open [reports/screener.html](reports/screener.html) in any browser. No server required — 270 HCI test papers with all four scores side-by-side, sortable and filterable by quadrant, with the impact-vs-novelty caveat surfaced inline.

## Proposal-to-final pivot (read this first)

The proposal targeted a "concept-introducing paper detector" using a hybrid
weak-/gold-label scheme. After the GSI flagged that weak novelty labels are
unreliable, two paths were open: (a) build a fragile silver-label pipeline
under a hard 3-week deadline, or (b) reframe to a sharper, well-posed
question. We chose (b): **train standard impact classifiers on a citation-derived label, then probe what they actually capture along independent novelty axes.** The pivot is documented in §I of the report. The labelling and modelling methodology described in the proposal (TF-IDF + embeddings, time-based splits, calibration, evaluation) is preserved exactly; only the *labelling target* changed from a noisy novelty signal to a defensible citation-percentile signal.

## Data

- Source: OpenAlex (HCI concept `C107457646`), 2010–2023.
- 22,297 raw → 3,935 HCI-filtered → 2,365 labelled by within-year citation percentile (top 10% positive, bottom 50% negative; middle dropped).
- Strict time-based splits: train ≤ 2019 (n=1,833), val 2020–21 (n=262), test 2022–23 (n=270).

## Pipeline

| step | script | purpose |
|---|---|---|
| 1 | `src/pull_openalex.py`  | pull HCI papers from OpenAlex |
| 2 | `src/filter_hci.py`     | top-5-concept HCI filter; drop surveys |
| 3 | `src/build_labels.py`   | within-year citation-percentile labels |
| 4 | `src/splits.py`         | time-based splits + evaluation harness |
| 5 | `src/model_tfidf.py`    | TF-IDF + LR classifier |
| 6 | `src/model_embed.py`    | MiniLM + LR classifier |
| 7 | `src/novelty_probe.py`  | lexical novelty (n-gram first-year-seen) |
| 8 | `src/novelty_embed.py`  | semantic novelty (cosine to prior centroid) |
| 9 | `src/temporal_drift.py` | train-cutoff × test-gap drift sweep |
| 10 | `src/analysis.py`         | core impact-vs-novelty plots |
| 11 | `src/analysis_extended.py`| bootstrap CIs, perm tests, calibration, drift, top-features |

## Reproduce

```bash
pip install -r requirements.txt
PYTHONPATH=src python3 src/pull_openalex.py        # ~20 min, do not re-run
PYTHONPATH=src python3 src/filter_hci.py
PYTHONPATH=src python3 src/build_labels.py
PYTHONPATH=src python3 src/model_tfidf.py
PYTHONPATH=src python3 src/model_embed.py          # downloads ~90 MB MiniLM model
PYTHONPATH=src python3 src/novelty_probe.py
PYTHONPATH=src python3 src/novelty_embed.py
PYTHONPATH=src python3 src/temporal_drift.py
PYTHONPATH=src python3 src/analysis.py
PYTHONPATH=src python3 src/analysis_extended.py
```

## Results summary (test set, 95% bootstrap CIs)

| model | F1 | PR-AUC | ρ vs n-gram nov. | ρ vs semantic nov. |
|---|---|---|---|---|
| random (prior) | 0.139 | 0.153 | — | — |
| TF-IDF + LR    | 0.170 [0.076, 0.273] | 0.181 [0.134, 0.253] | +0.130 [0.014, 0.251] | +0.269 [0.163, 0.374] |
| MiniLM + LR    | 0.213 [0.140, 0.292] | 0.172 [0.119, 0.266] | −0.013 [−0.133, 0.105] | +0.101 [−0.016, 0.231] |

Inter-probe agreement: ρ = −0.047 (the two novelty probes disagree).

## File layout

```
src/                model + analysis code
data/raw/           OpenAlex pull (gitignored)
data/processed/     filtered, labelled, scored parquet (gitignored)
reports/
  report_ieee.tex   final IEEE-format report (≤5 pages, with references)
  figures/          11 PNG figures referenced by the report
  examples.md       qualitative high/low-impact + novel/non-novel examples
  features.md       full top TF-IDF coefficient lists
  quadrants.md      impact × novelty quadrant breakdowns
  stats.md          bootstrap CIs + permutation test summary
```

## Limitations (excerpt; full discussion in report §VI)

Top-by-citation ingest creates an above-average negative class; both novelty
probes are proxies; survey filter is regex-based; both classifiers are
over-confident in their highest-probability bin.
