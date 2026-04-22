"""Render a self-contained HTML paper-screener over the test set, showing all
four scores side-by-side with a quadrant tag and an impact-vs-novelty warning.

Output: reports/screener.html (single file, no server, opens in any browser).
"""
from __future__ import annotations
import json, html
from pathlib import Path
import pandas as pd
import numpy as np

OUT = Path("reports/screener.html")

T = pd.read_parquet("data/processed/scores_tfidf.parquet").rename(
    columns={"score": "impact_tfidf", "pred": "pred_tfidf"})
E = pd.read_parquet("data/processed/scores_embed.parquet").rename(
    columns={"score": "impact_embed", "pred": "pred_embed"})[["paper_id", "impact_embed", "pred_embed"]]
NNG = pd.read_parquet("data/processed/novelty_scores.parquet")[["paper_id", "novelty_score"]].rename(
    columns={"novelty_score": "novelty_ngram"})
NEM = pd.read_parquet("data/processed/novelty_embed.parquet")[["paper_id", "semantic_novelty"]]
LAB = pd.read_parquet("data/processed/papers_labeled.parquet")[
    ["paper_id", "title", "abstract", "citation_count"]]

df = (T.merge(E, on="paper_id")
        .merge(NNG, on="paper_id", how="left")
        .merge(NEM, on="paper_id", how="left")
        .merge(LAB, on="paper_id", how="left"))


def quadrant(row, m_imp, m_nov):
    a = "hi" if row.impact_tfidf >= m_imp else "lo"
    b = "hi" if row.novelty_ngram >= m_nov else "lo"
    return f"impact-{a} / novelty-{b}"


m_imp = df.impact_tfidf.median(); m_nov = df.novelty_ngram.median()
df["quadrant"] = df.apply(lambda r: quadrant(r, m_imp, m_nov), axis=1)
df = df.sort_values("impact_tfidf", ascending=False).reset_index(drop=True)

QUAD_COLOR = {
    "impact-hi / novelty-hi": "#2ca02c",   # both high — best case
    "impact-hi / novelty-lo": "#d62728",   # popular but not novel — caution
    "impact-lo / novelty-hi": "#ff7f0e",   # under-recognised novel — opportunity
    "impact-lo / novelty-lo": "#7f7f7f",
}

# Build one row per paper as JSON for client-side filtering
records = []
for _, r in df.iterrows():
    records.append({
        "id": r.paper_id.split("/")[-1],
        "title": r.title or "",
        "abstract": (r.abstract or "")[:600],
        "year": int(r.year),
        "cites": int(r.citation_count),
        "label": int(r.label),
        "imp_t": round(float(r.impact_tfidf), 3),
        "imp_e": round(float(r.impact_embed), 3),
        "nov_n": round(float(r.novelty_ngram), 3) if pd.notna(r.novelty_ngram) else None,
        "nov_s": round(float(r.semantic_novelty), 3) if pd.notna(r.semantic_novelty) else None,
        "quad": r.quadrant,
        "color": QUAD_COLOR[r.quadrant],
    })

records_json = json.dumps(records)

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>HCI Paper Screener — Impact vs. Novelty (STATS 507 Final Project)</title>
<style>
 body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 0; color: #222; background: #fafafa; }}
 header {{ padding: 20px 28px; background: #fff; border-bottom: 1px solid #ddd; }}
 h1 {{ margin: 0 0 4px; font-size: 22px; }}
 .sub {{ color: #555; font-size: 14px; }}
 .warn {{ background: #fff8e1; border-left: 5px solid #f9a825; padding: 12px 16px; margin: 14px 28px;
         font-size: 14px; line-height: 1.45; }}
 .warn b {{ color: #b26a00; }}
 .controls {{ padding: 12px 28px; background: #fff; border-bottom: 1px solid #eee; display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
 .controls label {{ font-size: 13px; color: #444; }}
 .controls select, .controls input {{ font: inherit; padding: 4px 8px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }}
 th, td {{ padding: 8px 10px; border-bottom: 1px solid #eee; vertical-align: top; text-align: left; }}
 th {{ background: #f5f5f5; cursor: pointer; user-select: none; position: sticky; top: 0; z-index: 1; }}
 th.num, td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 .quad {{ display: inline-block; padding: 2px 8px; border-radius: 10px; color: white; font-size: 11px; }}
 .title-cell {{ max-width: 480px; }}
 .title-cell summary {{ cursor: pointer; font-weight: 500; }}
 .title-cell .abs {{ color: #666; margin-top: 4px; font-size: 12px; line-height: 1.4; }}
 footer {{ padding: 18px 28px; color: #666; font-size: 12px; }}
 .legend {{ display: inline-block; margin-right: 18px; }}
 .legend .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 5px; vertical-align: middle; margin-right: 5px; }}
</style></head>
<body>
<header>
  <h1>HCI Paper Screener — Impact vs. Novelty</h1>
  <div class="sub">270 HCI papers (test set, 2022–2023) scored along four independent axes. STATS 507 final project, X. Ye, April 2026.</div>
</header>

<div class="warn">
  <b>Impact ≠ novelty.</b> The two impact scores below predict citation-percentile labels; the two novelty scores measure (a) introduction of new vocabulary and (b) distance from the prior-year embedding cloud. In our data, classifier impact correlates only weakly with either novelty signal (Spearman ρ ∈ [−0.01, +0.27]), and the two novelty signals are themselves nearly orthogonal (ρ ≈ −0.05). <b>Treat the four columns as separate axes; do not collapse them into a single ranking.</b>
</div>

<div class="controls">
  <label>Filter quadrant:
    <select id="quadFilter">
      <option value="">all</option>
      <option>impact-hi / novelty-hi</option>
      <option>impact-hi / novelty-lo</option>
      <option>impact-lo / novelty-hi</option>
      <option>impact-lo / novelty-lo</option>
    </select>
  </label>
  <label>Search title: <input id="search" type="text" placeholder="keyword..."></label>
  <span style="margin-left:auto; color:#666; font-size:12px;">Click any column header to sort.</span>
</div>

<table id="tbl">
<thead><tr>
  <th data-key="title">Title / Abstract</th>
  <th data-key="year" class="num">Year</th>
  <th data-key="cites" class="num">Cites</th>
  <th data-key="label" class="num" title="True label: 1 = top-10% in its year, 0 = bottom-50%">Label</th>
  <th data-key="imp_t" class="num" title="TF-IDF + LR impact score">Impact (TF-IDF)</th>
  <th data-key="imp_e" class="num" title="MiniLM + LR impact score">Impact (MiniLM)</th>
  <th data-key="nov_n" class="num" title="Fraction of n-gram phrases first appearing in this paper's year">Novelty (n-gram)</th>
  <th data-key="nov_s" class="num" title="Cosine distance from the prior-year MiniLM centroid">Novelty (semantic)</th>
  <th data-key="quad">Quadrant</th>
</tr></thead>
<tbody></tbody>
</table>

<footer>
<div>
  <span class="legend"><span class="dot" style="background:#2ca02c"></span>impact-hi / novelty-hi — both: candidate for "real" influence</span>
  <span class="legend"><span class="dot" style="background:#d62728"></span>impact-hi / novelty-lo — popular but not novel: surveys, applications</span>
  <span class="legend"><span class="dot" style="background:#ff7f0e"></span>impact-lo / novelty-hi — under-recognised novel: methodology pieces</span>
  <span class="legend"><span class="dot" style="background:#7f7f7f"></span>impact-lo / novelty-lo — neither</span>
</div>
<div style="margin-top:8px">Quadrants are computed at the per-column median of the 270 test papers. Impact-axis uses TF-IDF; novelty-axis uses n-gram. See the <code>reports/report_ieee.tex</code> for the full methodology and confidence intervals.</div>
</footer>

<script>
const DATA = {records_json};
const tbody = document.querySelector('#tbl tbody');
let sortKey = 'imp_t', sortDir = -1;

function render() {{
  const q = document.getElementById('quadFilter').value;
  const s = document.getElementById('search').value.toLowerCase();
  const rows = DATA
    .filter(r => !q || r.quad === q)
    .filter(r => !s || r.title.toLowerCase().includes(s))
    .sort((a,b) => {{
      const va = a[sortKey], vb = b[sortKey];
      if (va == null) return 1; if (vb == null) return -1;
      return va < vb ? -sortDir : va > vb ? sortDir : 0;
    }});
  tbody.innerHTML = rows.map(r => `
    <tr>
      <td class="title-cell"><details><summary>${{escape(r.title)}}</summary><div class="abs">${{escape(r.abstract)}}…</div></details></td>
      <td class="num">${{r.year}}</td>
      <td class="num">${{r.cites}}</td>
      <td class="num">${{r.label}}</td>
      <td class="num">${{r.imp_t.toFixed(3)}}</td>
      <td class="num">${{r.imp_e.toFixed(3)}}</td>
      <td class="num">${{r.nov_n != null ? r.nov_n.toFixed(3) : '—'}}</td>
      <td class="num">${{r.nov_s != null ? r.nov_s.toFixed(3) : '—'}}</td>
      <td><span class="quad" style="background:${{r.color}}">${{r.quad}}</span></td>
    </tr>`).join('');
}}
function escape(s) {{ return (s||'').replace(/[&<>"']/g, c => ({{ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }}[c])); }}
document.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {{
  const k = th.dataset.key;
  if (sortKey === k) sortDir *= -1; else {{ sortKey = k; sortDir = -1; }}
  render();
}}));
document.getElementById('quadFilter').addEventListener('change', render);
document.getElementById('search').addEventListener('input', render);
render();
</script>
</body></html>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(HTML)
size_kb = OUT.stat().st_size / 1024
print(f"wrote {OUT}  ({len(records)} rows, {size_kb:.1f} KB)")
