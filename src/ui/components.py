"""
components.py — Reusable Streamlit UI helpers for Mini NotebookLM.

Public API
----------
render_grounding_badge(ragas_dict)
    Tiny inline badge shown right under every assistant response.
    Displays: Grounding 87% | Relevance 91% | Grade: Good
    Colour-coded green / amber / red based on faithfulness score.

render_ragas_panel(ragas_dict, key)
    Full RAGAS evaluation panel rendered inside a Streamlit expander.
    Sections:
      1. Grade banner        — weighted composite score + colour
      2. Metric gauges       — faithfulness / relevance / precision (+ recall, similarity if available)
      3. Chunk contribution  — per-chunk table with inline progress bars
      4. Ground-truth note   — shown when has_ground_truth is False
      5. Export              — st.download_button for JSON report

Changes from original
---------------------
* Chunk contribution table HTML is now cached with @st.cache_data so it is
  not rebuilt on every Streamlit re-run when data is unchanged.
* Metric gauge columns use st.metric() where possible, falling back to
  inline HTML only for the progress-bar rows (which Streamlit can't do natively).
* _score_colour / _score_emoji / _grade_colour promoted to module-level
  constants so they are importable by ragas_panel.py without circular imports.

All functions are SAFE to call with ragas_dict=None (they render nothing).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# ── Colour helpers (module-level so ragas_panel can import them) ──────────────

def score_colour(score: float) -> str:
    """Return a CSS hex colour for a 0–1 score."""
    if score >= 0.80: return "#22c55e"   # green-500
    if score >= 0.60: return "#f59e0b"   # amber-500
    return "#ef4444"                      # red-500

def score_emoji(score: float) -> str:
    if score >= 0.80: return "🟢"
    if score >= 0.60: return "🟡"
    return "🔴"

def grade_colour(grade: str) -> str:
    return {
        "Excellent": "#22c55e",
        "Good":      "#84cc16",
        "Fair":      "#f59e0b",
        "Poor":      "#ef4444",
    }.get(grade, "#94a3b8")

# Private aliases (backward compat with any internal callers)
_score_colour = score_colour
_score_emoji  = score_emoji
_grade_colour = grade_colour


# ── 1. Inline grounding badge ─────────────────────────────────────────────────

def render_grounding_badge(ragas_dict: Optional[Dict[str, Any]]) -> None:
    """
    Render a compact one-line evaluation summary directly in the chat stream.

    Usage:
        result = pipeline.generate(query)
        st.markdown(result["answer"])
        render_grounding_badge(result.get("ragas"))
    """
    try:
        import streamlit as st
    except ImportError:
        return

    if not ragas_dict:
        return

    faith   = ragas_dict.get("faithfulness",     0.0)
    relev   = ragas_dict.get("answer_relevance",  0.0)
    prec    = ragas_dict.get("context_precision", 0.0)
    overall = ragas_dict.get("overall_score",     0.0)
    grade   = ragas_dict.get("grade",             "N/A")
    supp    = ragas_dict.get("supported_sentences", 0)
    total   = ragas_dict.get("answer_sentences",     0)

    fc = score_colour(faith)
    rc = score_colour(relev)
    gc = grade_colour(grade)

    badge_html = f"""
<div style="
    display:flex; align-items:center; gap:10px; flex-wrap:wrap;
    margin-top:4px; padding:4px 8px;
    background:rgba(0,0,0,0.03); border-radius:6px;
    font-size:0.75rem; color:#64748b; font-family:monospace;
">
  <span title="Faithfulness: how well every claim is grounded in retrieved context">
    📎 Grounding
    <strong style="color:{fc}">{faith*100:.0f}%</strong>
    <span style="color:#94a3b8; font-size:0.7rem">({supp}/{total} sentences)</span>
  </span>
  <span style="color:#cbd5e1">|</span>
  <span title="Answer Relevance: how relevant the answer is to the question">
    🎯 Relevance
    <strong style="color:{rc}">{relev*100:.0f}%</strong>
  </span>
  <span style="color:#cbd5e1">|</span>
  <span title="Context Precision: fraction of retrieved chunks that contributed">
    🗂 Precision
    <strong>{prec*100:.0f}%</strong>
  </span>
  <span style="color:#cbd5e1">|</span>
  <span title="Overall weighted composite score">
    Overall
    <strong style="color:{gc}">{overall*100:.0f}% — {grade}</strong>
  </span>
</div>
"""
    st.markdown(badge_html, unsafe_allow_html=True)


# ── 2. Full RAGAS panel ───────────────────────────────────────────────────────

def render_ragas_panel(
    ragas_dict: Optional[Dict[str, Any]],
    key: str = "ragas",
) -> None:
    """
    Render the full RAGAS evaluation panel inside a Streamlit expander.

    Usage:
        render_ragas_panel(result.get("ragas"), key="turn_3")
    """
    try:
        import streamlit as st
    except ImportError:
        return

    if not ragas_dict:
        st.caption("⚠️ RAGAS evaluation not available for this response.")
        return

    faith   = ragas_dict.get("faithfulness",      0.0)
    relev   = ragas_dict.get("answer_relevance",  0.0)
    prec    = ragas_dict.get("context_precision", 0.0)
    recall  = ragas_dict.get("context_recall")
    sim     = ragas_dict.get("answer_similarity")
    overall = ragas_dict.get("overall_score",     0.0)
    grade   = ragas_dict.get("grade",             "N/A")
    supp    = ragas_dict.get("supported_sentences", 0)
    total   = ragas_dict.get("answer_sentences",     0)
    contribs= ragas_dict.get("chunks_contributed",   0)
    chunks_n= ragas_dict.get("chunks_evaluated",      0)
    details = ragas_dict.get("chunk_details",         [])
    has_gt  = ragas_dict.get("has_ground_truth",      False)

    gc = grade_colour(grade)

    # ── Section 1: Grade banner ───────────────────────────────────────────────
    st.markdown(
        f"""
<div style="
    display:flex;align-items:center;gap:16px;
    padding:12px 16px;border-radius:10px;
    background:linear-gradient(135deg,{gc}18,{gc}08);
    border:1px solid {gc}40;margin-bottom:8px;
">
  <div style="font-size:2rem;font-weight:800;color:{gc}">{overall*100:.1f}%</div>
  <div>
    <div style="font-size:1.1rem;font-weight:700;color:{gc}">{grade}</div>
    <div style="font-size:0.75rem;color:#64748b">
      Weighted composite (faithfulness × 0.40 + relevance × 0.35 + precision × 0.25)
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Section 2: Metric gauges via st.metric() ──────────────────────────────
    st.markdown("#### 📊 Metric Scores")

    base_metrics = [
        ("📎 Faithfulness",  faith,  f"{supp}/{total} sentences grounded"),
        ("🎯 Relevance",     relev,  "Question ↔ Answer semantic similarity"),
        ("🗂 Ctx Precision", prec,   f"{contribs}/{chunks_n} chunks contributed"),
    ]
    extra_metrics: list = []
    if recall is not None:
        extra_metrics.append(("🔁 Ctx Recall", recall, "Ground-truth coverage"))
    if sim is not None:
        extra_metrics.append(("🔗 Answer Sim", sim, "Answer ↔ Ground-truth similarity"))

    all_metrics = base_metrics + extra_metrics
    cols = st.columns(len(all_metrics))
    for col, (label, score, help_text) in zip(cols, all_metrics):
        emoji = score_emoji(score)
        col.metric(
            label=f"{emoji} {label.split(' ', 1)[-1]}",
            value=f"{score*100:.0f}%",
            help=help_text,
        )

    st.markdown("")

    # ── Section 3: Chunk contribution table (cached HTML builder) ─────────────
    if details:
        st.markdown("#### 🔍 Chunk Contributions")
        st.caption(
            f"{supp} of {total} answer sentences grounded · "
            f"{contribs} of {chunks_n} chunks contributed"
        )
        table_html = _build_chunk_table(tuple(json.dumps(d, sort_keys=True) for d in details), total)
        st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("")

    # ── Section 4: Ground-truth note ─────────────────────────────────────────
    if not has_gt:
        st.info(
            "💡 **Context Recall** and **Answer Similarity** are not available "
            "because no ground-truth answer was provided.  "
            "Pass `ground_truth='...'` to `pipeline.generate()` to unlock them.",
            icon="ℹ️",
        )

    # ── Section 5: Export ────────────────────────────────────────────────────
    st.markdown("#### 💾 Export")
    st.download_button(
        label="⬇️  Download RAGAS report (JSON)",
        data=json.dumps(ragas_dict, indent=2, default=str),
        file_name="ragas_report.json",
        mime="application/json",
        key=f"ragas_download_{key}",
    )


# ── Cached chunk table builder ─────────────────────────────────────────────────

try:
    import streamlit as st
    _cache = st.cache_data
except Exception:
    def _cache(fn):                          # no-op when Streamlit not available
        return fn

@_cache
def _build_chunk_table(detail_jsons: tuple, total_sentences: int) -> str:
    """
    Build the per-chunk contribution table HTML.
    Cached by (detail_jsons, total_sentences) so it is not rebuilt on every
    Streamlit re-run when evaluation data is unchanged.
    """
    rows_html = ""
    for raw in detail_jsons:
        cd     = json.loads(raw)
        check  = "✅" if cd.get("contributed") else "❌"
        n_sent = cd.get("sentences_supported", 0)
        sc     = cd.get("score", 0.0)
        src    = cd.get("source", "?")[:40]
        cite   = cd.get("citation", "")
        bar_w  = int(sc * 100)
        bar_c  = score_colour(sc)
        label  = cite or src
        rows_html += f"""
<tr style="border-bottom:1px solid #f1f5f9">
  <td style="padding:4px 8px;font-family:monospace;font-size:0.75rem">{label}</td>
  <td style="padding:4px 8px;text-align:center">{check}</td>
  <td style="padding:4px 8px;text-align:center;color:#64748b;font-size:0.75rem">{n_sent}/{total_sentences}</td>
  <td style="padding:4px 8px;min-width:100px">
    <div style="background:#f1f5f9;border-radius:4px;height:8px">
      <div style="background:{bar_c};width:{bar_w}%;height:8px;border-radius:4px"></div>
    </div>
  </td>
  <td style="padding:4px 8px;font-size:0.75rem;color:{bar_c};font-weight:600">{sc*100:.0f}%</td>
</tr>"""

    return f"""
<table style="width:100%;border-collapse:collapse;font-size:0.8rem">
  <thead>
    <tr style="background:#f8fafc;color:#475569">
      <th style="padding:6px 8px;text-align:left">Chunk / Source</th>
      <th style="padding:6px 8px">Used</th>
      <th style="padding:6px 8px">Sentences</th>
      <th style="padding:6px 8px">Coverage</th>
      <th style="padding:6px 8px">Score</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>"""