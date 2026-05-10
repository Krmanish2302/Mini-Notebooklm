"""
components.py  -  Reusable Streamlit UI helpers for Mini NotebookLM.

Public API
----------
render_grounding_badge(ragas_dict)        -> None
    Tiny inline badge shown right under every assistant response.
    Displays: Grounding  87%  |  Relevance  91%  |  Grade: Good
    Colour-coded green / amber / red based on faithfulness score.

render_ragas_panel(ragas_dict)            -> None
    Full RAGAS evaluation panel shown in a Streamlit expander or sidebar.
    Sections:
      1. Score gauges   - faithfulness, relevance, precision, (recall, similarity)
      2. Overall grade  - badge + weighted composite score
      3. Sentence breakdown - per-sentence support table with colour coding
      4. Chunk contributions - per-chunk contribution bar
      5. Export button  - download JSON report

All functions are SAFE to call with ragas_dict=None (they render nothing).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
#  Colour helpers
# ---------------------------------------------------------------------------

def _score_colour(score: float) -> str:
    """Return a CSS hex colour for a 0-1 score."""
    if score >= 0.80:
        return "#22c55e"   # green-500
    if score >= 0.60:
        return "#f59e0b"   # amber-500
    return "#ef4444"       # red-500


def _score_emoji(score: float) -> str:
    if score >= 0.80: return "🟢"
    if score >= 0.60: return "🟡"
    return "🔴"


def _grade_colour(grade: str) -> str:
    return {
        "Excellent": "#22c55e",
        "Good":      "#84cc16",
        "Fair":      "#f59e0b",
        "Poor":      "#ef4444",
    }.get(grade, "#94a3b8")


# ---------------------------------------------------------------------------
#  1.  Inline grounding badge  (shown under every chat message)
# ---------------------------------------------------------------------------

def render_grounding_badge(ragas_dict: Optional[Dict[str, Any]]) -> None:
    """
    Render a compact one-line evaluation summary directly in the chat stream.

    Usage (Streamlit):
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

    faith   = ragas_dict.get("faithfulness",    0.0)
    relev   = ragas_dict.get("answer_relevance", 0.0)
    prec    = ragas_dict.get("context_precision",0.0)
    overall = ragas_dict.get("overall_score",   0.0)
    grade   = ragas_dict.get("grade",           "N/A")
    supp    = ragas_dict.get("supported_sentences", 0)
    total   = ragas_dict.get("answer_sentences",     0)

    fc = _score_colour(faith)
    rc = _score_colour(relev)
    gc = _grade_colour(grade)

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


# ---------------------------------------------------------------------------
#  2.  Full RAGAS panel  (shown in expander or sidebar)
# ---------------------------------------------------------------------------

def render_ragas_panel(ragas_dict: Optional[Dict[str, Any]], key: str = "ragas") -> None:
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

    gc = _grade_colour(grade)

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
    <div style="font-size:0.75rem;color:#64748b">Weighted composite (faithfulness × 0.40 + relevance × 0.35 + precision × 0.25)</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    # ── Section 2: Metric gauges ──────────────────────────────────────────────
    st.markdown("#### 📊 Metric Scores")

    metrics = [
        ("📎 Faithfulness",  faith,  "How well every answer claim is grounded in retrieved context"),
        ("🎯 Relevance",     relev,  "Semantic similarity between the question and the answer"),
        ("🗂 Ctx Precision", prec,   f"Chunks contributing to answer: {contribs}/{chunks_n}"),
    ]
    if recall is not None:
        metrics.append(("🔁 Ctx Recall", recall, "Ground-truth coverage in retrieved context"))
    if sim is not None:
        metrics.append(("🔗 Answer Sim", sim, "Semantic similarity between answer and ground truth"))

    cols = st.columns(len(metrics))
    for col, (label, score, tip) in zip(cols, metrics):
        colour = _score_colour(score)
        emoji  = _score_emoji(score)
        col.markdown(
            f"""
<div title="{tip}" style="
    text-align:center;padding:12px 8px;border-radius:8px;
    background:rgba(0,0,0,0.02);border:1px solid #e2e8f0;
">
  <div style="font-size:1.6rem;font-weight:800;color:{colour}">{score*100:.0f}%</div>
  <div style="font-size:0.72rem;color:#64748b;margin-top:2px">{emoji} {label}</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Section 3: Faithfulness sentence breakdown ────────────────────────────
    if ragas_dict.get("chunk_details"):   # chunk_details carries sentence-level info
        st.markdown("#### 🔍 Faithfulness: Sentence-Level Breakdown")
        st.caption(
            f"{supp} of {total} answer sentences are supported by retrieved context "
            f"(overlap threshold {ragas_dict.get('overlap_threshold', 0.25):.0%})"
            if "overlap_threshold" in ragas_dict
            else f"{supp} of {total} answer sentences grounded."
        )

        # We reconstruct sentence detail from chunk_details if available
        # chunk_details contains per-chunk info; we show that as a table
        st.markdown("**Per-Chunk Contribution**")
        rows_html = ""
        for cd in details:
            check  = "✅" if cd.get("contributed") else "❌"
            n_sent = cd.get("sentences_supported", 0)
            score  = cd.get("score", 0.0)
            src    = cd.get("source", "?")[:40]
            cite   = cd.get("citation", "")
            bar_w  = int(score * 100)
            bar_c  = _score_colour(score)
            rows_html += f"""
<tr style="border-bottom:1px solid #f1f5f9">
  <td style="padding:4px 8px;font-family:monospace;font-size:0.75rem">{cite or src}</td>
  <td style="padding:4px 8px;text-align:center">{check}</td>
  <td style="padding:4px 8px;text-align:center;color:#64748b;font-size:0.75rem">{n_sent}/{total}</td>
  <td style="padding:4px 8px;min-width:100px">
    <div style="background:#f1f5f9;border-radius:4px;height:8px">
      <div style="background:{bar_c};width:{bar_w}%;height:8px;border-radius:4px"></div>
    </div>
  </td>
  <td style="padding:4px 8px;font-size:0.75rem;color:{bar_c};font-weight:600">{score*100:.0f}%</td>
</tr>"""

        table_html = f"""
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
    json_str = json.dumps(ragas_dict, indent=2, default=str)
    st.download_button(
        label="⬇️  Download RAGAS report (JSON)",
        data=json_str,
        file_name="ragas_report.json",
        mime="application/json",
        key=f"ragas_download_{key}",
    )
