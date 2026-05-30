"""
ragas_panel.py — Standalone Streamlit page / sidebar for RAGAS evaluation.

Entry points
------------
show_ragas_sidebar(ragas_dict_or_pipeline)
    Sidebar summary showing last RAGAS scores + "View Full Evaluation" button.
    Accepts either a ragas_dict directly OR a pipeline object with .last_ragas.

show_ragas_page(ragas_dict)
    Full-page RAGAS dashboard. Shows score trend when session history exists.

record_ragas_history(ragas_dict)
    Append the current result to st.session_state["ragas_history"] (max 20).

render_trend_chart(history)
    Plotly line chart of faithfulness / relevance / precision / overall across
    turns. Exported for use in app.py or custom dashboards.

Changes from original
---------------------
* show_ragas_sidebar() now accepts ragas_dict OR pipeline — decoupled.
* Trend chart upgraded from st.line_chart() (no axis control) to
  Plotly go.Scatter — proper labels, per-metric colours, hover tooltips.
* render_trend_chart() is now exported (was private _render_trend_chart).
* record_ragas_history() is exported and importable from src.ui directly.

Integration in your Streamlit app
----------------------------------
    import streamlit as st
    from src.ui import (
        render_grounding_badge, show_ragas_sidebar,
        show_ragas_page, record_ragas_history,
    )

    # After each assistant message:
    render_grounding_badge(result.get("ragas"))
    record_ragas_history(result.get("ragas"))

    if st.button("📊 Full Evaluation", key=f"ragas_btn_{turn}"):
        st.session_state["ragas_page_data"] = result["ragas"]
        st.session_state["show_ragas_page"] = True

    # Sidebar (call once per render cycle):
    show_ragas_sidebar(result.get("ragas"))

    # Page routing:
    if st.session_state.get("show_ragas_page"):
        show_ragas_page(st.session_state["ragas_page_data"])
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union


# ── Sidebar widget ────────────────────────────────────────────────────────────

def show_ragas_sidebar(
    ragas_or_pipeline: Union[Optional[Dict[str, Any]], Any] = None,
) -> None:
    """
    Show last RAGAS scores in the Streamlit sidebar.

    Parameters
    ----------
    ragas_or_pipeline :
        Either a ragas_dict (from result["ragas"]) OR a pipeline object
        that exposes a .last_ragas attribute.
        Passing None renders a "no evaluation yet" caption.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    # Resolve ragas_dict from either input type
    if ragas_or_pipeline is None:
        ragas: Optional[Dict] = None
    elif isinstance(ragas_or_pipeline, dict):
        ragas = ragas_or_pipeline
    else:
        ragas = getattr(ragas_or_pipeline, "last_ragas", None)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧪 RAGAS Evaluation")

        if not ragas:
            st.caption("No evaluation yet — ask a question first.")
            return

        faith   = ragas.get("faithfulness",     0.0)
        relev   = ragas.get("answer_relevance",  0.0)
        overall = ragas.get("overall_score",     0.0)
        grade   = ragas.get("grade",             "N/A")

        c1, c2 = st.columns(2)
        c1.metric("📎 Grounding", f"{faith*100:.0f}%")
        c2.metric("🎯 Relevance", f"{relev*100:.0f}%")
        st.metric("⭐ Overall",   f"{overall*100:.0f}%  ({grade})")

        if st.button(
            "📊 View Full Evaluation",
            use_container_width=True,
            key="sidebar_ragas_btn",
        ):
            st.session_state["ragas_page_data"] = ragas
            st.session_state["show_ragas_page"] = True
            st.rerun()


# ── Full-page dashboard ───────────────────────────────────────────────────────

def show_ragas_page(ragas_dict: Optional[Dict[str, Any]] = None) -> None:
    """
    Full RAGAS evaluation page.
    Reads from argument or st.session_state["ragas_page_data"].
    Shows Plotly trend chart when session history has ≥ 2 turns.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    from src.ui.components import render_ragas_panel

    data = ragas_dict or st.session_state.get("ragas_page_data")

    st.title("📊 RAGAS Evaluation Dashboard")
    st.caption(
        "Full quality report for the last AI response. "
        "Metrics computed locally — no external API required."
    )

    if not data:
        st.warning("No evaluation data available. Ask a question in Chat mode first.")
        return

    # ── Trend chart ───────────────────────────────────────────────────────────
    history: List[Dict] = st.session_state.get("ragas_history", [])
    if len(history) >= 2:
        st.markdown("#### 📈 Score Trend (last turns)")
        render_trend_chart(history)

    # ── Full panel for current turn ───────────────────────────────────────────
    render_ragas_panel(data, key="full_page")

    # ── Back button ───────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("← Back to Chat", key="ragas_back_btn"):
        st.session_state["show_ragas_page"] = False
        st.rerun()


# ── History tracking ──────────────────────────────────────────────────────────

def record_ragas_history(ragas_dict: Optional[Dict[str, Any]]) -> None:
    """
    Append the current RAGAS result to st.session_state["ragas_history"].
    Keeps the last 20 turns. Safe to call with None.

    Call once per completed chat turn:
        record_ragas_history(result.get("ragas"))
    """
    try:
        import streamlit as st
    except ImportError:
        return

    if not ragas_dict:
        return

    if "ragas_history" not in st.session_state:
        st.session_state["ragas_history"] = []

    st.session_state["ragas_history"].append({
        "turn":         len(st.session_state["ragas_history"]) + 1,
        "faithfulness": ragas_dict.get("faithfulness",      0.0),
        "relevance":    ragas_dict.get("answer_relevance",  0.0),
        "precision":    ragas_dict.get("context_precision", 0.0),
        "overall":      ragas_dict.get("overall_score",     0.0),
        "grade":        ragas_dict.get("grade",             "N/A"),
    })

    # Cap at 20 turns
    if len(st.session_state["ragas_history"]) > 20:
        st.session_state["ragas_history"] = st.session_state["ragas_history"][-20:]


# ── Trend chart (Plotly) ──────────────────────────────────────────────────────

def render_trend_chart(history: List[Dict]) -> None:
    """
    Render a Plotly line chart of RAGAS scores across turns.

    Exported so app.py or custom dashboards can call it directly:
        from src.ui import render_trend_chart
        render_trend_chart(st.session_state["ragas_history"])

    Upgrade from original: was st.line_chart() — no axis labels, no colour.
    Now uses Plotly go.Scatter with:
      - Named traces (Faithfulness, Relevance, Precision, Overall)
      - Per-metric brand colours
      - Hover tooltips showing turn + score
      - Y-axis 0–100% with grid lines
    """
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        # Graceful fallback: plain st.line_chart
        try:
            import streamlit as st
            import pandas as pd
            rows = [{
                "Turn": h["turn"],
                "Faithfulness": h["faithfulness"],
                "Relevance":    h["relevance"],
                "Precision":    h["precision"],
                "Overall":      h["overall"],
            } for h in history]
            st.line_chart(pd.DataFrame(rows).set_index("Turn"), height=220)
        except Exception:
            pass
        return

    turns = [h["turn"] for h in history]

    _TRACES = [
        ("Faithfulness", "faithfulness", "#22c55e"),   # green
        ("Relevance",    "relevance",    "#3b82f6"),   # blue
        ("Precision",    "precision",    "#f59e0b"),   # amber
        ("Overall",      "overall",      "#8b5cf6"),   # purple
    ]

    fig = go.Figure()
    for label, key, colour in _TRACES:
        values = [round(h.get(key, 0.0) * 100, 1) for h in history]
        fig.add_trace(go.Scatter(
            x=turns,
            y=values,
            mode="lines+markers",
            name=label,
            line=dict(color=colour, width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{label}</b><br>Turn %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        xaxis=dict(
            title="Turn",
            tickmode="linear",
            dtick=1,
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(
            title="Score (%)",
            range=[0, 100],
            ticksuffix="%",
            gridcolor="#f1f5f9",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=40, b=40),
        height=260,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)