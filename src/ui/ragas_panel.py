"""
ragas_panel.py  —  Standalone Streamlit page / sidebar for RAGAS evaluation.

This module exposes two entry points:

  show_ragas_sidebar(pipeline):
      Renders a collapsible sidebar section showing the last RAGAS result
      from pipeline.last_ragas.  Call once per app render cycle.

  show_ragas_page(ragas_dict):
      Full-page RAGAS dashboard — called when the user clicks
      "View Full Evaluation" from the chat.
      Shows historical trend if st.session_state contains prior turns.

Integration in your Streamlit app
----------------------------------
    # In your main app file:
    import streamlit as st
    from src.ui.ragas_panel import show_ragas_sidebar, show_ragas_page
    from src.ui.components import render_grounding_badge

    # After each assistant message:
    render_grounding_badge(result.get("ragas"))
    if st.button("📊 View Full Evaluation", key=f"ragas_btn_{turn}"):
        st.session_state["ragas_page"] = result["ragas"]
        st.session_state["show_ragas_page"] = True

    # In sidebar:
    show_ragas_sidebar(pipeline)

    # Page routing:
    if st.session_state.get("show_ragas_page"):
        show_ragas_page(st.session_state["ragas_page"])
        if st.button("← Back to Chat"):
            st.session_state["show_ragas_page"] = False
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
#  Sidebar widget  (always visible in chat)
# ---------------------------------------------------------------------------

def show_ragas_sidebar(pipeline) -> None:
    """
    Show last RAGAS scores in the Streamlit sidebar.
    Attach a button that navigates to the full RAGAS page.
    Safe to call when pipeline.last_ragas is None.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    ragas = getattr(pipeline, "last_ragas", None)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧪 RAGAS Evaluation")

        if not ragas:
            st.caption("No evaluation yet — ask a question first.")
            return

        faith   = ragas.get("faithfulness",      0.0)
        relev   = ragas.get("answer_relevance",  0.0)
        overall = ragas.get("overall_score",     0.0)
        grade   = ragas.get("grade",             "N/A")

        # Compact metric display
        c1, c2 = st.columns(2)
        c1.metric("Grounding", f"{faith*100:.0f}%")
        c2.metric("Relevance", f"{relev*100:.0f}%")
        st.metric("Overall", f"{overall*100:.0f}%  ({grade})")

        if st.button("📊 View Full Evaluation", use_container_width=True, key="sidebar_ragas_btn"):
            st.session_state["ragas_page_data"]  = ragas
            st.session_state["show_ragas_page"]  = True
            st.rerun()


# ---------------------------------------------------------------------------
#  Full-page dashboard
# ---------------------------------------------------------------------------

def show_ragas_page(ragas_dict: Optional[Dict[str, Any]] = None) -> None:
    """
    Full RAGAS evaluation page.
    Call when the user navigates to the evaluation view.
    Reads from argument or st.session_state["ragas_page_data"].
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
        st.warning("No evaluation data available.  Ask a question in Chat mode first.")
        return

    # ── History trend (if session_state stores prior turns) ──────────────────
    history: List[Dict] = st.session_state.get("ragas_history", [])
    if len(history) >= 2:
        st.markdown("#### 📈 Score Trend (last turns)")
        _render_trend_chart(history)

    # ── Full panel for current turn ──────────────────────────────────────────
    render_ragas_panel(data, key="full_page")

    # ── Back button ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("← Back to Chat", key="ragas_back_btn"):
        st.session_state["show_ragas_page"] = False
        st.rerun()


# ---------------------------------------------------------------------------
#  History tracking helper
# ---------------------------------------------------------------------------

def record_ragas_history(ragas_dict: Optional[Dict[str, Any]]) -> None:
    """
    Append the current ragas result to the session history list.
    Call once per completed turn:
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
    # Keep last 20 turns
    st.session_state["ragas_history"].append({
        "turn":         len(st.session_state["ragas_history"]) + 1,
        "faithfulness": ragas_dict.get("faithfulness",     0.0),
        "relevance":    ragas_dict.get("answer_relevance", 0.0),
        "precision":    ragas_dict.get("context_precision",0.0),
        "overall":      ragas_dict.get("overall_score",    0.0),
        "grade":        ragas_dict.get("grade",            "N/A"),
    })
    if len(st.session_state["ragas_history"]) > 20:
        st.session_state["ragas_history"] = st.session_state["ragas_history"][-20:]


# ---------------------------------------------------------------------------
#  Trend chart (internal)
# ---------------------------------------------------------------------------

def _render_trend_chart(history: List[Dict]) -> None:
    """Render a simple Streamlit line chart of scores across turns."""
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        return

    rows = []
    for h in history:
        rows.append({
            "Turn":        h["turn"],
            "Faithfulness": h["faithfulness"],
            "Relevance":    h["relevance"],
            "Precision":    h["precision"],
            "Overall":      h["overall"],
        })
    df = pd.DataFrame(rows).set_index("Turn")
    st.line_chart(df, height=200)
