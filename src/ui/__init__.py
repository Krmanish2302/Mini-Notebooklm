"""
src/ui — Streamlit UI components for Mini NotebookLM.

Public API:
    render_grounding_badge(ragas_dict)        → inline badge under each chat turn
    render_ragas_panel(ragas_dict, key)       → full evaluation panel (expander)
    show_ragas_sidebar(ragas_dict|pipeline)   → sidebar summary + nav button
    show_ragas_page(ragas_dict)               → full-page evaluation dashboard
    record_ragas_history(ragas_dict)          → append to session_state history
    render_trend_chart(history)               → Plotly line chart of score trends

Usage:
    from src.ui import render_grounding_badge, show_ragas_sidebar
    from src.ui import show_ragas_page, record_ragas_history
"""
from .components  import render_grounding_badge, render_ragas_panel   # noqa: F401
from .ragas_panel import (                                              # noqa: F401
    show_ragas_sidebar,
    show_ragas_page,
    record_ragas_history,
    render_trend_chart,
)

__all__ = [
    "render_grounding_badge",
    "render_ragas_panel",
    "show_ragas_sidebar",
    "show_ragas_page",
    "record_ragas_history",
    "render_trend_chart",
]