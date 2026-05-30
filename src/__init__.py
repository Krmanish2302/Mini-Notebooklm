"""
src/__init__.py — Top-level package for Mini NotebookLM.

Exposes the one-line public API:

    from src import MiniNotebookLM

    nb = MiniNotebookLM()
    nb.ingest("lecture_notes.pdf")
    result = nb.ask("What is attention?")
    print(result["answer"])
    print(result["citations"])

Sub-packages
------------
src.ingestion      — document loaders, chunkers, embedders, vector store
src.retrieval      — hybrid retriever (dense + BM25 + RRF), HyDE rewriter
src.generation     — LangGraph pipeline: prompt → LLM → parse → cite
src.ui             — Streamlit components (badge, RAGAS panel, sidebar)
src.evaluation     — RAGAS evaluator
"""
from __future__ import annotations

# ── Sub-package exports ────────────────────────────────────────────────────────
from src.generation import (            # noqa: F401
    generate,
    generation_app,
    GenerationState,
    PersonaConfig,
    LLMRegistry,
)
from src.ui import (                    # noqa: F401
    render_grounding_badge,
    render_ragas_panel,
    show_ragas_sidebar,
    show_ragas_page,
    record_ragas_history,
    render_trend_chart,
)

# ── Convenience re-export of the full pipeline ─────────────────────────────────
from src.master_pipeline import MiniNotebookLM   # noqa: F401

__version__ = "0.4.0"

__all__ = [
    # pipeline
    "MiniNotebookLM",
    # generation
    "generate",
    "generation_app",
    "GenerationState",
    "PersonaConfig",
    "LLMRegistry",
    # ui
    "render_grounding_badge",
    "render_ragas_panel",
    "show_ragas_sidebar",
    "show_ragas_page",
    "record_ragas_history",
    "render_trend_chart",
    "__version__",
]