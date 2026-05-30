"""
retrieval_graph.py

LangGraph StateGraph for the full retrieval pipeline.

Flow:
  expand_query
       │
       ▼
  retrieve_docs
       │
       ├─(use_rerank=True)──────────────────► rerank_docs
       │                                           │
       ├─(use_rerank=False,                        │
       │  use_compression=True)──► compress_docs ◄─┘
       │                                │
       └─(both False)──► build_context ◄┘
                              │
                             END

  Any node error → handle_error → END

BUG-RET-03 fix: retrieve_docs now correctly routes to build_context
directly when both use_rerank=False and use_compression=False.
Removes the phantom 'compress_or_build' routing target.
"""
from __future__ import annotations
import logging
from langgraph.graph import StateGraph, END

from .state import RetrievalState
from .nodes.expand_node        import expand_query
from .nodes.retrieve_node      import retrieve_docs
from .nodes.rerank_node        import rerank_docs
from .nodes.compress_node      import compress_docs
from .nodes.build_context_node import build_context, handle_error

logger = logging.getLogger(__name__)


def _after_retrieve(s: dict) -> str:
    """Route after retrieve_docs: rerank → compress → context, or short-circuit."""
    if s.get("error"):                  return "handle_error"
    if s.get("use_rerank", True):       return "rerank_docs"
    if s.get("use_compression", False): return "compress_docs"
    return "build_context"


def _after_rerank(s: dict) -> str:
    """Route after rerank_docs: compress if requested, else straight to context."""
    if s.get("error"):                  return "handle_error"
    if s.get("use_compression", False): return "compress_docs"
    return "build_context"


def build_retrieval_graph() -> StateGraph:
    wf = StateGraph(RetrievalState)

    wf.add_node("expand_query",  expand_query)
    wf.add_node("retrieve_docs", retrieve_docs)
    wf.add_node("rerank_docs",   rerank_docs)
    wf.add_node("compress_docs", compress_docs)
    wf.add_node("build_context", build_context)
    wf.add_node("handle_error",  handle_error)

    wf.set_entry_point("expand_query")

    wf.add_conditional_edges(
        "expand_query",
        lambda s: "handle_error" if s.get("error") else "retrieve_docs",
        {"retrieve_docs": "retrieve_docs", "handle_error": "handle_error"},
    )

    wf.add_conditional_edges(
        "retrieve_docs",
        _after_retrieve,
        {
            "rerank_docs":   "rerank_docs",
            "compress_docs": "compress_docs",
            "build_context": "build_context",
            "handle_error":  "handle_error",
        },
    )

    wf.add_conditional_edges(
        "rerank_docs",
        _after_rerank,
        {
            "compress_docs": "compress_docs",
            "build_context": "build_context",
            "handle_error":  "handle_error",
        },
    )

    wf.add_conditional_edges(
        "compress_docs",
        lambda s: "handle_error" if s.get("error") else "build_context",
        {"build_context": "build_context", "handle_error": "handle_error"},
    )

    wf.add_conditional_edges(
        "build_context",
        lambda s: "handle_error" if s.get("error") else END,
        {END: END, "handle_error": "handle_error"},
    )

    wf.add_edge("handle_error", END)
    return wf


retrieval_app = build_retrieval_graph().compile()
