"""
state.py — TypedDict that flows through every retrieval LangGraph node.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class RetrievalState(TypedDict, total=False):
    # ── inputs ────────────────────────────────────────────────────────────────
    query:            str
    vectorstore_path: str
    top_k:            int
    use_rerank:       bool
    use_reordering:   bool
    do_expand:        bool   # renamed from expand_query (clashed with node fn)
    # FIX #2: source_ids was missing — API source filtering was dead code in the graph
    source_ids:       Optional[List[str]]
    mode:             str    # "chat" | "deep_research" | "study"

    # ── intermediate ──────────────────────────────────────────────────────────────
    expanded_queries: List[str]          # after expand_node
    documents:        List[Document]     # after retrieve_node
    reranked_docs:    List[Document]     # after rerank_node
    reordered_docs:   List[Document]     # after reorder_node
    reordered_parents: List[Dict[str, Any]] # resolved parents, reordered
    graph_context:    List[Dict[str, Any]]  # retrieved graph concepts/edges for Study mode

    # ── output ─────────────────────────────────────────────────────────────────
    context:          str                # formatted context string for LLM
    metadata:         Dict[str, Any]     # retrieval stats

    # ── error handling ─────────────────────────────────────────────────────────────
    error:            Optional[str]
    failed_node:      Optional[str]

