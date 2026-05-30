"""
state.py — TypedDict that flows through every retrieval LangGraph node.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document


class RetrievalState(TypedDict, total=False):
    # ── inputs ──────────────────────────────────────────────────────────
    query:            str
    vectorstore_path: str
    top_k:            int
    use_rerank:       bool
    use_compression:  bool
    do_expand:        bool   # renamed from expand_query (clashed with node fn)

    # ── intermediate ────────────────────────────────────────────────────
    expanded_queries: List[str]          # after expand_node
    documents:        List[Document]     # after retrieve_node
    reranked_docs:    List[Document]     # after rerank_node
    compressed_docs:  List[Document]     # after compress_node

    # ── output ──────────────────────────────────────────────────────────
    context:          str                # formatted context string for LLM
    metadata:         Dict[str, Any]     # retrieval stats

    # ── error handling ───────────────────────────────────────────────────
    error:            Optional[str]
    failed_node:      Optional[str]
